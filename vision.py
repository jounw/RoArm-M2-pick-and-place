"""VLM object localization for the eye-in-hand camera.

Given a frame and a natural-language description, ask a vision-language model
for the target's bounding box, normalized to 0-1000 (top-left/bottom-right),
matching the convention used throughout this project's prompts.

Three backends (VLM_BACKEND env var or the `backend` argument):
  gemini  -- Gemini Robotics-ER, grounding-trained: precise boxes (default when
             a Gemini key is available)
  openai  -- GPT-4o family: understands language well but its box coordinates
             are unreliable (kept as fallback)
  molmo   -- fully local, no API key/network: a quantized Molmo2-4B GGUF served
             by Ollama (http://localhost:11434), OpenAI-compatible API.
             CAUTION: the only available 4B GGUF build of Molmo2 is an
             unmaintained community conversion; its author warns the vision
             (mmproj) path degrades and is not benchmarked. Box coordinates
             from this backend are unverified -- test against real objects
             before trusting it for unattended picks. Model pull:
               ollama pull hf.co/reubk/Molmo2-4B-GGUF:Q4_K_M
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from openai import OpenAI

VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o-mini")
# openai backend: task planning needs the stronger model (gpt-4o-mini confuses
# "where X was" with "where X started"); it is also no slower on this prompt
PLAN_MODEL = os.getenv("PLAN_MODEL", "gpt-4o")
# gemini backend: thinking budget for planning (0 = none; more = slower)
PLAN_THINKING = int(os.getenv("PLAN_THINKING", "0"))  # measured: 0 is fastest AND most consistent
GEMINI_MODEL = os.getenv("GEMINI_VLM_MODEL", "gemini-3.5-flash")
# Keys come from the environment, or from local key files next to this file.
# Both files are gitignored -- never commit keys.
API_KEY_FILE = Path(__file__).with_name("api.txt")
GEMINI_KEY_FILE = Path(__file__).with_name("gemini_api.txt")

# Local Molmo backend via Ollama's OpenAI-compatible endpoint -- no API key.
MOLMO_BASE_URL = os.getenv("MOLMO_BASE_URL", "http://localhost:11434/v1")
MOLMO_MODEL = os.getenv("MOLMO_MODEL", "hf.co/reubk/Molmo2-4B-GGUF:Q4_K_M")

# Language of everything the robot SAYS back (scene answers, chat). All of
# route()/ask()/plan_pick_place() honour this so TTS gets one language.
ANSWER_LANG = os.getenv("VLM_ANSWER_LANG", "ko")
_LANG_NAME = {"ko": "Korean", "en": "English"}


def _read_key(env_var: str, key_file: Path) -> str:
    key = os.getenv(env_var, "").strip()
    if not key and key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
    return key


def _api_key() -> str:
    key = _read_key("OPENAI_API_KEY", API_KEY_FILE)
    if not key:
        raise RuntimeError("no OpenAI API key: set OPENAI_API_KEY or put the key in api.txt")
    return key


def _gemini_key() -> str:
    key = _read_key("GEMINI_API_KEY", GEMINI_KEY_FILE)
    if not key:
        raise RuntimeError("no Gemini API key: set GEMINI_API_KEY or put the key in gemini_api.txt")
    return key


@dataclass
class Detection:
    answer: str
    box: tuple[int, int, int, int] | None  # x1, y1, x2, y2 normalized 0-1000

    @property
    def center(self) -> tuple[int, int] | None:
        if self.box is None:
            return None
        x1, y1, x2, y2 = self.box
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def height(self) -> int:
        if self.box is None:
            return 0
        return self.box[3] - self.box[1]


def _client() -> OpenAI:
    return OpenAI(api_key=_api_key())


def _molmo_client() -> OpenAI:
    # Ollama's OpenAI-compatible server ignores the key but the client requires
    # a non-empty string.
    return OpenAI(api_key="ollama", base_url=MOLMO_BASE_URL)


# Frames are downscaled before upload: a 1920x1080 Record3D frame is ~450KB,
# at 768px it is ~50KB and every VLM call gets 0.5-1.5s faster. Boxes are
# normalized 0-1000 so nothing downstream changes. 0 disables.
VLM_MAX_SIDE = int(os.getenv("VLM_MAX_SIDE", "768"))


def _image_bytes(image_path: str) -> bytes:
    if VLM_MAX_SIDE <= 0:
        with open(image_path, "rb") as f:
            return f.read()
    import cv2

    im = cv2.imread(image_path)
    if im is None:
        with open(image_path, "rb") as f:
            return f.read()
    h, w = im.shape[:2]
    scale = VLM_MAX_SIDE / max(h, w)
    if scale < 1.0:
        im = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        with open(image_path, "rb") as f:
            return f.read()
    return buf.tobytes()


def _encode_image(image_path: str) -> str:
    encoded = base64.b64encode(_image_bytes(image_path)).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _strip_markdown(text: str) -> str:
    return re.sub(r"```(?:json)?|```", "", text).strip()


def _sanitize_box(box, answer: str) -> Detection:
    """Validate/clamp a 0-1000 (x1,y1,x2,y2) box into a Detection."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return Detection(answer=answer, box=None)
    try:
        x1, y1, x2, y2 = (int(float(v)) for v in box)
    except (TypeError, ValueError):
        return Detection(answer=answer, box=None)
    x1, x2 = sorted((max(0, min(1000, x1)), max(0, min(1000, x2))))
    y1, y2 = sorted((max(0, min(1000, y1)), max(0, min(1000, y2))))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return Detection(answer=answer, box=None)
    return Detection(answer=answer, box=(x1, y1, x2, y2))


def _gemini_generate(image_path: str, prompt: str, thinking_budget: int = 0, temperature: float = 0.0,
                     audio_wav: bytes | None = None) -> str:
    """One image(+optional audio)+text call to Gemini; returns the raw text reply."""
    image_b64 = base64.b64encode(_image_bytes(image_path)).decode("ascii")
    parts = [{"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]
    if audio_wav is not None:
        parts.append({"inline_data": {"mime_type": "audio/wav", "data": base64.b64encode(audio_wav).decode("ascii")}})
    parts.append({"text": prompt})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"thinkingConfig": {"thinkingBudget": thinking_budget}, "temperature": temperature},
    }
    resp = requests.post(url, json=body, headers={"x-goog-api-key": _gemini_key()}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        return f"unexpected Gemini response: {str(data)[:200]}"


def _locate_gemini(image_path: str, description: str) -> Detection:
    """Gemini Robotics-ER: grounding-trained, returns box_2d as [ymin,xmin,ymax,xmax] 0-1000."""
    prompt = (
        f"Find this object in the image: {description}\n"
        'Return JSON only: [{"box_2d": [ymin, xmin, ymax, xmax], "label": "<what it is>"}]\n'
        "box_2d is normalized to 0-1000. Return [] if the object is not visible. "
        "Return at most the single best match."
    )
    # spatial grounding needs no long deliberation; thinking budget 0 keeps latency low
    raw = _strip_markdown(_gemini_generate(image_path, prompt, thinking_budget=0))

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Gemini occasionally emits ALMOST-json (e.g. a label without its key:
        # {"box_2d": [...], "the blue paper"}). The box coordinates are still
        # there, so salvage them by regex instead of failing the detection.
        m = re.search(r'"box_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', raw)
        if m:
            ymin, xmin, ymax, xmax = (float(v) for v in m.groups())
            return _sanitize_box([xmin, ymin, xmax, ymax], raw)
        return Detection(answer=raw, box=None)

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        return Detection(answer="not visible", box=None)

    first = parsed[0]
    answer = str(first.get("label", "")).strip()
    box_2d = first.get("box_2d")
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return Detection(answer=answer, box=None)
    ymin, xmin, ymax, xmax = box_2d  # Gemini order -> our (x1, y1, x2, y2)
    return _sanitize_box([xmin, ymin, xmax, ymax], answer)


def _locate_openai(image_path: str, description: str) -> Detection:
    prompt = f"""
Find this object in the image: {description}

Return JSON only, no other text:
{{"answer": "short Korean description of what you see", "box": [x1, y1, x2, y2]}}

Rules:
- box coordinates are normalized integers 0-1000, x1,y1 = top-left, x2,y2 = bottom-right.
- If the object is not visible, return "box": null.
""".strip()

    client = _client()
    response = client.responses.create(
        model=VLM_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _encode_image(image_path)},
                ],
            }
        ],
    )

    raw = _strip_markdown(response.output_text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Detection(answer=raw, box=None)

    answer = str(data.get("answer", "")).strip()
    return _sanitize_box(data.get("box"), answer)


def _molmo_chat(image_path: str, prompt: str) -> str:
    """One image+text call to the local Ollama-served Molmo model."""
    client = _molmo_client()
    response = client.chat.completions.create(
        model=MOLMO_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _encode_image(image_path)}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _locate_molmo(image_path: str, description: str) -> Detection:
    """Local Molmo2-4B (via Ollama). Same 0-1000 (x1,y1,x2,y2) box convention
    as the other backends. See the module docstring for the accuracy caveat
    on this backend's vision path."""
    prompt = f"""
Find this object in the image: {description}

Return JSON only, no other text:
{{"answer": "short description of what you see", "box": [x1, y1, x2, y2]}}

Rules:
- box coordinates are normalized integers 0-1000, x1,y1 = top-left, x2,y2 = bottom-right.
- If the object is not visible, return "box": null.
""".strip()
    raw = _strip_markdown(_molmo_chat(image_path, prompt))
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Detection(answer=raw, box=None)
    answer = str(data.get("answer", "")).strip()
    return _sanitize_box(data.get("box"), answer)


def _pick_backend(backend: str | None) -> str:
    backend = (backend or os.getenv("VLM_BACKEND", "")).strip().lower()
    if not backend:
        backend = "gemini" if _read_key("GEMINI_API_KEY", GEMINI_KEY_FILE) else "openai"
    if backend not in ("gemini", "openai", "molmo"):
        raise ValueError(f"unknown VLM backend: {backend!r} (use 'gemini', 'openai', or 'molmo')")
    return backend


def locate(image_path: str, description: str, backend: str | None = None) -> Detection:
    """Find `description` in the image. Backend: arg > VLM_BACKEND env > auto
    (gemini if a Gemini key is available, else openai)."""
    picked = _pick_backend(backend)
    if picked == "gemini":
        return _locate_gemini(image_path, description)
    if picked == "molmo":
        return _locate_molmo(image_path, description)
    return _locate_openai(image_path, description)


def locate_many(image_path: str, descriptions: list[str], backend: str | None = None) -> list[Detection]:
    """Find several things in ONE image with ONE call (gemini backend); returns
    one Detection per description, in the same order (box=None if not visible).
    Non-gemini backends fall back to one locate() call per description."""
    picked = _pick_backend(backend)
    if picked != "gemini":
        return [locate(image_path, d, backend) for d in descriptions]

    numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(descriptions))
    prompt = (
        "Find each of these in the image:\n"
        f"{numbered}\n"
        'Return JSON only: [{"id": <item number from the list>, "box_2d": [ymin, xmin, ymax, xmax], "label": "<what it is>"}]\n'
        "box_2d is normalized to 0-1000. Include exactly one entry per listed item; "
        "omit an item only if it is not visible in the image."
    )
    raw = _strip_markdown(_gemini_generate(image_path, prompt, thinking_budget=0))

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # combined answer unusable -- fall back to per-item calls rather than failing all
        return [locate(image_path, d, backend) for d in descriptions]
    if isinstance(parsed, dict):
        parsed = [parsed]

    results = [Detection(answer="not visible", box=None) for _ in descriptions]
    if isinstance(parsed, list):
        for i, item in enumerate(parsed):
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id")
            try:
                idx = int(raw_id) - 1
            except (TypeError, ValueError):
                idx = i
            if not 0 <= idx < len(descriptions):
                continue
            b = item.get("box_2d")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                ymin, xmin, ymax, xmax = b
                results[idx] = _sanitize_box([xmin, ymin, xmax, ymax], str(item.get("label", "")).strip())
    return results


def _answer_lang() -> str:
    return _LANG_NAME.get(ANSWER_LANG, ANSWER_LANG)


def _vlm_text(image_path: str, prompt: str, thinking_budget: int = 0, backend: str | None = None, model: str | None = None) -> str:
    """One image+text call on whichever backend is selected; raw text reply
    with markdown fences stripped. Shared by every non-grounding call below."""
    picked = _pick_backend(backend)
    if picked == "gemini":
        return _strip_markdown(_gemini_generate(image_path, prompt, thinking_budget=thinking_budget))
    if picked == "molmo":
        return _strip_markdown(_molmo_chat(image_path, prompt))
    client = _client()
    response = client.responses.create(
        model=model or VLM_MODEL,
        temperature=0.0,  # planning/routing must be repeatable, not creative
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": _encode_image(image_path)},
                ],
            }
        ],
    )
    return _strip_markdown(response.output_text)


def route(image_path: str, utterance: str, backend: str | None = None) -> dict:
    """Route a spoken utterance: a pick command or a question about the scene.

    Returns {"action": "pick", "target": ...} or {"action": "say", "text": ...}.
    """
    prompt = (
        f'The user said to the robot arm: "{utterance}"\n'
        "Using the attached camera view, respond with JSON only:\n"
        '- if it is a command to pick up/move an object: {"action": "pick", "target": "<English description of the object to pick>"}\n'
        f'- otherwise (a question or general conversation): {{"action": "say", "text": "<concise answer in {_answer_lang()}, at most 2 sentences>"}}\n'
    )
    raw = _vlm_text(image_path, prompt, thinking_budget=128, backend=backend)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("action") in ("pick", "say"):
            return data
    except json.JSONDecodeError:
        pass
    return {"action": "say", "text": raw}


def _clean_str(v) -> str | None:
    return str(v).strip() if isinstance(v, str) and v.strip() else None


def plan_pick_place(
    image_path: str,
    utterance: str,
    backend: str | None = None,
    history: list[str] | None = None,
    holding: str | None = None,
    memory: str | None = None,
    spots: str | None = None,
    audio_wav: bytes | None = None,
) -> dict:
    """Turn a natural-language command (likely Korean) into an ordered list of
    single-object pick/place steps, using the attached top-down camera view.

    `history` is the recent dialogue/action transcript (most recent last) and
    `holding` describes the object currently in the gripper (None if empty),
    so references like "그거", "아까 거기", "하나 더" can be resolved.
    `spots` describes the named place spots on the table (default: handwritten
    numbers 1-5; with ArUco floor markers: "marker <id>" names).
    `audio_wav`: the spoken command as wav bytes instead of `utterance` text
    (gemini backend only) -- transcription and planning happen in ONE call and
    the result also carries "transcript".

    Returns {"steps": [step, ...], "answer": <str|None>, "error": <str|None>}
    where each step is
        {"pick": <English visual description | None>,
         "place": {"kind": "spot",   "desc": "<English visual description of the target spot>"}
               | {"kind": "where_was", "of": "<pick description>"}  # where that object is/was when this command started (swaps, "X 자리에")
               | {"kind": "origin", "of":   "<pick description>"}  # where that object was when FIRST seen this session ("원래 자리로")
               | {"kind": "last_place"}    # where the robot most recently put something ("아까 거기")
               | {"kind": "free"}          # any empty patch of table (temporary parking)
               | None}                     # grab and hold, no placing
    Examples:
      "X 잡아서 3에 놔"      -> [{pick X, place spot 3}]
      "X 잡아"               -> [{pick X, place None}]
      "3에 놔" (holding)     -> [{pick None, place spot 3}]
      "X는 1에, Y는 2에 놔"  -> [{pick X, spot 1}, {pick Y, spot 2}]
      "X랑 Y 자리 바꿔"      -> [{pick X, free}, {pick Y, where_was X}, {pick X, where_was Y}]
      "X 원래 자리로 돌려놔" -> [{pick X, origin of X}]
      "테이블에 뭐 있어?"    -> steps [], answer "<answer>"
    The SAME object must use the IDENTICAL description string in every step,
    because the executor keys original positions by that string.
    """
    context = ""
    if memory:
        context += f"Session memory (positions are robot-frame mm, x,y):\n{memory}\n"
    if history:
        transcript = "\n".join(history[-12:])
        context += f"Conversation and actions so far (most recent last):\n{transcript}\n"
    context += (
        f"The robot's gripper is currently holding: {holding}.\n" if holding
        else "The robot's gripper is currently empty.\n"
    )
    if audio_wav is not None:
        said = ("The user's command to the robot arm is in the attached AUDIO clip (Korean speech). "
                "First transcribe it exactly, then plan.\n")
    else:
        said = f'The user now says to the robot arm (likely in Korean): "{utterance}"\n'
    prompt = (
        f"{context}"
        f"{said}"
        "The attached image is a top-down camera view of the table. "
        + (spots or "Large handwritten numbers (1 to 5) are written on the table and serve as named place locations.")
        + "\n"
        "You are the robot's task planner. The robot has ONE gripper and can hold ONE "
        "object at a time; every step is: pick up one object (or keep the held one) and "
        "put it somewhere. Break the request into an ordered list of such steps.\n"
        "Use the conversation context to resolve references such as '그거', '그것', "
        "'아까 거기', '같은 곳', '하나 더', '다시'.\n"
        "Respond with JSON only, no other text:\n"
        + ('{"transcript": "<exact transcript of the audio>", ' if audio_wav is not None else "{")
        + '"steps": [ {"pick": <string or null>, "place": <place or null>}, ... ], '
        '"answer": <string or null>}\n'
        '- "pick": an English VISUAL description of the object to pick up, phrased for a '
        'visual detector (e.g. "the black rectangular object", "the red pen"). Use the '
        "IDENTICAL string whenever the same object appears in several steps, and if the "
        "object is already listed in the session memory, reuse EXACTLY the string listed "
        "there (that is how its remembered positions are looked up). null only "
        "when the step places the object the robot is ALREADY holding (e.g. the user just "
        "says '4에 놔' while holding something) -- in that case it must be the first step.\n"
        '- "place" is one of:\n'
        '    {"kind": "spot", "desc": "<English visual description of where to put it, e.g. '
        'the large handwritten number 3 on the table, or next to the blue cup>"}\n'
        '    {"kind": "where_was", "of": "<pick string of another object>"}  -- the spot where '
        "that object is NOW / was when this command started (\"X 자리에 놔\", swaps). The "
        "object itself must be moved away by an earlier step of this plan first.\n"
        '    {"kind": "origin", "of": "<pick string>"}  -- the spot where that object was when it '
        "was FIRST seen in this session, per the memory ('원래 자리', '원래대로 돌려놔'). "
        '"X를 원래 자리로" is {"pick": X, "place": {"kind": "origin", "of": X}}.\n'
        '    {"kind": "last_place"}  -- EXACTLY the spot where the robot most recently put '
        "something ('아까 거기', '같은 곳에', '방금 놓은 자리'). It is usually still occupied by "
        "that object, so for 'next to / beside' ('옆에') use a spot description such as "
        '"next to the black object" instead.\n'
        '    {"kind": "free"}  -- any empty patch of table. ONLY for temporary parking during a '
        "swap or when the destination is currently occupied by an object that must itself move. "
        "Never park an object that just goes to an empty spot.\n"
        "    null  -- the user only asked to grab/hold, no placing\n"
        "- Two objects cannot occupy the same place, so never place onto a spot that is "
        "still occupied: move the occupant away first. To SWAP A and B: "
        '[{"pick": A, "place": {"kind": "free"}}, {"pick": B, "place": {"kind": "where_was", "of": A}}, '
        '{"pick": A, "place": {"kind": "where_was", "of": B}}]. '
        "To put A where B is (B not needed): first move B to a free spot, then A to where_was B. "
        "A numbered spot is an AREA: several objects may be put there, so a numbered spot is "
        "never 'occupied' and nothing has to be moved off it first. Only where_was/origin/"
        "last_place are exact points that must be vacated by an earlier step if an object is "
        "still there. Use the fewest steps possible and never move an object to where it "
        "already is.\n"
        f'- "answer": if the utterance is NOT a manipulation command but a question or '
        "conversation (what is on the table, where something is, general chat), answer it "
        f"concisely in {_answer_lang()} (1-2 sentences) using the image and return "
        '"steps": []. Otherwise null.\n'
        "- If the request is ambiguous or impossible with what is visible, or is ALREADY "
        "satisfied (the memory says the object is at the requested place), return "
        f'"steps": [] and say so briefly in {_answer_lang()} in "answer". Never substitute a '
        "different object for the one the user named.\n"
    )
    if audio_wav is not None:
        if _pick_backend(backend) != "gemini":
            raise ValueError("audio planning needs the gemini backend")
        raw = _strip_markdown(_gemini_generate(image_path, prompt, thinking_budget=PLAN_THINKING, audio_wav=audio_wav))
    else:
        raw = _vlm_text(image_path, prompt, thinking_budget=PLAN_THINKING, backend=backend, model=PLAN_MODEL)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"steps": [], "answer": None, "error": raw, "transcript": None}
    if not isinstance(data, dict):
        return {"steps": [], "answer": None, "error": raw, "transcript": None}

    steps: list[dict] = []
    for item in data.get("steps") or []:
        if not isinstance(item, dict):
            continue
        pick = _clean_str(item.get("pick"))
        place_raw = item.get("place")
        place: dict | None = None
        if isinstance(place_raw, str):  # tolerate a bare description
            d = _clean_str(place_raw)
            place = {"kind": "spot", "desc": d} if d else None
        elif isinstance(place_raw, dict):
            kind = _clean_str(place_raw.get("kind")) or "spot"
            if kind in ("origin", "where_was") and _clean_str(place_raw.get("of")):
                place = {"kind": kind, "of": _clean_str(place_raw.get("of"))}
            elif kind == "free":
                place = {"kind": "free"}
            elif kind == "last_place":
                place = {"kind": "last_place"}
            elif _clean_str(place_raw.get("desc")):
                place = {"kind": "spot", "desc": _clean_str(place_raw.get("desc"))}
        if pick is None and place is None:
            continue
        steps.append({"pick": pick, "place": place})
    return {"steps": steps, "answer": _clean_str(data.get("answer")), "error": None,
            "transcript": _clean_str(data.get("transcript"))}


def parse_pick_place(
    image_path: str,
    utterance: str,
    backend: str | None = None,
    history: list[str] | None = None,
    holding: str | None = None,
) -> dict:
    """Single-step view of plan_pick_place(): {"pick", "place", "answer", "error"}
    with "place" flattened to a description string (first step only).
    Kept for callers that only handle one pick/place at a time."""
    plan = plan_pick_place(image_path, utterance, backend=backend, history=history, holding=holding)
    first = plan["steps"][0] if plan["steps"] else {"pick": None, "place": None}
    place = first.get("place")
    return {
        "pick": first.get("pick"),
        "place": place.get("desc") if isinstance(place, dict) and place.get("kind") == "spot" else None,
        "answer": plan.get("answer"),
        "error": plan.get("error"),
    }


def ask(image_path: str, question: str, backend: str | None = None) -> str:
    """Free-form question about the image ("what do you see?"), answered in
    the configured answer language (VLM_ANSWER_LANG, default Korean)."""
    prompt = f"{question}\n\nAnswer concisely in {_answer_lang()} (at most 2-3 sentences)."
    # scene Q&A benefits from a little thinking, unlike box grounding
    return _vlm_text(image_path, prompt, thinking_budget=256, backend=backend)


def check_grasp(image_path: str, description: str, backend: str | None = None) -> tuple[bool | None, str]:
    """After a pick, with the arm lifted: is `description` in the gripper, or
    still lying on the table? Returns (held, reason); held is None when the
    model could not tell."""
    prompt = (
        "The attached image is a top-down camera view of a table with a robot arm. The robot just tried to "
        f"pick up this object: {description}. The arm is now lifted, so a held object hangs under the gripper, "
        "often partly hidden by the arm, and looks displaced from where it was.\n"
        "Decide whether the grasp succeeded. Answer held=false ONLY if you clearly see the whole object lying "
        "flat on the table surface, separate from the gripper fingers (e.g. the gripper is somewhere else, or "
        "the object sits beside/under the fingers with a visible gap). If the object is at the gripper, touching "
        "the fingers, partly hidden by the arm, or you are not sure, answer held=true. If the object is not "
        "visible at all, answer null.\n"
        'Respond with JSON only: {"held": true | false | null, "reason": "<one short sentence>"}'
    )
    raw = _vlm_text(image_path, prompt, thinking_budget=0, backend=backend)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None, raw[:120]
    if not isinstance(data, dict):
        return None, raw[:120]
    held = data.get("held")
    return (held if isinstance(held, bool) else None), str(data.get("reason", "")).strip()
