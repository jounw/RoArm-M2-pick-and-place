"""RoArm-M2 low-level driver.

JSON command set per official Waveshare docs (waveshareteam/roarm_m2, json_cmd.h):
  T=100  CMD_MOVE_INIT           {"T":100}
  T=104  CMD_XYZT_GOAL_CTRL      {"T":104,"x":..,"y":..,"z":..,"t":..,"spd":..}   (interpolated)
  T=1041 CMD_XYZT_DIRECT_CTRL    {"T":1041,"x":..,"y":..,"z":..,"t":..}           (direct, for servoing)
  T=105  CMD_SERVO_RAD_FEEDBACK  {"T":105} -> {x,y,z,t,torB,torS,torE}
  T=106  CMD_EOAT_HAND_CTRL      {"T":106,"cmd":<rad>,"spd":..,"acc":..} (1.57=open .. 3.14=closed)
  T=121  CMD_SINGLE_JOINT_ANGLE  {"T":121,"joint":1-4,"angle":deg,"spd":..,"acc":..}
  T=122  CMD_JOINTS_ANGLE_CTRL   {"T":122,"b":..,"s":..,"e":..,"h":..,"spd":..,"acc":..}
  T=210  CMD_TORQUE_CTRL         {"T":210,"cmd":0|1}

Transport per official examples:
  HTTP:   GET http://<ip>/js?json=<url-encoded json>
  Serial: write json line + b"\n" @ 115200 baud
"""

from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass

import requests

# macOS names for the USB-UART bridges the RoArm-M2 ships with (CP210x /
# CH340); the number suffix changes with the USB port, hence the globs.
SERIAL_PORT_PATTERNS = (
    "/dev/cu.usbserial*",
    "/dev/cu.wchusbserial*",
    "/dev/cu.SLAB_USBtoUART*",
    "/dev/cu.usbmodem*",
    "/dev/ttyUSB*",
    "/dev/ttyACM*",
)


def list_serial_candidates() -> list[str]:
    found: list[str] = []
    for pat in SERIAL_PORT_PATTERNS:
        found += sorted(glob.glob(pat))
    return found


def probe_serial_port(port: str, baudrate: int = 115200, timeout: float = 1.5) -> dict | None:
    """Ask for feedback (T:105) on `port`; the arm's reply dict if it answers
    with a pose, else None."""
    import serial

    try:
        with serial.Serial(port, baudrate=baudrate, timeout=0.3, dsrdtr=None) as ser:
            ser.setRTS(False)
            ser.setDTR(False)
            ser.reset_input_buffer()
            ser.write(b'{"T":105}\n')
            ser.flush()
            deadline = time.time() + timeout
            while time.time() < deadline:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if all(k in data for k in ("x", "y", "z")):
                    return data
    except (OSError, serial.SerialException):
        return None
    return None


def find_serial_port(baudrate: int = 115200, verbose: bool = True) -> str | None:
    """The serial port the arm answers on, or None. One candidate: taken as-is
    (no probe, so a busy/slow arm still works); several: the first that
    replies to T:105, else the first candidate."""
    cands = list_serial_candidates()
    if not cands:
        if verbose:
            print("(RoArm: no USB serial device found -- is the arm plugged in via USB?)")
        return None
    if len(cands) == 1:
        if verbose:
            print(f"(RoArm: serial port {cands[0]})")
        return cands[0]
    for port in cands:
        if probe_serial_port(port, baudrate) is not None:
            if verbose:
                print(f"(RoArm: serial port {port} -- answered feedback; other candidates: {[c for c in cands if c != port]})")
            return port
    if verbose:
        print(f"(RoArm: none of {cands} answered; using {cands[0]})")
    return cands[0]


@dataclass
class Pose:
    x: float
    y: float
    z: float
    t: float

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z, "t": self.t}


class RoArm:
    GRIPPER_OPEN = 1.57
    GRIPPER_CLOSED = 3.14
    GRIPPER_MAX = 3.4  # firm-close ceiling, a bit past nominal "closed" for a tighter grip

    def __init__(self, ip: str | None = None, serial_port: str | None = None, baudrate: int = 115200):
        """serial_port: a device path, "auto" (pick the USB serial device the
        arm answers on), or a path that no longer exists (the usbserial-NNN
        suffix changes with the USB port) -- the latter two auto-detect, and
        fall back to HTTP at `ip` when no USB serial device is present."""
        self.ip = ip
        self.baudrate = baudrate
        self._serial = None
        if serial_port and (serial_port == "auto" or not os.path.exists(serial_port)):
            if serial_port != "auto":
                print(f"(RoArm: {serial_port} not present, looking for the arm's USB serial port)")
            detected = find_serial_port(baudrate)
            if detected is None and ip:
                print(f"(RoArm: falling back to HTTP at {ip})")
            serial_port = detected
        self.serial_port = serial_port

    # ---- transport -------------------------------------------------
    def _send_serial(self, payload: dict) -> dict | None:
        import serial

        if self._serial is None or not self._serial.is_open:
            self._serial = serial.Serial(self.serial_port, baudrate=self.baudrate, timeout=1.0, dsrdtr=None)
            self._serial.setRTS(False)
            self._serial.setDTR(False)

        self._serial.reset_input_buffer()
        self._serial.write(json.dumps(payload).encode("utf-8") + b"\n")
        self._serial.flush()

        if payload.get("T") != 105:
            return {"sent": True}

        deadline = time.time() + 1.0
        last = None
        while time.time() < deadline:
            line = self._serial.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(k in last for k in ("x", "y", "z")):
                return last
        return last

    def _send_http(self, payload: dict) -> dict | None:
        r = requests.get(f"http://{self.ip}/js", params={"json": json.dumps(payload)}, timeout=1.0)
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            return None

    def send(self, payload: dict) -> dict | None:
        if self.serial_port:
            return self._send_serial(payload)
        return self._send_http(payload)

    # ---- commands ----------------------------------------------------
    def move_init(self) -> None:
        self.send({"T": 100})

    def feedback(self) -> Pose | None:
        res = self.send({"T": 105})
        if not res or not all(k in res for k in ("x", "y", "z")):
            return None
        return Pose(x=float(res["x"]), y=float(res["y"]), z=float(res["z"]), t=float(res.get("t", self.GRIPPER_CLOSED)))

    def move_to(self, pose: Pose, speed: float = 0.25) -> None:
        """Interpolated move (T:104) -- use for larger repositioning moves."""
        self.send({"T": 104, "x": pose.x, "y": pose.y, "z": pose.z, "t": pose.t, "spd": speed})

    def servo_to(self, pose: Pose) -> None:
        """Direct move (T:1041) -- use for small steps in a visual-servo loop."""
        self.send({"T": 1041, "x": pose.x, "y": pose.y, "z": pose.z, "t": pose.t})

    def set_gripper(self, rad: float, speed: float = 0, acc: float = 0) -> None:
        rad = max(self.GRIPPER_OPEN, min(self.GRIPPER_MAX, rad))
        self.send({"T": 106, "cmd": rad, "spd": speed, "acc": acc})

    def open_gripper(self) -> None:
        self.set_gripper(self.GRIPPER_OPEN)

    def close_gripper(self) -> None:
        self.set_gripper(self.GRIPPER_CLOSED)

    def torque(self, on: bool) -> None:
        self.send({"T": 210, "cmd": 1 if on else 0})
