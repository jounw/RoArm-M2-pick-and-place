"""Find which port the RoArm-M2 is on.

Lists USB serial candidates, probes each with a feedback request, and also
checks the arm's own Wi-Fi access point (192.168.4.1) over HTTP. Prints the
export line to paste so every script picks the right transport.

Run: python3 find_roarm.py
"""

from __future__ import annotations

import os

import requests

from roarm_driver import list_serial_candidates, probe_serial_port


def main() -> int:
    found = False
    cands = list_serial_candidates()
    print(f"USB serial candidates: {cands or 'none'}")
    for port in cands:
        fb = probe_serial_port(port)
        if fb is not None:
            print(f"  {port}: RoArm answered  x={fb.get('x')} y={fb.get('y')} z={fb.get('z')} t={fb.get('t')}")
            print(f"\n  export ROARM_SERIAL_PORT={port}")
            found = True
        else:
            print(f"  {port}: no reply (not the arm, or it is busy)")

    ip = os.getenv("ROARM_IP", "192.168.4.1")
    try:
        r = requests.get(f"http://{ip}/js", params={"json": '{"T":105}'}, timeout=1.5)
        print(f"HTTP {ip}: reachable ({r.status_code}) -> {r.text[:80]}")
        print(f"\n  export ROARM_SERIAL_PORT=   # empty = use HTTP at {ip}")
        found = True
    except requests.RequestException:
        print(f"HTTP {ip}: not reachable (not on the arm's Wi-Fi)")

    if not found:
        print("\nRoArm not found. Plug it in via USB (the data cable, not charge-only) or join its Wi-Fi, then rerun.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
