"""
scripts/test_bacnet_adapter.py

Self-contained "Verify" step for 2.1 (BACnet/hardware bridge). Starts the
free virtual BMS point (integrations/virtual_bacnet_point.py), points a
real BACnetAdapter at it, writes a candidate setpoint the same way
core/energyplus_bridge.py would on the real control path, reads it back
over the network, and asserts the two match.

Produces logs/bacnet_adapter_test.json as an evidence artifact -- same
spirit as logs/control_log.jsonl: something you can show a judge on
request, not just an assertion that "the bridge works."

Usage:
    python scripts/test_bacnet_adapter.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integrations.virtual_bacnet_point import start_virtual_point
from integrations.bacnet_adapter import BACnetAdapter

TEST_WRITE_VALUE_C = 19.5
VIRTUAL_POINT_PORT = 47830
ADAPTER_LOCAL_PORT = 47821


def main():
    print("[1/4] Starting virtual BMS point (simulated hardware target)...")
    start_virtual_point(port=VIRTUAL_POINT_PORT, initial_value_c=22.0)
    time.sleep(1)  # let the device finish registering before we hit it

    print("[2/4] Connecting BACnetAdapter (the real control-path client)...")
    adapter = BACnetAdapter(
        device_ip=f"127.0.0.1:{VIRTUAL_POINT_PORT}",
        point_instance=1,
        local_port=ADAPTER_LOCAL_PORT,
    )
    adapter.connect()

    print(f"[3/4] Writing setpoint {TEST_WRITE_VALUE_C}°C over BACnet...")
    write_ok = adapter.write_setpoint(TEST_WRITE_VALUE_C)
    time.sleep(1)  # allow the write to land before reading back

    print("[4/4] Reading back presentValue to verify the write landed...")
    try:
        readback = adapter.read_setpoint()
        read_ok = True
    except Exception as e:  # noqa: BLE001
        readback = None
        read_ok = False
        print(f"    Read failed: {e}")

    matched = (
        read_ok and readback is not None
        and abs(float(readback) - TEST_WRITE_VALUE_C) < 0.05
    )
    passed = write_ok and matched

    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "written_value_c": TEST_WRITE_VALUE_C,
        "write_reported_ok": write_ok,
        "read_back_value": readback,
        "values_matched": matched,
        "result": "PASS" if passed else "FAIL",
    }

    os.makedirs("logs", exist_ok=True)
    with open("logs/bacnet_adapter_test.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'PASS' if passed else 'FAIL'}: wrote {TEST_WRITE_VALUE_C}, "
          f"read back {readback}. Evidence written to "
          f"logs/bacnet_adapter_test.json")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
