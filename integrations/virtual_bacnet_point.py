"""
integrations/virtual_bacnet_point.py

A free, local stand-in for the real BMS hardware point that
integrations/bacnet_adapter.py writes to -- so the adapter and its
write-then-read-back verification (scripts/test_bacnet_adapter.py) can be
demoed and tested without any physical BACnet gateway, matching the
blueprint's "test against a free BACnet simulator... rather than real
hardware" guidance.

Starts a second local BACnet/IP device (on its own port, since it's a
separate "device" on the loopback segment from the adapter's own device)
exposing one commandable analogValue point -- the same object type/shape a
real BMS analog setpoint point would expose. A real deployment simply
points BACnetAdapter.device_ip at the actual gateway/controller instead of
at this script.

Run standalone to leave a live point up for manual poking (e.g. with the
YABE BACnet browser, or scripts/test_bacnet_adapter.py in another
terminal):

    python integrations/virtual_bacnet_point.py
"""

import asyncio
import threading
import time


def start_virtual_point(ip="127.0.0.1", port=47830, instance=1,
                         name="hvacSetpointAV", initial_value_c=22.0,
                         ready_timeout_s=10):
    """
    Start a background BACnet/IP device exposing one commandable
    analogValue point and return (device_handle, loop). The loop is
    exposed so a caller that wants to shut the device down cleanly can do
    `loop.call_soon_threadsafe(loop.stop)`; for a short-lived test process
    just letting the daemon thread die with the process is fine.
    """
    ready = threading.Event()
    state = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def _init():
            try:
                import BAC0
                from BAC0.core.devices.local.factory import ObjectFactory
                from bacpypes3.local.analog import AnalogValueObject

                device = BAC0.lite(ip=ip, port=port, ping=False)
                factory = ObjectFactory(
                    AnalogValueObject, instance, name,
                    properties={"units": "degreesCelsius"},
                    presentValue=float(initial_value_c),
                    is_commandable=True,
                )
                factory.add_objects_to_application(device)
                state["device"] = device
                state["loop"] = loop
            except Exception as e:  # noqa: BLE001
                state["error"] = e
            finally:
                ready.set()

        loop.call_soon(_init)
        loop.run_forever()

    thread = threading.Thread(target=_runner, daemon=True,
                               name="virtual-bacnet-point")
    thread.start()

    if not ready.wait(timeout=ready_timeout_s):
        raise RuntimeError("Virtual BACnet point did not start in time")
    if "error" in state:
        raise RuntimeError(f"Failed to start virtual point: {state['error']}")

    return state["device"], state["loop"]


if __name__ == "__main__":
    device, loop = start_virtual_point()
    print(f"Virtual BACnet point 'hvacSetpointAV' (AV:1) live on this device. "
          f"Point it at from BACnetAdapter as device_ip pointing here. "
          f"Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        loop.call_soon_threadsafe(loop.stop)
