"""
integrations/bacnet_adapter.py

"""

import asyncio
import threading
import time


class BACnetWriteError(RuntimeError):
    """Raised when a BACnet write/read fails or times out."""


class BACnetAdapter:
    def __init__(self, device_ip, point_instance, local_ip="127.0.0.1",
                 local_port=47820, priority=8, connect_timeout_s=10,
                 io_timeout_s=8):
        self.device_ip = device_ip
        self.point_instance = int(point_instance)
        self.local_ip = local_ip
        self.local_port = int(local_port)
        self.priority = int(priority)
        self.connect_timeout_s = connect_timeout_s
        self.io_timeout_s = io_timeout_s

        self._loop = None
        self._thread = None
        self._client = None
        self._ready = threading.Event()
        self._connect_error = None
        self.last_write_ok = None
        self.last_write_value = None
        self.last_error = None

    # -- lifecycle ----------------------------------------------------------

    def connect(self):
        """Start the background loop/thread and register the BACnet/IP device."""
        if self._client is not None:
            return  # already connected

        def _runner():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            def _init():
                try:
                    import BAC0  # optional dependency, imported lazily
                    self._client = BAC0.lite(
                        ip=self.local_ip, port=self.local_port, ping=False
                    )
                except Exception as e:  # noqa: BLE001 -- surface any init failure
                    self._connect_error = e
                finally:
                    self._ready.set()

            self._loop.call_soon(_init)
            self._loop.run_forever()

        self._thread = threading.Thread(target=_runner, daemon=True,
                                         name="bacnet-adapter-loop")
        self._thread.start()

        if not self._ready.wait(timeout=self.connect_timeout_s):
            raise BACnetWriteError(
                f"BACnet adapter did not come up within {self.connect_timeout_s}s"
            )
        if self._connect_error is not None:
            raise BACnetWriteError(
                f"Failed to start local BACnet/IP device on "
                f"{self.local_ip}:{self.local_port}: {self._connect_error}"
            )

    def disconnect(self):
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        self._client = None

    # -- control path ---------------------------------------------------------

    def write_setpoint(self, value_c):
        """
        Write a cooling/heating setpoint (deg C) to the target device's
        analogValue point. Blocking, with a timeout -- returns True/False
        rather than raising, so a caller on the EnergyPlus hot path (which
        must never stall the simulation on a flaky network write) can just
        check the return value and keep going.
        """
        if self._client is None:
            self.connect()

        args = (
            f"{self.device_ip} analogValue {self.point_instance} "
            f"presentValue {float(value_c):.2f} - {self.priority}"
        )
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._client._write(args), self._loop
            )
            fut.result(timeout=self.io_timeout_s)
            self.last_write_ok = True
            self.last_write_value = float(value_c)
            self.last_error = None
            return True
        except Exception as e:  # noqa: BLE001
            self.last_write_ok = False
            self.last_error = str(e)
            return False

    def read_setpoint(self):
        """
        Read back the target point's presentValue. Used by the self-test
        (scripts/test_bacnet_adapter.py) to verify a write actually landed,
        and available for anyone who wants to confirm the BMS didn't
        override the commanded value at a higher priority.
        """
        if self._client is None:
            self.connect()

        args = f"{self.device_ip} analogValue {self.point_instance} presentValue"
        fut = asyncio.run_coroutine_threadsafe(
            self._client.read(args), self._loop
        )
        try:
            return fut.result(timeout=self.io_timeout_s)
        except Exception as e:  # noqa: BLE001
            raise BACnetWriteError(f"BACnet read failed: {e}") from e


def build_adapter_from_policy(policy):
    """
    Construct a BACnetAdapter from building_policy.yaml's `bacnet:` section,
    or return None if bacnet.enabled is falsy/missing. Kept out of main.py's
    body so main.py doesn't need to know BACnetAdapter's constructor shape.
    """
    cfg = (policy or {}).get("bacnet", {})
    if not cfg.get("enabled"):
        return None
    return BACnetAdapter(
        device_ip=cfg["device_ip"],
        point_instance=cfg.get("point_instance", 1),
        local_ip=cfg.get("local_ip", "127.0.0.1"),
        local_port=cfg.get("local_port", 47820),
        priority=cfg.get("priority", 8),
    )


if __name__ == "__main__":
    # Quick manual smoke test against integrations/virtual_bacnet_point.py --
    # run that in one terminal first, then this in another:
    #   python integrations/virtual_bacnet_point.py
    #   python integrations/bacnet_adapter.py
    adapter = BACnetAdapter(device_ip="127.0.0.1:47830", point_instance=1,
                             local_port=47821)
    adapter.connect()
    ok = adapter.write_setpoint(19.5)
    time.sleep(1)
    print("write ok:", ok, "| readback:", adapter.read_setpoint())
