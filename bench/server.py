"""Launch / health-check / stop an inference server as a subprocess."""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

PREFIX_FLAGS = {
    "cache_on": "--enable-prefix-caching",
    "cache_off": "--no-enable-prefix-caching",
}


def build_cmd(template: str, mode: str, **fields) -> list[str]:
    """Fill the server command template from the config.

    `{prefix_caching_flag}` is set explicitly in *both* modes: newer vLLM enables
    prefix caching by default, so "off" must be requested, not assumed.
    """
    return shlex.split(template.format(prefix_caching_flag=PREFIX_FLAGS[mode], **fields))


class ServerProcess:
    def __init__(self, cmd: list[str], base_url: str, log_path: Path, startup_timeout_s: float = 900):
        self.cmd, self.base_url, self.log_path = cmd, base_url, log_path
        self.startup_timeout_s = startup_timeout_s
        self.proc: subprocess.Popen | None = None

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def __enter__(self) -> "ServerProcess":
        if self._healthy():
            raise RuntimeError(f"something is already serving on {self.base_url}; stop it first")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w")
        print(f"[server] $ {' '.join(self.cmd)}\n[server] log -> {self.log_path}", flush=True)
        self.proc = subprocess.Popen(self.cmd, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=os.environ.copy())
        t0 = time.time()
        while time.time() - t0 < self.startup_timeout_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}; see {self.log_path}")
            if self._healthy():
                print(f"[server] healthy after {time.time() - t0:.1f}s", flush=True)
                return self
            time.sleep(2)
        self.__exit__(None, None, None)
        raise TimeoutError(f"server not healthy after {self.startup_timeout_s}s; see {self.log_path}")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)        # graceful: frees GPU memory
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        # wait until the port is actually free before the next mode starts
        for _ in range(30):
            if not self._healthy():
                break
            time.sleep(1)
        if getattr(self, "_log", None):
            self._log.close()
