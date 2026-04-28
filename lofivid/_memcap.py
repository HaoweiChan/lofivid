"""Soft memory cap for lofivid.

WSL2 has a fixed RAM allocation set in `~/.wslconfig` on the Windows side; once
the pipeline blows past it, the whole WSL distro can lock up. We can't change
the WSL allocation from inside the distro, but we CAN:

1. Cap the Python process's data-segment size via `RLIMIT_DATA`. CUDA + GPU
   memory mappings use anonymous `mmap`, which isn't counted against
   `RLIMIT_DATA`, so this safely limits Python heap growth without breaking
   torch.
2. Run a background watcher that compares process RSS to the cap every few
   seconds and shouts in the log when we're getting close. This gives the user
   a visible early-warning instead of a frozen WSL.
3. Force aggressive collection between pipeline stages.

Usage:
    from lofivid._memcap import apply_memory_cap
    apply_memory_cap(gb=12)      # called once near CLI entry, before torch loads
"""

from __future__ import annotations

import gc
import logging
import os
import resource
import threading
import time

log = logging.getLogger(__name__)


_WATCHER_STARTED = False


def apply_memory_cap(gb: float | None) -> None:
    """Install RLIMIT_DATA and start an RSS watcher. Idempotent.

    `gb=None` disables the cap entirely (default behaviour).
    """
    global _WATCHER_STARTED
    if gb is None or gb <= 0:
        return

    bytes_cap = int(gb * (1024 ** 3))

    # RLIMIT_DATA limits the data segment (brk + sbrk + many mmap allocations).
    # PyTorch CUDA mappings go through cuMemMap which is anon-mmap and doesn't
    # count, so this is safe. We set the *soft* limit; hard limit stays at
    # whatever the OS gave us so we can lift the cap at runtime if needed.
    soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
    new_hard = max(hard, bytes_cap) if hard != resource.RLIM_INFINITY else hard
    try:
        resource.setrlimit(resource.RLIMIT_DATA, (bytes_cap, new_hard))
        log.info("Memory cap: RLIMIT_DATA → %.1f GB (soft)", gb)
    except (ValueError, OSError) as e:
        log.warning("Could not install RLIMIT_DATA cap: %s", e)

    if _WATCHER_STARTED:
        return
    _WATCHER_STARTED = True
    t = threading.Thread(target=_rss_watcher, args=(bytes_cap,), daemon=True)
    t.start()


def _rss_watcher(cap_bytes: int) -> None:
    """Log a WARN when RSS climbs past 75% of cap, ERROR past 90%."""
    pid = os.getpid()
    last_state = "ok"
    while True:
        try:
            rss = _read_rss(pid)
        except FileNotFoundError:
            return
        ratio = rss / cap_bytes if cap_bytes else 0
        if ratio > 0.90:
            state = "critical"
        elif ratio > 0.75:
            state = "warning"
        else:
            state = "ok"
        if state != last_state:
            mb = rss / (1024 ** 2)
            cap_mb = cap_bytes / (1024 ** 2)
            if state == "critical":
                log.error(
                    "MEMCAP CRITICAL: RSS %.0f MB / cap %.0f MB (%.0f%%) — "
                    "consider killing the process before WSL freezes",
                    mb, cap_mb, ratio * 100,
                )
            elif state == "warning":
                log.warning(
                    "MEMCAP WARNING: RSS %.0f MB / cap %.0f MB (%.0f%%)",
                    mb, cap_mb, ratio * 100,
                )
            else:
                log.info("MEMCAP recovered: RSS %.0f MB / cap %.0f MB", mb, cap_mb)
            last_state = state
        time.sleep(5)


def _read_rss(pid: int) -> int:
    """Resident set size in bytes via /proc/<pid>/statm (page count)."""
    with open(f"/proc/{pid}/statm") as f:
        # fields: size resident shared text lib data dt
        resident_pages = int(f.read().split()[1])
    return resident_pages * resource.getpagesize()


def collect_between_stages(stage_name: str) -> None:
    """Cheap inter-stage clean-up. Call after each major pipeline phase."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    log.debug("collect_between_stages(%s) done", stage_name)
