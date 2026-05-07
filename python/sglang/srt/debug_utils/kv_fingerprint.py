"""KV-block fingerprint + alloc/free timeline logging.

Used to discriminate between candidate root causes of corrupted decode
output under sustained long-CoT load on disagg + NSA stacks:

    H1'  decode-side physical KV-block pool aliasing
    H2'  NIXL transport corruption / decode-side descriptor remap
    H6'  NSA per-active-sequence sparse-index / centroid table corruption

The hooks scattered through the runtime (NIXL send/recv, NSA
``forward_decode``, KV pool allocator) emit JSONL events into per-rank
files. ``post_process.py`` consumes them and produces a verdict.

Everything is gated on the ``SGLANG_DEBUG_KV_FINGERPRINT`` environment
variable. When unset, ``is_enabled()`` returns False and every public
function short-circuits — zero overhead in normal runs.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

_ENABLED: Optional[bool] = None


def is_enabled() -> bool:
    """Cached env-gated check. Set ``SGLANG_DEBUG_KV_FINGERPRINT=1`` to enable."""
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("SGLANG_DEBUG_KV_FINGERPRINT", "0") == "1"
    return _ENABLED


# ---------------------------------------------------------------------------
# Async JSONL logger (bounded SPSC-ish queue, dropped events counted)
# ---------------------------------------------------------------------------

_QUEUE_MAXSIZE = 1 << 18  # 256 K events; ~25 MB at 100 B/event
_BATCH = 4096

_q: Optional[queue.Queue] = None
_thread: Optional[threading.Thread] = None
_path: Optional[Path] = None
_dropped = 0
_drop_lock = threading.Lock()
_started = threading.Event()
_role: str = "unknown"
_rank: int = -1


def _encode(ev: dict) -> bytes:
    return (json.dumps(ev, separators=(",", ":")) + "\n").encode("ascii")


def _drain_loop(path: Path) -> None:
    global _dropped
    last_drop_log = time.monotonic()
    with open(path, "ab", buffering=1 << 20) as f:
        while True:
            batch: List[dict] = []
            try:
                ev = _q.get(timeout=1.0)  # type: ignore[union-attr]
                if ev is None:
                    break
                batch.append(ev)
                while len(batch) < _BATCH:
                    try:
                        ev = _q.get_nowait()  # type: ignore[union-attr]
                        if ev is None:
                            f.write(b"".join(_encode(x) for x in batch))
                            f.flush()
                            return
                        batch.append(ev)
                    except queue.Empty:
                        break
            except queue.Empty:
                pass
            if batch:
                f.write(b"".join(_encode(ev) for ev in batch))
            now = time.monotonic()
            if now - last_drop_log > 30.0:
                with _drop_lock:
                    if _dropped:
                        f.write(_encode({
                            "ev": "_dropped",
                            "n": _dropped,
                            "t_ns": time.time_ns(),
                        }))
                        _dropped = 0
                last_drop_log = now


def init(role: str = "unknown", rank: int = -1) -> None:
    """Idempotent. Safe to call from every disagg subsystem on bring-up."""
    if not is_enabled() or _started.is_set():
        return
    global _q, _thread, _path, _role, _rank
    _role = role
    if rank == -1:
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = int(os.environ.get("RANK", "0"))
        except Exception:
            rank = int(os.environ.get("RANK", "0"))
    _rank = rank
    log_dir = os.environ.get("SGLANG_DEBUG_KV_FP_LOG_DIR", "/tmp/sglang_kv_fp")
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    _path = p / f"fp_{role}_rank{rank}_pid{os.getpid()}.jsonl"
    _q = queue.Queue(maxsize=_QUEUE_MAXSIZE)
    _thread = threading.Thread(
        target=_drain_loop, args=(_path,), name="sglang-kv-fp", daemon=True
    )
    _thread.start()
    _started.set()
    atexit.register(shutdown)
    log({
        "ev": "_init", "role": role, "rank": rank, "pid": os.getpid(),
        "t_ns": time.time_ns(),
    })


def log(event: dict) -> None:
    if not _started.is_set():
        return
    try:
        _q.put_nowait(event)  # type: ignore[union-attr]
    except queue.Full:
        global _dropped
        with _drop_lock:
            _dropped += 1


def shutdown() -> None:
    if not _started.is_set():
        return
    try:
        _q.put_nowait(None)  # type: ignore[union-attr]
    except queue.Full:
        pass
    if _thread is not None:
        _thread.join(timeout=10.0)


def role() -> str:
    return _role


def rank() -> int:
    return _rank


# ---------------------------------------------------------------------------
# KV pool registry
#
# The NIXL manager holds raw GPU pointers, not torch references. When a pool
# reports its data ptrs to the manager (``get_contiguous_buf_infos`` for the
# main KV channel, ``get_state_buf_infos`` for NSA's index-K state channel)
# we stash the pool on the side so the fingerprint hook can read pages back
# via the pool's own accessors.
# ---------------------------------------------------------------------------

_KV_POOL = None       # exposes ``get_key_buffer(layer_id)``
_STATE_POOL = None    # exposes ``index_k_with_scale_buffer[layer_id]``


def register_kv_pool(pool) -> None:
    """Called by ``MLA/MHATokenToKVPool.get_contiguous_buf_infos``."""
    if not is_enabled():
        return
    global _KV_POOL
    _KV_POOL = pool
    log({
        "ev": "_pool_register", "kind": "kv",
        "cls": type(pool).__name__,
        "layer_num": getattr(pool, "layer_num", -1),
        "page_size": getattr(pool, "page_size", -1),
        "t_ns": time.time_ns(),
    })


def register_state_pool(pool) -> None:
    """Called by ``NSATokenToKVPool.get_state_buf_infos``."""
    if not is_enabled():
        return
    global _STATE_POOL
    _STATE_POOL = pool
    log({
        "ev": "_pool_register", "kind": "state",
        "cls": type(pool).__name__,
        "layer_num": getattr(pool, "layer_num", -1),
        "page_size": getattr(pool, "page_size", -1),
        "t_ns": time.time_ns(),
    })


def get_kv_pool():
    return _KV_POOL


def get_state_pool():
    return _STATE_POOL


# ---------------------------------------------------------------------------
# Page fingerprint
#
# blake2b-8 over (first 64 bytes ++ last 64 bytes) of the page. Catches
# random-bit-flip corruption and full block-aliasing replacement with
# vanishingly small false-match rate (~2^-64 per page-pair).
# ---------------------------------------------------------------------------

def page_fingerprint(buf, page_idx: int, tokens_per_page: int = 1) -> str:
    """Single-page fingerprint. Returns hex string or ``"err:<exc>"``.

    ``tokens_per_page`` is the buffer's row-to-page ratio:
      * ``1`` for page-major buffers (NSA ``index_k_with_scale_buffer``)
      * ``page_size`` (typically 64) for token-major buffers
        (``MHATokenToKVPool.k_buffer``, ``MLATokenToKVPool.kv_buffer``)
    """
    try:
        import torch
        p = int(page_idx)
        if tokens_per_page <= 1:
            page = buf[p].contiguous().view(torch.uint8)
        else:
            start = p * tokens_per_page
            end = start + tokens_per_page
            page = buf[start:end].contiguous().view(torch.uint8)
        n = page.numel()
        sample = page if n <= 128 else torch.cat([page[:64], page[n - 64:]])
        arr = sample.detach().cpu().numpy().tobytes()
        return hashlib.blake2b(arr, digest_size=8).hexdigest()
    except Exception as e:  # pragma: no cover
        return f"err:{type(e).__name__}"


def batch_page_fingerprints(
    buf, page_ids, tokens_per_page: int = 1
) -> List[str]:
    """Vectorized fingerprint for many pages with one D2H copy.

    ``tokens_per_page`` selects the indexing convention:
      * ``1`` — buffer first dim is num_pages; ``buf[i]`` is page ``i``.
      * ``>1`` — buffer first dim is num_tokens; page ``i`` is rows
        ``[i*tokens_per_page : (i+1)*tokens_per_page]``.

    Returns one hex string per ``page_ids`` entry, or one ``"err:<exc>"``
    string per entry on failure.
    """
    try:
        import torch
        n_pages = len(page_ids)
        if n_pages == 0:
            return []
        if tokens_per_page <= 1:
            idx = torch.as_tensor(
                page_ids, dtype=torch.long, device=buf.device
            )
            sub = (
                buf.index_select(0, idx)
                .contiguous()
                .view(n_pages, -1)
                .view(torch.uint8)
            )
        else:
            ids_arr = [int(p) for p in page_ids]
            base = torch.as_tensor(
                ids_arr, dtype=torch.long, device=buf.device
            ) * tokens_per_page
            offsets = torch.arange(
                tokens_per_page, dtype=torch.long, device=buf.device
            )
            row_idx = (base[:, None] + offsets[None, :]).reshape(-1)
            sub = (
                buf.index_select(0, row_idx)
                .contiguous()
                .view(n_pages, -1)
                .view(torch.uint8)
            )
        n_bytes = sub.shape[1]
        if n_bytes <= 128:
            sample = sub
        else:
            sample = torch.cat([sub[:, :64], sub[:, n_bytes - 64:]], dim=1)
        host = sample.cpu().numpy()
        return [
            hashlib.blake2b(row.tobytes(), digest_size=8).hexdigest()
            for row in host
        ]
    except Exception as e:  # pragma: no cover
        return [f"err:{type(e).__name__}"] * len(page_ids)


def kv_pool_tokens_per_page(pool) -> int:
    """Resolve the row-to-page ratio for a registered KV pool.

    NSA ``index_k_with_scale_buffer`` is page-major (rows == pages →
    ratio 1). MLA ``kv_buffer`` and MHA ``k_buffer``/``v_buffer`` are
    token-major: each page occupies ``page_size`` rows. Default of 1 is
    safe — at worst we sample fewer bytes than expected.
    """
    return int(getattr(pool, "page_size", 1) or 1)


# ---------------------------------------------------------------------------
# NIXL notif parser
# ---------------------------------------------------------------------------

def parse_notif(notif: str):
    """Parse the NIXL notification string emitted by the disagg KV manager.

    ``"{room}_kv_{chunk}_{is_last}_{pp}"``  (main KV channel)
    ``"{room}_state_{pp}"``                 (NSA state channel)
    ``"{room}_aux_{pp}"``                   (aux metadata)

    Returns ``(room: int, kind: str, chunk_id: int, is_last: bool, pp_rank: int)``.
    Falls back to a permissive parse on unexpected formats.
    """
    parts = notif.split("_", 4)
    room = int(parts[0])
    kind = parts[1]
    if kind == "kv":
        chunk_id = int(parts[2])
        is_last = bool(int(parts[3]))
        pp_rank = int(parts[4]) if len(parts) > 4 else 0
    else:
        chunk_id = -1
        is_last = True
        pp_rank = int(parts[2]) if len(parts) > 2 else 0
    return room, kind, chunk_id, is_last, pp_rank
