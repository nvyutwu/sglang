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


def _flush_dropped(f) -> None:
    """Flush the residual ``_dropped`` counter to disk + reset to zero.

    Called periodically (every 30 s in the steady-state loop) and once
    more on shutdown so the final tail of dropped-event count is never
    silently lost when the drain loop sees the sentinel and breaks.
    """
    global _dropped
    with _drop_lock:
        if _dropped:
            f.write(_encode({
                "ev": "_dropped",
                "n": _dropped,
                "t_ns": time.time_ns(),
            }))
            _dropped = 0


def _drain_loop(path: Path) -> None:
    last_drop_log = time.monotonic()
    with open(path, "ab", buffering=1 << 20) as f:
        while True:
            batch: List[dict] = []
            try:
                ev = _q.get(timeout=1.0)  # type: ignore[union-attr]
                if ev is None:
                    # Final flush before exiting: residual dropped count
                    # must land on disk; the daemon thread is killed at
                    # process exit and would otherwise lose the buffer.
                    _flush_dropped(f)
                    f.flush()
                    return
                batch.append(ev)
                while len(batch) < _BATCH:
                    try:
                        ev = _q.get_nowait()  # type: ignore[union-attr]
                        if ev is None:
                            f.write(b"".join(_encode(x) for x in batch))
                            _flush_dropped(f)
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
                _flush_dropped(f)
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
    """Gracefully drain the queue + flush remaining buffered I/O.

    The drain thread is daemon=True and would be killed at process exit
    without flushing its 1 MB write buffer (~1 MB of tail events lost).
    Use a blocking ``put`` with timeout so the sentinel always lands
    even when the queue is at capacity. The drain loop performs the
    final flush + writes residual ``_dropped`` count before returning.
    """
    if not _started.is_set():
        return
    try:
        _q.put(None, block=True, timeout=5.0)  # type: ignore[union-attr]
    except queue.Full:
        # Queue still saturated after 5 s — give up on a clean exit but
        # at least record the failure mode in stderr; the daemon thread
        # will be killed at process exit losing the tail.
        import sys
        print("kv_fingerprint.shutdown: queue saturated; tail may be lost",
              file=sys.stderr)
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

# ---------------------------------------------------------------------------
# Retract/resume + KV snapshot helpers
#
# In SGLang disagg, retract→resume on the decode side is NOT re-prefill via
# NIXL — it is a CPU↔GPU memcpy. ``offload_kv_cache`` saves the seq's KV
# bytes (MLA ``kv_buffer`` only — NSA ``index_k_with_scale_buffer`` is
# never touched), then slots are freed; ``load_kv_cache`` allocates fresh
# slots, writes the saved CPU bytes back. Hooks A (NIXL send) and B (NIXL
# recv) do not fire on this path. Without an explicit pre-offload /
# post-load snapshot, the post-processor cannot tell H1' (block contents
# wrong) from H2' (req_to_token wrong) from H6' (NSA index stale) on
# resumed sequences. These helpers close that gap.
# ---------------------------------------------------------------------------


def emit_retract(
    rid: str,
    rpi: int = -1,
    gen_idx: int = -1,
    n_tokens: int = -1,
    pool_usage: float = -1.0,
    extra: Optional[dict] = None,
) -> None:
    """Scheduler-side retract event. Fires per req returned by retract_decode."""
    if not is_enabled():
        return
    ev = {
        "ev": "retract",
        "rid": rid, "rpi": rpi,
        "gen_idx": gen_idx, "n_tokens": n_tokens,
        "pool_usage": pool_usage,
        "t_ns": time.time_ns(),
    }
    if extra:
        ev.update(extra)
    log(ev)


def emit_resume(
    rid: str,
    rpi: int = -1,
    gen_idx: int = -1,
    n_tokens: int = -1,
    extra: Optional[dict] = None,
) -> None:
    """Disagg decode-side resume event. Fires per req in resume_retracted_reqs."""
    if not is_enabled():
        return
    ev = {
        "ev": "resume",
        "rid": rid, "rpi": rpi,
        "gen_idx": gen_idx, "n_tokens": n_tokens,
        "t_ns": time.time_ns(),
    }
    if extra:
        ev.update(extra)
    log(ev)


def req_to_token_fp(token_indices) -> str:
    """blake2b-8 over an int array. Use to detect when ``req_to_token``
    changes between reads (a resume rewrites the row), or to confirm
    pre-offload vs post-load slot-id maps differ as expected."""
    try:
        if hasattr(token_indices, "detach"):
            arr = token_indices.detach().cpu().numpy()
        else:
            import numpy as np
            arr = np.asarray(token_indices)
        return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()
    except Exception as e:
        return f"err:{type(e).__name__}"


def snapshot_seq_pages(
    rid: str,
    token_indices,
    *,
    rpi: int = -1,
    label: str = "snap",
    extra: Optional[dict] = None,
) -> None:
    """Fingerprint every (layer, page) for a sequence's KV+state buffers.

    ``token_indices`` is the seq's slot list (typically
    ``req_to_token[rpi, :seq_len]``). Page IDs are derived via floor-div
    by ``page_size``. Fingerprints both ``kv_buffer`` (token-major) and
    ``index_k_with_scale_buffer`` (page-major) when their pools are
    registered. Each fingerprint event carries ``label`` so the
    post-processor can pair the same page across pre-offload /
    post-load / first-read snapshots.

    This is the only path that exercises the NSA state buffer at the
    retract→resume boundary — the SGLang offload/load path doesn't touch
    it, so post-load fp != pre-offload fp on the state channel is the
    H6'-via-retract smoking gun.
    """
    if not is_enabled():
        return
    kv_pool = get_kv_pool()
    state_pool = get_state_pool()
    if kv_pool is None and state_pool is None:
        return
    try:
        if hasattr(token_indices, "detach"):
            slots = token_indices.detach().cpu().numpy().tolist()
        else:
            slots = list(token_indices)
        if not slots:
            return
        page_size = (
            kv_pool_tokens_per_page(kv_pool)
            if kv_pool is not None
            else kv_pool_tokens_per_page(state_pool)
        )
        pages = sorted({int(s) // max(1, page_size) for s in slots})
    except Exception as e:
        log({
            "ev": "_snap_err", "label": label, "rid": rid, "rpi": rpi,
            "err": f"{type(e).__name__}: {e}", "t_ns": time.time_ns(),
        })
        return

    n_pages = len(pages)
    t_ns = time.time_ns()
    base = {
        "ev": "snap", "label": label, "rid": rid, "rpi": rpi,
        "n_slots": len(slots), "n_pages": n_pages,
    }
    if extra:
        base.update(extra)

    if kv_pool is not None:
        kv_start = getattr(kv_pool, "start_layer", 0) or 0
        layer_num = getattr(kv_pool, "layer_num", 0)
        tpp = kv_pool_tokens_per_page(kv_pool)
        for li in range(layer_num):
            layer_id = li + kv_start
            try:
                buf = kv_pool.get_key_buffer(layer_id)
            except Exception:
                continue
            fps = batch_page_fingerprints(buf, pages, tokens_per_page=tpp)
            for p, fp in zip(pages, fps):
                ev = dict(base)
                ev.update({
                    "channel": "kv", "layer": layer_id,
                    "page": int(p), "fp": fp, "t_ns": t_ns,
                })
                log(ev)

    if state_pool is not None:
        state_start = getattr(state_pool, "start_layer", 0) or 0
        state_layer_num = getattr(state_pool, "layer_num", 0)
        for li in range(state_layer_num):
            layer_id = li + state_start
            try:
                buf = state_pool.index_k_with_scale_buffer[layer_id - state_start]
            except Exception:
                continue
            fps = batch_page_fingerprints(buf, pages, tokens_per_page=1)
            for p, fp in zip(pages, fps):
                ev = dict(base)
                ev.update({
                    "channel": "state", "layer": layer_id,
                    "page": int(p), "fp": fp, "t_ns": t_ns,
                })
                log(ev)


# ---------------------------------------------------------------------------
# NIXL notif parser
# ---------------------------------------------------------------------------

def parse_notif(notif: str):
    """Parse the NIXL notification string emitted by the disagg KV manager.

    ``"{room}_kv_{chunk}_{is_last}_{pp}"``  (main KV channel)
    ``"{room}_state_{pp}"``                 (NSA state channel)
    ``"{room}_aux"``                        (aux metadata; no pp suffix)

    Note: aux notifs do NOT carry the pp_rank suffix in the current
    NIXL conn.py emit path (``send_aux`` sends ``f"{room}_aux"``); the
    helper falls back to ``pp_rank=0`` for them. They never reach the
    fingerprint hook regardless — ``_emit_send_fingerprints`` is only
    called from ``_send_kvcache_generic``, not from ``send_aux``.

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
