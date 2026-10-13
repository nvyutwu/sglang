"""NaN forensics (SGLANG_NAN_FORENSICS=1): where did a request's first NaN come from?

Observation only; no output changes. Built for the Kimi-K3 token-0 ('!') collapse, where the
spread is fixed (zero-on-alloc + DCP page-table tail) but the origin of the first NaN is open.

Always on (CUDA-graph safe, no host sync; in-place updates of preallocated device buffers,
captured with the graph and run on every replay):
  * per decoder layer, for the self_attn output and the mlp output: the first forward step
    with a NaN row, that row's position, and the cumulative NaN row count;
  * per FP8 conversion site (prefill q/k/v in trtllm_mla_backend._quantize_fp8_qkv; decode
    q, q_rope, k, k_rope in the fused KV/Q writer and the mla_quantize_* helpers): the
    running max |x| of the input, the number of calls with |x| > 448 (an e4m3 overflow,
    which a non-saturating conversion stores as NaN) and the first step of such a call.

On an incident (a sampled logits row is NaN; called by the batch result processor, on
every TP rank, after a device sync), each rank logs one line prefixed NAN_INCIDENT with:
  * the request: rid, prompt / output length, cached and host-hit prefix lengths,
    retraction, pool slots, and the batch's forward mode and size;
  * this rank's per-layer first-NaN table and FP8 overflow stats (then the first-NaN table
    is reset so the next incident is localized too);
  * a census of the request's own state on this rank: MLA KV rows (fp8 NaN bytes) per MLA
    layer with the first / last NaN position, and the KDA (linear attention) recurrent state
    per layer (NaN / Inf / all-zero);
  * client aborts seen by the scheduler in the last 300 s.
Rank 0 also logs a NAN_FORENSICS heartbeat every SGLANG_NAN_FORENSICS_HEARTBEAT_S seconds
when anything changed (FP8 headroom per site, NaN rows per kind, its page-0 census). Every rank
checks the reserved page 0 of the MLA KV pool (CUDA-graph padding tokens write there; it is never
handed out, so zero-on-alloc never clears it) and logs a NAN_FORENSICS page0 line if it holds NaN.

How to read an incident:
  * NaN first in an mlp output while every attn output of that step is clean -> MoE /
    dense MLP numerics (or an expert all-reduce); attn first -> attention at that layer.
  * prefill_* overflow calls > 0 and KV NaN rows at prefix positions -> the non-saturating
    prefill FP8 conversion wrote NaN KV (seed candidate (a), prefill path).
  * KV or KDA NaN at positions the request did not compute in this process (inside the
    cached / host-hit prefix) -> corrupt cached state (HiCache load-back, abort race).
  * NaN first on one rank only (compare the ranks' lines) -> that GPU or its local KV.
  * page-0 NaN (heartbeat / page0 lines) -> padding writes landed NaN in the reserved slot; the
    open question is then which read path reaches page 0 (the fused FP8 KV/Q writer has no
    slot-0 skip under DCP, unlike store_cache / set_mla_kv_buffer since #32477).
"""

import collections
import json
import logging
import os
import socket
import time
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

FP8_MAX = 448.0
KINDS = ("attn", "mlp")
SITES = ("prefill_q", "prefill_k", "prefill_v", "dec_q", "dec_q_rope", "dec_k", "dec_k_rope")
_ABORT_WINDOW_S = 300.0
_MAX_FULL_INCIDENTS = 50  # full census for the first N incidents, then every 100th


def enabled() -> bool:
    return envs.SGLANG_NAN_FORENSICS.get()


class _State:
    installed = False
    step: Optional[torch.Tensor] = None  # [1] int64, forward-step counter
    positions: Optional[torch.Tensor] = None  # positions of the current forward
    first_step: Optional[torch.Tensor] = None  # [L, 2] int64, -1 = none since last reset
    first_pos: Optional[torch.Tensor] = None  # [L, 2] int64
    nan_rows: Optional[torch.Tensor] = None  # [L, 2] int64, cumulative
    ovf_max: Optional[torch.Tensor] = None  # [S] float32
    ovf_calls: Optional[torch.Tensor] = None  # [S] int64
    ovf_first: Optional[torch.Tensor] = None  # [S] int64
    n_layers = 0
    aborts: "collections.deque" = collections.deque(maxlen=512)
    incidents = 0
    hb_host: Optional[List[torch.Tensor]] = None
    hb_last: Any = None
    hb_t = 0.0
    hb_pending = False
    hb_rank0 = False
    hb_p0_host: Optional[torch.Tensor] = None
    hb_p0_last: Any = None


_S = _State()


def _log(tag: str, obj: Dict[str, Any]) -> None:
    logger.error("%s %s", tag, json.dumps(obj, separators=(",", ":"), default=str))


# ----------------------------------------------------------------------------- recording
def _record_nan(layer_idx: int, kind: int, out) -> None:
    if isinstance(out, (tuple, list)):
        out = next((o for o in out if isinstance(o, torch.Tensor)), None)
    if out is None or out.dim() == 0 or out.numel() == 0:
        return
    # A NaN anywhere in a row makes the fp32 row sum NaN (an Inf row sums to Inf, not NaN).
    rows = torch.isnan(out.reshape(out.shape[0], -1).sum(-1, dtype=torch.float32))
    cnt = rows.sum()
    newly = (_S.first_step[layer_idx, kind] < 0) & (cnt > 0)
    _S.first_step[layer_idx, kind] = torch.where(newly, _S.step[0], _S.first_step[layer_idx, kind])
    pos = _S.positions
    if pos is not None and pos.dim() == 1 and pos.shape[0] == rows.shape[0]:
        # index_select, not pos[idx]: a 0-d index in __getitem__ calls .item() (illegal in capture)
        p = torch.index_select(pos, 0, rows.to(torch.int64).argmax().view(1)).squeeze(0)
        _S.first_pos[layer_idx, kind] = torch.where(newly, p.to(torch.int64), _S.first_pos[layer_idx, kind])
    _S.nan_rows[layer_idx, kind] += cnt


def _record_ovf(site: str, x) -> None:
    if not isinstance(x, torch.Tensor) or x.numel() == 0 or _S.ovf_max is None:
        return
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return  # already quantized
    i = SITES.index(site)
    m = x.detach().abs().amax().float()
    _S.ovf_max[i] = torch.maximum(_S.ovf_max[i], m)
    hit = m > FP8_MAX
    _S.ovf_calls[i] += hit.to(torch.int64)
    _S.ovf_first[i] = torch.where(hit & (_S.ovf_first[i] < 0), _S.step[0], _S.ovf_first[i])


def _wrap_fn(fn, sites):
    def wrapped(*args, **kwargs):
        try:
            for site, x in zip(sites, args):
                if site:
                    _record_ovf(site, x)
        except Exception:  # never break the engine
            pass
        return fn(*args, **kwargs)

    wrapped.__wrapped__ = fn
    return wrapped


def _patch_fp8_sites() -> List[str]:
    patched = []
    dec = ["dec_q", "dec_q_rope", "dec_k", "dec_k_rope"]
    try:
        from sglang.srt.layers.attention import trtllm_mla_backend as trt

        if hasattr(trt, "_quantize_fp8_qkv"):
            trt._quantize_fp8_qkv = _wrap_fn(trt._quantize_fp8_qkv, ["prefill_q", "prefill_k", "prefill_v"])
            patched.append("trtllm._quantize_fp8_qkv")
        for name in ("mla_quantize_without_rope_for_fp8", "mla_quantize_and_rope_for_fp8"):
            if hasattr(trt, name):
                setattr(trt, name, _wrap_fn(getattr(trt, name), dec))
                patched.append("trtllm." + name)
        cls = getattr(trt, "TRTLLMMLABackend", None)
        if cls is not None and hasattr(cls, "_set_kv_and_concat_q_fp8_fused"):
            orig = cls._set_kv_and_concat_q_fp8_fused

            def fused(self, *args, **kwargs):
                try:
                    for site, key in zip(dec, ("q", "q_rope", "k", "k_rope")):
                        _record_ovf(site, kwargs.get(key))
                except Exception:
                    pass
                return orig(self, *args, **kwargs)

            cls._set_kv_and_concat_q_fp8_fused = fused
            patched.append("TRTLLMMLABackend._set_kv_and_concat_q_fp8_fused")
    except Exception as e:
        _log("NAN_FORENSICS", {"event": "patch_failed", "module": "trtllm_mla_backend", "error": repr(e)})
    try:
        from sglang.srt.layers.attention import cutedsl_mla_backend as cute

        for name in ("mla_quantize_without_rope_for_fp8", "mla_quantize_and_rope_for_fp8"):
            if hasattr(cute, name):
                setattr(cute, name, _wrap_fn(getattr(cute, name), dec))
                patched.append("cutedsl." + name)
    except Exception as e:
        _log("NAN_FORENSICS", {"event": "patch_failed", "module": "cutedsl_mla_backend", "error": repr(e)})
    return patched


def _find_decoder(model: torch.nn.Module) -> Optional[torch.nn.Module]:
    for m in model.modules():
        layers = getattr(m, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and any(
            hasattr(layer, "self_attn") or hasattr(layer, "mlp") for layer in layers
        ):
            return m
    return None


def maybe_install(model: torch.nn.Module) -> None:
    """Install the recorders. Call after the model is loaded and before CUDA graph capture."""
    if not enabled() or _S.installed:
        return
    try:
        dec = _find_decoder(model)
        if dec is None:
            _log("NAN_FORENSICS", {"event": "install_skipped", "reason": "no decoder layers"})
            return
        dev = next(model.parameters()).device
        n = len(dec.layers)
        _S.n_layers = n
        _S.step = torch.zeros(1, dtype=torch.int64, device=dev)
        _S.first_step = torch.full((n, len(KINDS)), -1, dtype=torch.int64, device=dev)
        _S.first_pos = torch.full((n, len(KINDS)), -1, dtype=torch.int64, device=dev)
        _S.nan_rows = torch.zeros((n, len(KINDS)), dtype=torch.int64, device=dev)
        _S.ovf_max = torch.zeros(len(SITES), dtype=torch.float32, device=dev)
        _S.ovf_calls = torch.zeros(len(SITES), dtype=torch.int64, device=dev)
        _S.ovf_first = torch.full((len(SITES),), -1, dtype=torch.int64, device=dev)

        def pre(mod, args, kwargs):
            _S.step.add_(1)
            _S.positions = kwargs.get("positions", args[1] if len(args) > 1 else None)

        dec.register_forward_pre_hook(pre, with_kwargs=True)
        for i, layer in enumerate(dec.layers):
            for k, name in enumerate(KINDS):
                sub = getattr(layer, "self_attn" if name == "attn" else "mlp", None)
                if sub is not None:
                    sub.register_forward_hook(lambda m, a, o, i=i, k=k: _record_nan(i, k, o))
        patched = _patch_fp8_sites()
        _S.installed = True
        _log("NAN_FORENSICS", {"event": "installed", "layers": n, "decoder": type(dec).__name__,
                               "fp8_sites": patched, "device": str(dev), "pid": os.getpid()})
    except Exception as e:  # never break the engine
        _log("NAN_FORENSICS", {"event": "install_failed", "error": repr(e)})


# ----------------------------------------------------------------------------- events
def note_abort(rid: str, abort_all: bool = False) -> None:
    """Called by the scheduler for every client AbortReq."""
    if enabled():
        _S.aborts.append((time.time(), "*" if abort_all else str(rid)))


# ----------------------------------------------------------------------------- incident
def _rank_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {"host": socket.gethostname(), "pid": os.getpid()}
    try:
        info["cuda_device"] = torch.cuda.current_device()
        info["gpu_uuid"] = str(torch.cuda.get_device_properties(info["cuda_device"]).uuid)
    except Exception:
        pass
    try:
        from sglang.srt.runtime_context import get_parallel

        par = get_parallel()
        for key in ("tp_rank", "tp_size", "attn_dcp_rank", "attn_dcp_size"):
            if hasattr(par, key):
                info[key] = getattr(par, key)
    except Exception:
        pass
    return info


def _probe_table() -> Dict[str, Any]:
    if _S.first_step is None:
        return {"installed": False}
    fs, fp, nr = _S.first_step.cpu(), _S.first_pos.cpu(), _S.nan_rows.cpu()
    om, oc, of = _S.ovf_max.cpu(), _S.ovf_calls.cpu(), _S.ovf_first.cpu()
    nan_first = [
        {"layer": li, "kind": KINDS[k], "first_step": int(fs[li, k]), "first_pos": int(fp[li, k]),
         "rows_total": int(nr[li, k])}
        for li in range(fs.shape[0]) for k in range(fs.shape[1]) if fs[li, k] >= 0
    ]
    nan_first.sort(key=lambda d: (d["first_step"], d["layer"], d["kind"]))
    return {
        "step": int(_S.step[0]),
        "nan_first": nan_first[:24],
        "nan_layers": len(nan_first),
        "ovf": {s: {"max": round(float(om[j]), 2), "calls": int(oc[j]), "first_step": int(of[j])}
                for j, s in enumerate(SITES)},
    }


def _is_nan_store(x: torch.Tensor) -> torch.Tensor:
    """Per-row NaN flag for a KV row tensor [n, ...] in its storage dtype."""
    if x.dtype in (torch.uint8, torch.int8) or "float8" in str(x.dtype):
        u = x.view(torch.uint8).reshape(x.shape[0], -1)
        return ((u & 0x7F) == 0x7F).any(-1)  # e4m3fn NaN = S.1111.111
    return torch.isnan(x.float().reshape(x.shape[0], -1)).any(-1)


def _kv_census(proc, req, seq_len: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    kvcache = proc.token_to_kv_pool_allocator.get_kvcache()
    full = getattr(kvcache, "full_kv_pool", kvcache)
    bufs = getattr(full, "kv_buffer", None)
    if bufs is None or req.req_pool_idx is None or seq_len <= 0:
        return {"skipped": "no kv_buffer or no req slot"}
    slots = proc.req_to_token_pool.req_to_token[req.req_pool_idx, :seq_len].to(torch.int64)
    dcp_size, dcp_rank = 1, 0
    try:
        from sglang.srt.runtime_context import get_parallel

        dcp_size = int(getattr(get_parallel(), "attn_dcp_size", 1) or 1)
        dcp_rank = int(getattr(get_parallel(), "attn_dcp_rank", 0) or 0)
    except Exception:
        pass
    pos = torch.arange(seq_len, device=slots.device)
    if dcp_size > 1:  # owner rule: slot % dcp_size == rank, row = slot // dcp_size
        own = (slots % dcp_size) == dcp_rank
        rows, pos = slots[own] // dcp_size, pos[own]
    else:
        rows = slots
    out["own_rows"] = int(rows.numel())
    out["zero_slot_rows"] = int((rows == 0).sum())  # padded slot 0 = never written for this req
    layer_ids = getattr(kvcache, "full_attention_layer_id_mapping", None)
    inv = {v: k for k, v in layer_ids.items()} if isinstance(layer_ids, dict) else {}
    nan_layers = []
    for j, buf in enumerate(bufs):
        if buf is None or buf.shape[0] == 0 or rows.numel() == 0:
            continue
        flag = _is_nan_store(buf.index_select(0, rows))
        n = int(flag.sum())
        if n:
            p = pos[flag]
            nan_layers.append({"mla": j, "layer": inv.get(j, -1), "nan_rows": n,
                               "first_pos": int(p.min()), "last_pos": int(p.max())})
    out["nan_layers"] = nan_layers
    return out


def _kda_census(proc, req) -> Dict[str, Any]:
    pool = getattr(proc.req_to_token_pool, "mamba_pool", None)
    idx = getattr(req, "mamba_pool_idx", None)
    if pool is None or idx is None:
        return {"skipped": "no mamba pool or slot"}
    idx = int(idx)
    st = pool.mamba_cache.temporal[:, idx]  # [L, ...]
    flat = st.float().reshape(st.shape[0], -1)
    nan = torch.isnan(flat).any(-1)
    inf = torch.isinf(flat).any(-1)
    zero = flat.abs().amax(-1) == 0
    out = {"slot": idx, "layers": int(st.shape[0]),
           "nan_layers": torch.nonzero(nan).flatten().tolist(),
           "inf_layers": torch.nonzero(inf).flatten().tolist(),
           "all_zero_layers": int(zero.sum())}
    conv = getattr(pool.mamba_cache, "conv", None)
    if isinstance(conv, (list, tuple)):
        out["conv_nan_layers"] = sorted({li for c in conv
                                         for li in torch.nonzero(torch.isnan(c[:, idx].float().reshape(c.shape[0], -1)).any(-1)).flatten().tolist()})
    return out


def incident(proc, batch, flagged_idx: List[int]) -> None:
    """Log one NAN_INCIDENT line on this rank for the flagged rows of ``batch``."""
    if not enabled():
        return
    _S.incidents += 1
    full = _S.incidents <= _MAX_FULL_INCIDENTS or _S.incidents % 100 == 0
    try:
        torch.cuda.synchronize()
        now = time.time()
        rec: Dict[str, Any] = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                               "incident": _S.incidents, **_rank_info()}
        fm = getattr(batch, "forward_mode", None)
        rec["batch"] = {"forward_mode": getattr(fm, "name", str(fm)), "size": len(batch.reqs),
                        "overlap": bool(getattr(proc, "enable_overlap", False))}
        reqs = []
        for i in flagged_idx:
            if i >= len(batch.reqs):
                continue
            req = batch.reqs[i]
            seq_len = len(req.origin_input_ids) + len(req.output_ids)
            r: Dict[str, Any] = {
                "rid": req.rid, "row": i, "prompt_len": len(req.origin_input_ids),
                "output_len": len(req.output_ids), "cached_tokens": getattr(req, "cached_tokens", None),
                "host_hit_length": getattr(req, "host_hit_length", None),
                "retracted": bool(getattr(req, "is_retracted", False)),
                "req_pool_idx": req.req_pool_idx, "mamba_pool_idx": getattr(req, "mamba_pool_idx", None),
            }
            if full:
                for name, fn in (("kv", lambda: _kv_census(proc, req, seq_len)), ("kda", lambda: _kda_census(proc, req))):
                    try:
                        r[name] = fn()
                    except Exception as e:
                        r[name] = {"error": repr(e)}
            reqs.append(r)
        rec["reqs"] = reqs
        rec["probe"] = _probe_table()
        rec["recent_aborts"] = [{"age_s": round(now - t, 1), "rid": rid}
                                for t, rid in list(_S.aborts) if now - t <= _ABORT_WINDOW_S][-20:]
        _log("NAN_INCIDENT", rec)
        if _S.first_step is not None:  # localize the next incident too
            _S.first_step.fill_(-1)
            _S.first_pos.fill_(-1)
            torch.cuda.synchronize()
    except Exception as e:  # never break the engine
        _log("NAN_INCIDENT", {"incident": _S.incidents, "error": repr(e)})


def _page0_census(proc) -> Optional[torch.Tensor]:
    """Device [L, 2]: per MLA layer, the rows of reserved page 0 (rows [0, page_size): CUDA-graph
    padding tokens write there, and it is never handed out, so zero-on-alloc never clears it) that
    hold an fp8 NaN byte, and those that hold a +-448 byte (the saturated e4m3 maximum)."""
    kvcache = proc.token_to_kv_pool_allocator.get_kvcache()
    full = getattr(kvcache, "full_kv_pool", kvcache)
    bufs = getattr(full, "kv_buffer", None)
    ps = int(getattr(full, "page_size", 0) or 0)
    if not bufs or ps <= 0:
        return None
    out = []
    for buf in bufs:
        if buf is None or buf.shape[0] < ps:
            continue
        rows = buf[:ps]
        if rows.dtype in (torch.uint8, torch.int8) or "float8" in str(rows.dtype):
            u = rows.view(torch.uint8).reshape(ps, -1) & 0x7F
            nan, sat = (u == 0x7F).any(-1), (u == 0x7E).any(-1)
        else:
            f = rows.float().reshape(ps, -1)
            nan, sat = torch.isnan(f).any(-1), torch.isinf(f).any(-1)
        out.append(torch.stack([nan.sum(), sat.sum()]))
    return torch.stack(out) if out else None


def heartbeat(proc=None) -> None:
    """Low rate, no sync: every SGLANG_NAN_FORENSICS_HEARTBEAT_S seconds the counters (rank 0) and
    the page-0 census (every rank, needs ``proc``) are copied (non_blocking) to pinned memory; the
    next call, one batch later, logs that copy. Rank 0 logs FP8 headroom per site, NaN rows per
    kind and its page-0 counts when anything changed; any rank whose page 0 holds NaN logs that."""
    if not enabled() or _S.ovf_max is None:
        return
    try:
        if _S.hb_pending:
            _S.hb_pending = False
            p0 = None
            if _S.hb_p0_host is not None:
                h = _S.hb_p0_host.clone()
                p0 = {"nan_rows": int(h[:, 0].sum()), "sat_rows": int(h[:, 1].sum()),
                      "nan_layers": torch.nonzero(h[:, 0]).flatten().tolist(),
                      "sat_layers": torch.nonzero(h[:, 1]).flatten().tolist()}
            if _S.hb_rank0:
                step, om, oc, nr = [h.clone() for h in _S.hb_host]
                snap = ([round(v, 1) for v in om.tolist()], oc.tolist(), nr.sum(0).tolist(), p0)
                if snap != _S.hb_last:
                    _S.hb_last = snap
                    _log("NAN_FORENSICS", {"event": "heartbeat", "step": int(step[0]),
                                           "ovf_max": dict(zip(SITES, snap[0])), "ovf_calls": dict(zip(SITES, snap[1])),
                                           "nan_rows": dict(zip(KINDS, snap[2])), "page0": p0,
                                           "incidents": _S.incidents})
            elif p0 is not None and p0["nan_rows"] and p0 != _S.hb_p0_last:
                _S.hb_p0_last = p0
                _log("NAN_FORENSICS", {"event": "page0", **_rank_info(), "page0": p0})
            return
        now = time.time()
        if now - _S.hb_t < envs.SGLANG_NAN_FORENSICS_HEARTBEAT_S.get():
            return
        _S.hb_t = now
        from sglang.srt.runtime_context import get_parallel

        _S.hb_rank0 = getattr(get_parallel(), "tp_rank", 0) == 0
        if _S.hb_rank0:
            bufs = [_S.step, _S.ovf_max, _S.ovf_calls, _S.nan_rows]
            if _S.hb_host is None:
                _S.hb_host = [torch.empty_like(b, device="cpu").pin_memory() for b in bufs]
            for h, b in zip(_S.hb_host, bufs):
                h.copy_(b, non_blocking=True)
        if proc is not None:
            p0 = _page0_census(proc)
            if p0 is not None:
                if _S.hb_p0_host is None or _S.hb_p0_host.shape != p0.shape:
                    _S.hb_p0_host = torch.empty(p0.shape, dtype=p0.dtype, pin_memory=True)
                _S.hb_p0_host.copy_(p0, non_blocking=True)
        _S.hb_pending = _S.hb_rank0 or _S.hb_p0_host is not None
    except Exception as e:
        _log("NAN_FORENSICS", {"event": "heartbeat_failed", "error": repr(e)})
