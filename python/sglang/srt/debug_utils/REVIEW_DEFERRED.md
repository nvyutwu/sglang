# `feat/glm51-kv-fingerprint` — deliberate non-fixes from local cross-review

The local cross-review (Claude + Codex, 2026-05-06) surfaced findings
that are **deliberately not addressed** on this branch because the
GLM-5.1-NVFP4 long-CoT bug under investigation only fires under the
narrow surface (Dynamo + SGLang **disagg 1P:1D** + NSA + sustained
long-CoT load) and these findings live outside it. They become real
bugs the moment this code is reused; document them here so the next
person doesn't have to re-derive the analysis.

## M2 — MHA value-channel pages not fingerprinted

`_send_kvcache_generic` transfers both K and V for MHA, registering
`2 * layer_num` contiguous buffers. Hooks A and B in `nixl/conn.py`
loop only `layer_num` and call `get_key_buffer(li + start_layer)` —
the V channel never reaches `batch_page_fingerprints`. Any corruption
of MHA value channels would be invisible.

**Why deferred:** GLM-5.1 uses MLA + NSA. `is_mla_backend` is `True`
for the broken Step-5 stack, and the NSA state channel is fingerprinted
correctly. MHA disagg with this branch would silently miss V-channel
corruption.

**Fix when needed:** loop `2 * layer_num` for MHA, fingerprint K and V,
add a `channel: "k"|"v"` field. Update `post_process.py` to key on
`(channel, layer, page, is_state)`.

## M5 — Pool registry single-slot overwrite (HiSparse / draft pool)

`_KV_POOL` and `_STATE_POOL` are single global slots in
`debug_utils/kv_fingerprint.py`. `register_kv_pool` is called from
every `get_contiguous_buf_infos`. With HiSparse enabled,
`disaggregation/decode.py:320` calls `host_pool.get_contiguous_buf_infos()`
first → CPU host pool gets registered → main GPU pool registration at
:324 then overwrites correctly, but a draft pool registration at :330
overwrites with the speculative draft pool. Fingerprint hooks then
read from the wrong pool.

**Why deferred:** GLM-5.1 Step-5 recipe runs without HiSparse and
without separate speculative-decode draft pool (MTP shares the main
pool). The single-slot model is correct for this run.

**Fix when needed:** key the registry by purpose (`"main"`, `"draft"`,
`"host"`); refuse non-CUDA pools; or take the first registration only.

## M6 — Hook C is over-broad (fingerprints all `req_to_token`, not topk)

`_emit_first_read_fingerprints` is called at the top of NSA
`forward_decode` and fingerprints all pages in
`req_to_token[rpi, :seq_len]` — the full sequence-level page set, not
the topk-selected page set NSA actually consults via
`page_table_1` (built around lines 1659-1673 of `nsa_backend.py`
from `topk_indices`). The hook captures the **superset**.

**Why deferred:** The current verdict matrix from §7 of the plan only
needs Hook C to detect H1' (B≠C) and the absence-of-page case for H2'
(pointer remap) — both of which work fine with the superset
(corruption on a real read would still register as B≠C; missing pages
still register as recv-without-read). Only **H6'-narrow** (NSA picks
the wrong topk page from a correct page set) is degraded — the
surrounding pages match recv-fps and drown the smoking gun. The plan
already calls H6'-narrow out as a "Step D2 follow-up" (§8), so this
is an explicit phase boundary, not an oversight.

**Fix when needed:** thread `page_table_1` through the hook signature
(it varies per `nsa_decode_impl` — `flashmla_sparse`, `fa3_sparse`,
`tilelang`, `trtllm`); call the hook **after** `page_table_1` is
computed (post line 1673); fingerprint only the unique pages in
`page_table_1`. Different NSA backends use slightly different
indexing semantics for `page_table_1`, so be careful — log the raw
entries and let the post-processor join on whatever namespace the
recv side emits.

## m1 — Non-paged `TokenToKVPoolAllocator.free` deferred-free double-log

In `mem_cache/allocator.py`, the non-paged free path's `_log_alloc_free`
call sits **outside** the `if self.is_not_in_free_group:` branch, so a
free during `free_group_begin()`/`free_group_end()` brackets logs
once at append-to-group time and again when `free_group_end()`
flushes via `self.free(torch.cat(...))`. The paged variant
(`PagedTokenToKVPoolAllocator.free`) gets it right.

**Why deferred:** NSA uses `PagedTokenToKVPoolAllocator` (`page_size=64`).
The non-paged allocator only fires for legacy / page_size=1 backends
that aren't in the broken Step-5 recipe.

**Fix when needed:** move the call inside the conditional, mirroring
the paged variant.
