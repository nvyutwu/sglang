# `feat/glm51-kv-fingerprint` — deliberate non-fixes from local cross-review

The local cross-review (Claude + Codex, 2026-05-06) surfaced findings
that are **deliberately not addressed** on this branch because the
GLM-5.1-NVFP4 long-CoT bug under investigation only fires under the
narrow surface (Dynamo + SGLang **disagg 1P:1D** + NSA + sustained
long-CoT load) and these findings live outside it. They become real
bugs the moment this code is reused; document them here so the next
person doesn't have to re-derive the analysis.

> **2026-05-06 update — post-Arm-B retract→resume framing.**
> The P2 retract-pressure ablation (Arm B, job 2039659) localized the
> bug to the **decode-side retract→resume code path**: clamping
> `--max-running-requests=8` holds peak `token_usage` at 0.69, fires
> zero retracts, and produces 94 % pass@1 / 0 gibberish on the
> otherwise-broken Step 5 recipe. The new top suspects are
> H1'-via-retract / H2'-pointer-via-retract / H6'-via-retract.
>
> SGLang's disagg retract→resume is **not re-prefill via NIXL** — it
> is a CPU↔GPU memcpy: `req.offload_kv_cache` saves the seq's MLA
> `kv_buffer` bytes to a CPU buffer attached to the req, the slots are
> freed, then `req.load_kv_cache` allocates fresh slots and writes the
> CPU bytes back. **The NSA `index_k_with_scale_buffer` is never
> offloaded/loaded** (NSATokenToKVPool inherits MLATokenToKVPool's
> `get/load_cpu_copy` without override). Post-resume the new state-pool
> pages contain whatever was at those page IDs from prior owners — the
> direct H6'-via-retract surface.
>
> Two of the deferred items below have been promoted to **fixed** as a
> result; one remains deferred but is explicitly noted as the next
> follow-up if the trace localizes to PP>1 or HiSparse paths.

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

## ~~M6~~ — Hook C-narrow on `topk_indices` ✅ **PROMOTED TO FIXED (2026-05-06)**

Originally deferred as "H6'-narrow is a Step D2 follow-up." After Arm B
localized the bug to retract→resume and made H6'-via-retract a top
suspect, the superset Hook C alone is no longer adequate — it would
mask the smoking gun (the surrounding pages match recv-fp; only the
topk-selected page diverges).

**Now implemented** in `nsa_backend.py`:

- `_emit_topk_read_fingerprints(forward_batch, layer, physical_pages)` —
  per-(rid, layer, page) fingerprint, dedup'd via `_FP_SEEN_PAGES_TOPK`,
  events emitted with `is_topk: 1`.
- Wired into `forward_decode`:
  - **trtllm path:** before `_forward_trtllm`, do
    `metadata.page_table_1.gather(1, topk_indices.clamp(min=0).long())`
    to resolve position indices → physical pages, fingerprint those.
  - **non-trtllm paths** (flashmla_sparse / flashmla_kv / tilelang /
    fa3 / aiter): call after `page_table_1` is built (the
    transform_index_page_table_decode result is already physical pages).
  - Skipped during cuda-graph capture, same as superset hook.
- The original superset hook is still emitted (with `is_topk: 0`) to
  keep H1'/H2' detection — superset and narrow are complementary.

`forward_extend` not yet wired — extends are usually short and the
H6'-via-retract surface is a decode-side bookkeeping issue. Add
symmetrically if a follow-on run needs it.

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
