"""Unit tests for NSATokenToKVPool.get_cpu_copy / load_cpu_copy.

Covers the per-token flat-index gather/scatter on the page-major
``index_k_with_scale_buffer`` layout that was added in
``feat/glm51-nsa-indexer-offload`` to close the H6'-via-retract surface.
The patch round-trips the NSA K-indexer state alongside the inherited
MLA ``kv_buffer`` across the disagg-decode retract→resume CPU↔GPU
memcpy. These tests verify byte-exact restore on a non-page-aligned
token slice, with destination pages poisoned between save and restore
so any drop or mis-mapped offset is caught.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu


class TestNSATokenToKVPoolCPUOffload(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for NSA CPU-offload tests.")
        if is_npu() or is_xpu():
            self.skipTest("NSA CPU-offload tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

    def _make_pool(self, layer_num: int = 3, num_pages: int = 6):
        page_size = 1 if is_hip() else 64
        size = page_size * num_pages
        return NSATokenToKVPool(
            size=size,
            page_size=page_size,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=576,
            index_head_dim=128,
        )

    @staticmethod
    def _fill_pool(pool: NSATokenToKVPool, salt: int = 0):
        for li in range(pool.layer_num):
            ibuf = pool.index_k_with_scale_buffer[li]
            data = torch.arange(ibuf.numel(), device=ibuf.device, dtype=torch.uint8)
            ibuf.copy_(((data + li * 7 + salt) % 256).view_as(ibuf))
            kbuf = pool.kv_buffer[li]
            kdata = torch.arange(kbuf.numel(), device=kbuf.device, dtype=kbuf.dtype)
            kbuf.copy_(kdata.view_as(kbuf) + (li + salt))

    @staticmethod
    def _per_token_state_slice(pool: NSATokenToKVPool, layer_id: int, token: int):
        buf = pool.index_k_with_scale_buffer[layer_id]
        head_dim = pool.index_head_dim
        page_size = pool.page_size
        page = int(token) // page_size
        offset = int(token) % page_size
        k = buf[page, offset * head_dim : (offset + 1) * head_dim].clone()
        s_start = page_size * head_dim + offset * 4
        s = buf[page, s_start : s_start + 4].clone()
        return k, s

    def test_round_trip_byte_exact_with_poisoned_pages(self):
        """get_cpu_copy → poison destination pages → load_cpu_copy must
        restore both kv_buffer and index_k_with_scale_buffer byte-exact
        for the saved tokens, even when the requested token slice spans
        partial pages and the pages have been clobbered in between."""
        pool = self._make_pool(layer_num=3, num_pages=6)
        self._fill_pool(pool, salt=0)

        page_size = pool.page_size
        # Non-page-aligned slice: tokens [page_size//2 .. 3*page_size + page_size//4].
        # Spans the tail of page 0 (partial), all of pages 1+2, and the head
        # of page 3 (partial). Exactly the pattern a retracted long-CoT seq
        # would hit.
        start = page_size // 2
        end = 3 * page_size + max(1, page_size // 4)
        token_indices = torch.arange(start, end, device="cuda", dtype=torch.int64)

        expected_state = []
        for li in range(pool.layer_num):
            per_layer = []
            for t in token_indices.tolist():
                k, s = self._per_token_state_slice(pool, li, t)
                per_layer.append((k, s))
            expected_state.append(per_layer)
        expected_kv = [
            pool.kv_buffer[li][token_indices].clone()
            for li in range(pool.layer_num)
        ]

        cpu_copy = pool.get_cpu_copy(token_indices)
        self.assertIsInstance(cpu_copy, dict)
        self.assertIn("kv", cpu_copy)
        self.assertIn("state", cpu_copy)

        # Poison every page touched by the slice. This simulates a
        # different sequence (or stale prior-owner bytes) writing into
        # the pages while the retracted seq is parked on the host.
        # If the override miscomputes any offset, the restored bytes
        # will be the poison value, not the saved bytes.
        self._fill_pool(pool, salt=171)

        pool.load_cpu_copy(cpu_copy, token_indices)

        for li in range(pool.layer_num):
            restored_kv = pool.kv_buffer[li][token_indices]
            self.assertTrue(
                torch.equal(restored_kv, expected_kv[li]),
                f"kv_buffer mismatch at layer {li}",
            )
            for idx, t in enumerate(token_indices.tolist()):
                k_got, s_got = self._per_token_state_slice(pool, li, t)
                k_exp, s_exp = expected_state[li][idx]
                self.assertTrue(
                    torch.equal(k_got, k_exp),
                    f"index K bytes mismatch layer={li} token={t}",
                )
                self.assertTrue(
                    torch.equal(s_got, s_exp),
                    f"index scale bytes mismatch layer={li} token={t}",
                )

    def test_round_trip_does_not_touch_other_tokens(self):
        """The override writes per-token slices, not full pages. Tokens
        adjacent to the saved slice but outside it must not be modified
        by load_cpu_copy. This guards against accidentally over-writing
        other sequences sharing the same page."""
        pool = self._make_pool(layer_num=2, num_pages=4)
        self._fill_pool(pool, salt=0)
        page_size = pool.page_size

        start = page_size // 2
        end = page_size + page_size // 2
        token_indices = torch.arange(start, end, device="cuda", dtype=torch.int64)

        # Snapshot the bytes for tokens that should NOT be restored.
        untouched_tokens = list(range(0, start)) + list(range(end, 2 * page_size))
        before_untouched = []
        for li in range(pool.layer_num):
            per_layer = []
            for t in untouched_tokens:
                per_layer.append(self._per_token_state_slice(pool, li, t))
            before_untouched.append(per_layer)
        before_kv_untouched = [
            pool.kv_buffer[li][torch.tensor(untouched_tokens, device="cuda")].clone()
            for li in range(pool.layer_num)
        ]

        cpu_copy = pool.get_cpu_copy(token_indices)
        # Refill the pool with the SAME salt so the untouched tokens stay
        # at their pre-save values. We re-poison only the saved-token
        # slice using direct indexing.
        for li in range(pool.layer_num):
            pool.kv_buffer[li][token_indices] = 0
            for t in token_indices.tolist():
                k, s = self._per_token_state_slice(pool, li, t)
                # zero out via in-place writes on the buffer
                buf = pool.index_k_with_scale_buffer[li]
                page = int(t) // page_size
                offset = int(t) % page_size
                head_dim = pool.index_head_dim
                buf[page, offset * head_dim : (offset + 1) * head_dim] = 0
                s_start = page_size * head_dim + offset * 4
                buf[page, s_start : s_start + 4] = 0
        pool.load_cpu_copy(cpu_copy, token_indices)

        # Untouched-token bytes must be unchanged.
        for li in range(pool.layer_num):
            after_untouched_kv = pool.kv_buffer[li][
                torch.tensor(untouched_tokens, device="cuda")
            ]
            self.assertTrue(
                torch.equal(after_untouched_kv, before_kv_untouched[li]),
                f"kv_buffer leaked into untouched tokens at layer {li}",
            )
            for idx, t in enumerate(untouched_tokens):
                k_got, s_got = self._per_token_state_slice(pool, li, t)
                k_exp, s_exp = before_untouched[li][idx]
                self.assertTrue(
                    torch.equal(k_got, k_exp),
                    f"index K leaked into untouched token layer={li} token={t}",
                )
                self.assertTrue(
                    torch.equal(s_got, s_exp),
                    f"index scale leaked into untouched token layer={li} token={t}",
                )

    def test_load_cpu_copy_rejects_non_dict_payload(self):
        """Non-dict payload reaching the override means a producer
        bypassed get_cpu_copy; restoring just kv_buffer would silently
        leak prior-owner bytes into index_k_with_scale_buffer. Must
        fail loud."""
        pool = self._make_pool(layer_num=1, num_pages=2)
        self._fill_pool(pool, salt=0)
        token_indices = torch.arange(0, pool.page_size, device="cuda", dtype=torch.int64)

        legacy_payload = [pool.kv_buffer[0][token_indices].cpu()]
        with self.assertRaises(TypeError):
            pool.load_cpu_copy(legacy_payload, token_indices)

        with self.assertRaises(TypeError):
            pool.load_cpu_copy({"kv": legacy_payload}, token_indices)


if __name__ == "__main__":
    unittest.main()
