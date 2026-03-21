"""Tests for the OTel bridge metric name sanitization logic.

These tests exercise _sanitize_metric_name, _METRIC_NAME_MAP, and
_NATIVE_PROMETHEUS_METRICS from otel_instrumentation.py without requiring a
running SGLang server or any OTel/Prometheus infrastructure.
"""
import importlib
import sys
import types
import unittest


# ---------------------------------------------------------------------------
# Lightweight stubs so we can import otel_instrumentation without the real
# OTel / Prometheus packages being installed.
# ---------------------------------------------------------------------------

def _make_stub_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _install_stubs():
    """Install minimal stubs for OTel and prometheus_client packages."""
    # prometheus_client.parser stub — must expose text_string_to_metric_families
    # because otel_instrumentation.py imports it at module level.
    parser_mod = _make_stub_module(
        "prometheus_client.parser",
        text_string_to_metric_families=lambda *a, **kw: [],
    )
    prom_mod = _make_stub_module(
        "prometheus_client",
        parser=parser_mod,
        text_string_to_metric_families=lambda *a, **kw: [],
    )
    sys.modules.setdefault("prometheus_client", prom_mod)
    sys.modules.setdefault("prometheus_client.parser", parser_mod)

    # opentelemetry stubs
    for pkg in [
        "opentelemetry",
        "opentelemetry.sdk",
        "opentelemetry.sdk.metrics",
        "opentelemetry.sdk.metrics.export",
        "opentelemetry.sdk.resources",
        "opentelemetry.sdk.logs",
        "opentelemetry.sdk.logs.export",
        "opentelemetry.exporter",
        "opentelemetry.exporter.otlp",
        "opentelemetry.exporter.otlp.proto",
        "opentelemetry.exporter.otlp.proto.grpc",
        "opentelemetry.exporter.otlp.proto.grpc._log_exporter",
        "opentelemetry.exporter.otlp.proto.grpc.metric_exporter",
        "opentelemetry.metrics",
    ]:
        sys.modules.setdefault(pkg, _make_stub_module(pkg))


_install_stubs()

# Now import the module under test
import importlib.util
import os

_OTEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "sglang", "srt", "tracing", "otel_instrumentation.py",
)
_spec = importlib.util.spec_from_file_location(
    "otel_instrumentation", os.path.abspath(_OTEL_PATH)
)
_otel_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_otel_mod)  # type: ignore[union-attr]

_sanitize = _otel_mod._sanitize_metric_name
_MAP = _otel_mod._METRIC_NAME_MAP
_NATIVE = _otel_mod._NATIVE_PROMETHEUS_METRICS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSanitizeMetricName(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1. Colon → underscore sanitization
    # ------------------------------------------------------------------

    def test_colon_replaced_with_underscore(self):
        self.assertEqual(_sanitize("foo:bar"), "foo_bar")

    def test_colon_in_sglang_prefixed_name(self):
        # colon sanitization happens before prefix strip
        result = _sanitize("sglang:some_metric")
        # After colon→_, becomes sglang_some_metric; strip → some_metric
        self.assertEqual(result, "some_metric")

    # ------------------------------------------------------------------
    # 2. Native Prometheus metrics are NOT stripped
    # ------------------------------------------------------------------

    def test_native_process_cpu_kept(self):
        name = "sglang_process_cpu_seconds_total"
        result = _sanitize(name)
        # Must NOT be stripped because process_cpu_seconds_total is native
        self.assertEqual(result, "sglang_process_cpu_seconds_total")

    def test_native_process_open_fds_kept(self):
        self.assertEqual(_sanitize("sglang_process_open_fds"), "sglang_process_open_fds")

    def test_native_process_max_fds_kept(self):
        self.assertEqual(_sanitize("sglang_process_max_fds"), "sglang_process_max_fds")

    def test_native_process_virtual_memory_kept(self):
        self.assertEqual(
            _sanitize("sglang_process_virtual_memory_bytes"),
            "sglang_process_virtual_memory_bytes",
        )

    def test_native_process_resident_memory_kept(self):
        self.assertEqual(
            _sanitize("sglang_process_resident_memory_bytes"),
            "sglang_process_resident_memory_bytes",
        )

    def test_native_process_start_time_kept(self):
        self.assertEqual(
            _sanitize("sglang_process_start_time_seconds"),
            "sglang_process_start_time_seconds",
        )

    def test_native_python_gc_collected_kept(self):
        self.assertEqual(
            _sanitize("sglang_python_gc_objects_collected_total"),
            "sglang_python_gc_objects_collected_total",
        )

    def test_native_python_gc_uncollectable_kept(self):
        self.assertEqual(
            _sanitize("sglang_python_gc_objects_uncollectable_total"),
            "sglang_python_gc_objects_uncollectable_total",
        )

    def test_native_python_gc_collections_kept(self):
        self.assertEqual(
            _sanitize("sglang_python_gc_collections_total"),
            "sglang_python_gc_collections_total",
        )

    def test_native_python_info_kept(self):
        self.assertEqual(_sanitize("sglang_python_info"), "sglang_python_info")

    # ------------------------------------------------------------------
    # 3. Explicit _METRIC_NAME_MAP entries produce correct unified names
    # ------------------------------------------------------------------

    def _check_map(self, sglang_name: str, expected_otel: str):
        # Verify the map contains the entry
        self.assertIn(sglang_name, _MAP, f"{sglang_name} not in _METRIC_NAME_MAP")
        self.assertEqual(_MAP[sglang_name], expected_otel)
        # Verify _sanitize_metric_name uses the map
        self.assertEqual(_sanitize(sglang_name), expected_otel)

    def test_map_e2e_request_latency(self):
        self._check_map("sglang_e2e_request_latency_seconds", "e2e_request_latency_seconds")

    def test_map_time_to_first_token(self):
        self._check_map("sglang_time_to_first_token_seconds", "time_to_first_token_seconds")

    def test_map_inter_token_latency(self):
        self._check_map("sglang_inter_token_latency_seconds", "inter_token_latency_seconds")

    def test_map_queue_time(self):
        self._check_map("sglang_queue_time_seconds", "request_queue_time_seconds")

    def test_map_request_inference_time(self):
        self._check_map("sglang_request_inference_time_seconds", "request_inference_time_seconds")

    def test_map_request_prefill_time(self):
        self._check_map("sglang_request_prefill_time_seconds", "request_prefill_time_seconds")

    def test_map_request_decode_time(self):
        self._check_map("sglang_request_decode_time_seconds", "request_decode_time_seconds")

    def test_map_request_tpot(self):
        self._check_map(
            "sglang_request_time_per_output_token_seconds",
            "request_time_per_output_token_seconds",
        )

    def test_map_prompt_tokens_total(self):
        self._check_map("sglang_prompt_tokens_total", "prompt_tokens_total")

    def test_map_generation_tokens_total(self):
        self._check_map("sglang_generation_tokens_total", "generation_tokens_total")

    def test_map_request_prompt_tokens(self):
        self._check_map("sglang_request_prompt_tokens", "request_prompt_tokens")

    def test_map_request_generation_tokens(self):
        self._check_map("sglang_request_generation_tokens", "request_generation_tokens")

    def test_map_request_success_total(self):
        self._check_map("sglang_request_success_total", "request_success_total")

    def test_map_num_retracted_requests_total(self):
        self._check_map("sglang_num_retracted_requests_total", "num_retracted_requests_total")

    def test_map_num_running_reqs_renamed(self):
        self._check_map("sglang_num_running_reqs", "num_requests_running")

    def test_map_num_queue_reqs_renamed(self):
        self._check_map("sglang_num_queue_reqs", "num_requests_waiting")

    def test_map_token_usage_renamed(self):
        self._check_map("sglang_token_usage", "kv_cache_usage_perc")

    def test_map_num_requests_total(self):
        self._check_map("sglang_num_requests_total", "num_requests_total")

    def test_map_gen_throughput(self):
        self._check_map("sglang_gen_throughput", "gen_throughput")

    def test_map_engine_startup_time(self):
        self._check_map("sglang_engine_startup_time", "engine_startup_time")

    def test_map_engine_load_weights_time(self):
        self._check_map("sglang_engine_load_weights_time", "engine_load_weights_time")

    def test_map_model_config_info(self):
        self._check_map("sglang_model_config_info", "model_config_info")

    def test_map_parallel_config_info(self):
        self._check_map("sglang_parallel_config_info", "parallel_config_info")

    def test_map_speculative_config_info(self):
        self._check_map("sglang_speculative_config_info", "speculative_config_info")

    def test_map_cache_hit_rate(self):
        self._check_map("sglang_cache_hit_rate", "cache_hit_rate")

    def test_map_cached_tokens_total(self):
        self._check_map("sglang_cached_tokens_total", "cached_tokens_total")

    def test_map_mm_cache_queries_adds_total(self):
        # mm_cache_queries → mm_cache_queries_total (adds _total suffix)
        self._check_map("sglang_mm_cache_queries", "mm_cache_queries_total")

    def test_map_mm_cache_hits_adds_total(self):
        self._check_map("sglang_mm_cache_hits", "mm_cache_hits_total")

    def test_map_spec_accept_rate(self):
        self._check_map("sglang_spec_accept_rate", "spec_accept_rate")

    def test_map_spec_accept_length(self):
        self._check_map("sglang_spec_accept_length", "spec_accept_length")

    def test_map_spec_decode_num_drafts_adds_total(self):
        self._check_map("sglang_spec_decode_num_drafts", "spec_decode_num_drafts_total")

    def test_map_spec_decode_per_pos_adds_total(self):
        self._check_map(
            "sglang_spec_decode_num_accepted_tokens_per_pos",
            "spec_decode_num_accepted_tokens_per_pos_total",
        )

    def test_map_lora_pool_utilization(self):
        self._check_map("sglang_lora_pool_utilization", "lora_pool_utilization")

    def test_map_request_params_max_tokens(self):
        self._check_map("sglang_request_params_max_tokens", "request_params_max_tokens")

    def test_map_request_type_image_total(self):
        self._check_map("sglang_request_type_image_total", "request_type_image_total")

    def test_map_request_type_video_total(self):
        self._check_map("sglang_request_type_video_total", "request_type_video_total")

    def test_map_request_type_tool_call_total(self):
        self._check_map("sglang_request_type_tool_call_total", "request_type_tool_call_total")

    def test_map_request_type_structured_output_total(self):
        self._check_map(
            "sglang_request_type_structured_output_total",
            "request_type_structured_output_total",
        )

    # ------------------------------------------------------------------
    # 4. Non-sglang metrics pass through unchanged (after colon fix)
    # ------------------------------------------------------------------

    def test_non_sglang_prefix_passthrough(self):
        self.assertEqual(_sanitize("some_other_metric"), "some_other_metric")

    def test_vllm_prefixed_metric_passthrough(self):
        # vLLM metrics go through a different bridge; they should not be stripped here
        self.assertEqual(_sanitize("vllm_prompt_tokens"), "vllm_prompt_tokens")

    def test_already_unprefixed_metric_passthrough(self):
        self.assertEqual(_sanitize("prompt_tokens_total"), "prompt_tokens_total")

    # ------------------------------------------------------------------
    # 5. Default strip for sglang_ metrics NOT in the map
    # ------------------------------------------------------------------

    def test_default_strip_unknown_sglang_metric(self):
        # A metric added in the future that is not in the explicit map should
        # have its sglang_ prefix stripped by the fallback logic.
        result = _sanitize("sglang_some_future_metric")
        self.assertEqual(result, "some_future_metric")

    def test_default_strip_pd_queue_depth(self):
        # PD disaggregation gauges — not in the explicit map — strip by default
        self.assertEqual(
            _sanitize("sglang_num_prefill_prealloc_queue_reqs"),
            "num_prefill_prealloc_queue_reqs",
        )

    def test_default_strip_grammar_metric(self):
        self.assertEqual(
            _sanitize("sglang_num_grammar_cache_hit_total"),
            "num_grammar_cache_hit_total",
        )

    # ------------------------------------------------------------------
    # 6. Idempotency: running sanitize twice should be stable
    # ------------------------------------------------------------------

    def test_idempotent_on_already_stripped(self):
        # Once a name has been stripped, running sanitize again should not alter it
        first = _sanitize("sglang_gen_throughput")
        second = _sanitize(first)
        self.assertEqual(first, second)

    # ------------------------------------------------------------------
    # 7. Map completeness: all values are non-empty strings without sglang_ prefix
    # ------------------------------------------------------------------

    def test_map_values_have_no_sglang_prefix(self):
        for sglang_name, otel_name in _MAP.items():
            self.assertFalse(
                otel_name.startswith("sglang_"),
                f"OTel name '{otel_name}' for '{sglang_name}' should not start with 'sglang_'",
            )

    def test_map_keys_all_have_sglang_prefix(self):
        for sglang_name in _MAP:
            self.assertTrue(
                sglang_name.startswith("sglang_"),
                f"Map key '{sglang_name}' should start with 'sglang_'",
            )

    def test_native_set_all_lack_sglang_prefix(self):
        for name in _NATIVE:
            self.assertFalse(
                name.startswith("sglang_"),
                f"Native metric '{name}' should not include the sglang_ prefix",
            )


if __name__ == "__main__":
    unittest.main()
