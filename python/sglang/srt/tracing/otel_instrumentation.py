"""
Minimal OpenTelemetry setup for SGLang.

- Initializes OTLP exporters for logs and metrics when the corresponding
  OTEL_* environment variables are configured. Trace setup is intentionally
  omitted here to avoid conflicting with SGLang's native tracing in
  sglang.srt.tracing.trace. If you want traces, use ServerArgs --enable-trace.
- Returns a standard logging handler that can be attached to the root logger
  so that all logs are forwarded to the OTEL collector.

Includes Kratos offload support for large in-log payloads (e.g. base64 media
URIs) and NVCF secrets loader for SSA credentials.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import logging
import os
import queue
import re
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from logging.handlers import QueueHandler, QueueListener
from typing import Any, Dict, Optional, Tuple, Tuple as Tup

from prometheus_client.parser import text_string_to_metric_families


logger = logging.getLogger(__name__)

# Global handles for program-wide access
_GLOBAL_METER = None
_PROM_BRIDGE_THREAD = None
_PROM_COUNTER_PREV: Dict[Tup[str, Tup[Tup[str, str], ...]], float] = {}
_PROM_BUCKET_PREV: Dict[Tup[str, Tup[Tup[str, str], ...]], float] = {}
_PROM_COUNTERS: Dict[str, Any] = {}
_PROM_GAUGE_VALUES: Dict[str, Dict[Tup[Tup[str, str], ...], float]] = {}
_PROM_GAUGES: Dict[str, Any] = {}

# Logging queue/processor globals
_LOG_QUEUE: Optional[queue.Queue] = None
_QUEUE_LISTENER: Optional[QueueListener] = None

# Cached bucket per namespace (best-effort)
_KRATOS_BUCKET_CACHE: Dict[str, str] = {}
_OFFLOAD_CACHE: Dict[str, str] = {}  # sha256(data) -> uri (best-effort dedup)


# NVCF secrets reload state
_SECRETS_LOCK = threading.Lock()
_NVCF_WATCH_THREAD: Optional[threading.Thread] = None
_NVCF_LAST_HASH: Optional[str] = None
_NVCF_SET_KEYS: set = set()  # env keys we set from secrets (eligible for override on reload)
_NVCF_CREATED_CRED_FILE: Optional[str] = None
_NVCF_DEFAULT_CRED_PATH = "/tmp/kratos_ssa.json"


def _file_sha256(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return None


def _is_truthy(value: Optional[str]) -> bool:
    if value is None:
        return False
    value = value.strip().lower()
    return value in ("1", "true", "yes", "on")


def _maybe_update_env(
    key: str, value: Optional[str], *, force: bool = False, only_if_absent: bool = False
) -> None:
    if not isinstance(value, str) or not value:
        return
    try:
        with _SECRETS_LOCK:
            if only_if_absent:
                if key in os.environ:
                    return
                os.environ[key] = value
                _NVCF_SET_KEYS.add(key)
                return
            if key in _NVCF_SET_KEYS or (force and key not in os.environ):
                os.environ[key] = value
                _NVCF_SET_KEYS.add(key)
    except Exception:
        return


def _write_credentials_file(path: str, client_id: str, client_secret: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as cf:
            json.dump({"client_id": client_id, "client_secret": client_secret}, cf)
    except Exception:
        return


def _apply_nvcf_secrets(secrets: Dict[str, Any], *, initial: bool) -> None:
    prod_cid = secrets.get("PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID")
    prod_csec = secrets.get("PRODUCTION_KRATOS_CLI_SSA_CLIENT_SECRET")
    gen_cid = secrets.get("KRATOS_CLI_SSA_CLIENT_ID")
    gen_csec = secrets.get("KRATOS_CLI_SSA_CLIENT_SECRET")

    cid = prod_cid or gen_cid
    csec = prod_csec or gen_csec

    _maybe_update_env(
        "PRODUCTION_KRATOS_CLI_SSA_CLIENT_ID", prod_cid, only_if_absent=initial, force=(not initial)
    )
    _maybe_update_env(
        "PRODUCTION_KRATOS_CLI_SSA_CLIENT_SECRET", prod_csec, only_if_absent=initial, force=(not initial)
    )

    _maybe_update_env("KRATOS_CLI_SSA_CLIENT_ID", cid, only_if_absent=initial, force=(not initial))
    _maybe_update_env("KRATOS_CLI_SSA_CLIENT_SECRET", csec, only_if_absent=initial, force=(not initial))

    if isinstance(cid, str) and cid and isinstance(csec, str) and csec:
        if initial:
            if "KRATOS_CLI_SSA_CREDENTIALS_FILE" not in os.environ:
                cred_path = _NVCF_DEFAULT_CRED_PATH
                _write_credentials_file(cred_path, cid, csec)
                os.environ["KRATOS_CLI_SSA_CREDENTIALS_FILE"] = cred_path
                _NVCF_SET_KEYS.add("KRATOS_CLI_SSA_CREDENTIALS_FILE")
                global _NVCF_CREATED_CRED_FILE
                _NVCF_CREATED_CRED_FILE = cred_path
        else:
            if _NVCF_CREATED_CRED_FILE:
                _write_credentials_file(_NVCF_CREATED_CRED_FILE, cid, csec)


def _load_nvcf_secrets_to_env() -> None:
    secrets_path = os.getenv("NVCF_SECRETS_PATH", "/var/secrets/secrets.json")
    try:
        if not os.path.exists(secrets_path):
            return
        with open(secrets_path, "r", encoding="utf-8") as f:
            secrets = json.load(f)
        global _NVCF_LAST_HASH
        _NVCF_LAST_HASH = _file_sha256(secrets_path)
        _apply_nvcf_secrets(secrets, initial=True)
    except Exception:
        return


def _start_nvcf_secrets_watcher() -> None:
    global _NVCF_WATCH_THREAD
    if _NVCF_WATCH_THREAD is not None:
        return
    if not _is_truthy(os.getenv("NVCF_SECRETS_RELOAD_ENABLE", "1")):
        return

    path = os.getenv("NVCF_SECRETS_PATH", "/var/secrets/secrets.json")
    interval_s = float(os.getenv("NVCF_SECRETS_RELOAD_INTERVAL_SEC", "60"))

    def _loop():
        while True:
            try:
                if os.path.exists(path):
                    new_hash = _file_sha256(path)
                    if new_hash and new_hash != globals().get("_NVCF_LAST_HASH"):
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                secrets = json.load(f)
                            _apply_nvcf_secrets(secrets, initial=False)
                            globals()["_NVCF_LAST_HASH"] = new_hash
                            logger.info("Reloaded NVCF secrets from %s", path)
                        except Exception:
                            pass
                time.sleep(interval_s)
            except Exception:
                try:
                    time.sleep(interval_s)
                except Exception:
                    pass

    t = threading.Thread(target=_loop, name="sglang-nvcf-secrets", daemon=True)
    _NVCF_WATCH_THREAD = t
    try:
        t.start()
    except Exception:
        _NVCF_WATCH_THREAD = None


def _check_collector_health() -> bool:
    health_url = os.getenv("OTEL_HEALTH_CHECK_ENDPOINT")
    if not health_url:
        return True
    try:
        import requests  # type: ignore
    except Exception:
        logger.warning("requests is not available; skipping OTEL collector health-check")
        return True

    timeout_s = float(os.getenv("OTEL_HEALTH_CHECK_TIMEOUT", "10"))
    try:
        resp = requests.get(health_url, timeout=timeout_s)  # type: ignore
        return resp.status_code == 200
    except Exception as e:  # noqa: BLE001
        logger.warning("OTEL collector health-check failed: %s", e)
        return False


def _maybe_wait_for_collector():
    if not _is_truthy(os.getenv("OTEL_HEALTH_CHECK_WAIT", "0")):
        return
    max_retries = int(os.getenv("OTEL_HEALTH_CHECK_RETRIES", "12"))
    backoff_s = float(os.getenv("OTEL_HEALTH_CHECK_BACKOFF", "5"))
    retries = 0
    while retries <= max_retries:
        if _check_collector_health():
            return
        retries += 1
        if retries > max_retries:
            logger.warning("OTEL collector not healthy after %d retries", max_retries)
            return
        time.sleep(backoff_s)


def _get_otlp_exporter(protocol: str, exporter_type: str, exporter_class: str):
    import importlib

    module_name = f"opentelemetry.exporter.otlp.proto.{protocol}.{exporter_type}"
    module = importlib.import_module(module_name)
    return getattr(module, exporter_class)()


def _get_log_exporter():
    protocol = os.getenv("OTEL_EXPORTER_OTLP_LOGS_PROTOCOL", "grpc").lower()
    return _get_otlp_exporter(protocol, "_log_exporter", "OTLPLogExporter")


def _get_metric_exporter():
    protocol = os.getenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "grpc").lower()
    return _get_otlp_exporter(protocol, "metric_exporter", "OTLPMetricExporter")


def _apply_kratos_log_level_from_env() -> None:
    lvl_str = os.getenv("KRATOS_LOG_LEVEL")
    if not lvl_str:
        return
    level = getattr(logging, lvl_str.upper(), logging.WARNING)
    try:
        def _set_levels(l: logging.Logger):
            l.setLevel(level)
            for h in list(l.handlers):
                try:
                    h.setLevel(level)
                except Exception:
                    pass

        _set_levels(logging.getLogger("kratos"))
        for name in list(logging.root.manager.loggerDict.keys()):
            if isinstance(name, str) and name.startswith("kratos"):
                _set_levels(logging.getLogger(name))
    except Exception:
        pass


def _kratos_defaults() -> Dict[str, str]:
    return {
        "namespace": os.getenv("KRATOS_BULKUPLOAD_NAMESPACE", "nim-payload-metrics-byoo-kratos"),
        "profile": os.getenv("KRATOS_BULKUPLOAD_PROFILE", "production"),
        "db_name": os.getenv("KRATOS_BULKUPLOAD_DB_NAME", "nim_payload_metrics_byoo_kratos"),
        "table_name": os.getenv("KRATOS_BULKUPLOAD_TABLE_NAME", "nim_byoo_payloads"),
        "mode": os.getenv("KRATOS_BULKUPLOAD_MODE", "append"),
        "create_table": os.getenv("KRATOS_BULKUPLOAD_CREATE_TABLE", "0"),
    }


def _guess_ext_from_mime(mime: str) -> str:
    m = (mime or "").lower()
    if m.endswith("png") or m == "image/png":
        return "png"
    if m.endswith("jpeg") or m == "image/jpeg":
        return "jpg"
    if m.endswith("jpg") or m == "image/jpg":
        return "jpg"
    if m.endswith("webp") or m == "image/webp":
        return "webp"
    if m.endswith("gif") or m == "image/gif":
        return "gif"
    if m.endswith("mp4") or m == "video/mp4":
        return "mp4"
    if m.endswith("webm") or m == "video/webm":
        return "webm"
    if m.endswith("ogg") or m == "video/ogg":
        return "ogv"
    return "bin"


_DATA_URI_RE = re.compile(r"data:(?P<mime>[\w.+\-\/]+);base64,(?P<b64>[A-Za-z0-9+/=\r\n]+)")


def _extract_first_data_uri(text: str, threshold_bytes: int) -> Optional[Dict[str, Any]]:
    if not text or "base64," not in text:
        return None
    m = _DATA_URI_RE.search(text)
    if not m:
        return None
    b64 = m.group("b64")
    est_bytes = int(len(b64) * 3 / 4)
    if est_bytes < threshold_bytes:
        return None
    return {
        "full_match": m.group(0),
        "mime": m.group("mime"),
        "b64": b64,
    }


def _extract_data_uris(text: str) -> Any:
    if not text or "base64," not in text:
        return []
    out = []
    for m in _DATA_URI_RE.finditer(text):
        out.append({
            "full_match": m.group(0),
            "mime": m.group("mime"),
            "b64": m.group("b64"),
        })
    return out


def _kratos_get_bucket(namespace: str, profile: str) -> Optional[str]:
    if namespace in _KRATOS_BUCKET_CACHE:
        return _KRATOS_BUCKET_CACHE[namespace]
    try:
        from kratos.storage_metadata import get_storage_metadata  # type: ignore

        ace_map = {
            "production": "kratos-db-ca2",
            "china-production": "kratos-xp-xpcn",
            "china-staging": "kratos-xp-xpcnstage",
        }
        ace_id = ace_map.get(profile, "kratos-db-ca2")
        scopes = "bulkuploads-read bulkuploads-write"
        resp = get_storage_metadata(
            profile=profile, namespace=namespace, ace_id=ace_id, scopes=scopes
        )
        storage_list = resp.get("storageMetadata", []) if isinstance(resp, dict) else []
        for st in storage_list:
            if st.get("aceId") == ace_id and st.get("isDefault"):
                bucket = st.get("storageName")
                if bucket:
                    _KRATOS_BUCKET_CACHE[namespace] = bucket
                    return bucket
    except Exception:
        return None
    return None


def _kratos_bulk_upload_bytes(data: bytes, fname: str, cfg: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    try:
        from kratos.bulksync import bulk_upload as _bulk_upload  # type: ignore

        _apply_kratos_log_level_from_env()

        def _try_in_dir(base_dir: str) -> Tuple[Optional[str], Optional[str]]:
            with tempfile.TemporaryDirectory(prefix="sglang_otel_kratos_", dir=base_dir) as tmp_dir:
                fpath = os.path.join(tmp_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(data)
                bulk_id = _bulk_upload(
                    namespace=cfg["namespace"],
                    source_path=fpath,
                    enable_ingestion=False,
                    profile=cfg["profile"],
                    mode=cfg["mode"],
                    db_name=cfg["db_name"],
                    table_name=cfg["table_name"],
                    create_table=_is_truthy(cfg["create_table"]),
                )
                bucket = _kratos_get_bucket(cfg["namespace"], cfg["profile"]) or ""
                if bucket:
                    uri = f"s3://{bucket}/bulk-upload-temp-folder/{bulk_id}/{fname}"
                else:
                    uri = f"kratos://{cfg['namespace']}/bulkuploads/{bulk_id}/{fname}"
                return bulk_id, uri

        preferred_tmp = os.getenv("KRATOS_OFFLOAD_TMPDIR") or "/tmp"
        try:
            return _try_in_dir(preferred_tmp)
        except OSError as oe:
            if getattr(oe, 'errno', None) == errno.ENOSPC:
                return None, None
            return None, None
    except Exception:
        return None, None


class _KratosOffloadFilter(logging.Filter):
    def __init__(self, threshold_bytes: int, enabled: bool, cfg: Dict[str, str]):
        super().__init__()
        self.threshold_bytes = threshold_bytes
        self.enabled = enabled
        self.cfg = cfg

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        if not self.enabled:
            return True
        try:
            targets: Dict[str, Any] = {}
            setters: Dict[str, Any] = {}

            if isinstance(record.msg, str):
                targets["__message__"] = record.getMessage()

                def _set_msg(val, rec=record):
                    rec.msg = val

                setters["__message__"] = _set_msg

            for k, v in list(record.__dict__.items()):
                if isinstance(v, str):
                    targets[k] = v
                    setters[k] = (lambda key: (lambda val, rec=record: rec.__dict__.__setitem__(key, val)))(k)
                elif isinstance(v, list):
                    for idx, elem in enumerate(v):
                        if isinstance(elem, str):
                            key = f"{k}[{idx}]"
                            targets[key] = elem

                            def _make_setter(field, i):
                                def _setter(val, rec=record):
                                    rec.__dict__[field][i] = val

                                return _setter

                            setters[key] = _make_setter(k, idx)

            total_est_bytes = 0
            items = []
            for key, text in targets.items():
                matches = _extract_data_uris(text)
                for m in matches:
                    est_bytes = int(len(m["b64"]) * 3 / 4)
                    total_est_bytes += est_bytes
                    items.append((key, text, m, est_bytes))

            if total_est_bytes < self.threshold_bytes or not items:
                return True

            items.sort(key=lambda x: x[3], reverse=True)
            remaining = total_est_bytes
            updated_texts: Dict[str, str] = {}
            for key, text, m, est_bytes in items:
                if remaining < self.threshold_bytes:
                    break
                try:
                    b = base64.b64decode(m["b64"], validate=False)
                except Exception:
                    continue
                ext = _guess_ext_from_mime(m["mime"]) or "bin"
                fname = f"payload_{uuid.uuid4().hex}.{ext}"
                digest = hashlib.sha256(b).hexdigest()
                uri = _OFFLOAD_CACHE.get(digest)
                bulk_id = None
                if uri is None:
                    bulk_id, uri = _kratos_bulk_upload_bytes(b, fname, self.cfg)
                    if uri:
                        if len(_OFFLOAD_CACHE) > 1024:
                            try:
                                _OFFLOAD_CACHE.pop(next(iter(_OFFLOAD_CACHE)))
                            except Exception:
                                _OFFLOAD_CACHE.clear()
                        _OFFLOAD_CACHE[digest] = uri
                if uri:
                    replacement = f"[offloaded:{uri}]"
                    base = updated_texts.get(key, text)
                    updated_texts[key] = base.replace(m["full_match"], replacement)
                    remaining -= est_bytes
                    if bulk_id is not None:
                        record.__dict__["kratos_bulk_upload_id"] = str(bulk_id)
                    record.__dict__["kratos_uri"] = str(uri)
                else:
                    continue

            for key, new_text in updated_texts.items():
                setter = setters.get(key)
                if setter is not None:
                    setter(new_text)
        except Exception:
            return True
        return True


def _wrap_with_queue(logging_handler: logging.Handler) -> logging.Handler:
    global _LOG_QUEUE, _QUEUE_LISTENER
    if _LOG_QUEUE is not None and _QUEUE_LISTENER is not None:
        return QueueHandler(_LOG_QUEUE)

    _LOG_QUEUE = queue.Queue(maxsize=_env_int("KRATOS_OFFLOAD_MAX_QUEUE", 0))

    threshold = _env_int("KRATOS_OFFLOAD_THRESHOLD_BYTES", 262144)
    enabled = _is_truthy(os.getenv("KRATOS_BULKUPLOAD_ENABLE", "0"))
    cfg = _kratos_defaults()
    offload_filter = _KratosOffloadFilter(threshold, enabled, cfg)
    logging_handler.addFilter(offload_filter)

    _QUEUE_LISTENER = QueueListener(_LOG_QUEUE, logging_handler, respect_handler_level=True)
    _QUEUE_LISTENER.daemon = True
    _QUEUE_LISTENER.start()
    return QueueHandler(_LOG_QUEUE)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def init_otel(resource_attributes: Optional[dict] = None) -> Tuple[Optional[object], Optional[object], Optional[logging.Handler]]:
    """
    Initialize OTEL metrics and logs.

    Returns: (meter, tracer, logging_handler)
    tracer is always None here to avoid conflict with sglang tracing.
    """
    try:
        _load_nvcf_secrets_to_env()
        _start_nvcf_secrets_watcher()
        _maybe_wait_for_collector()
        _apply_kratos_log_level_from_env()

        from opentelemetry import metrics as otel_metrics
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
    except Exception as e:  # noqa: BLE001
        logger.warning("OpenTelemetry not available: %s", e)
        return None, None, None

    res_attrs = resource_attributes or {}
    if "service.name" not in res_attrs:
        res_attrs["service.name"] = os.getenv("OTEL_SERVICE_NAME", "sglang")
    if "service.instance.id" not in res_attrs:
        res_attrs["service.instance.id"] = os.getenv("OTEL_SERVICE_INSTANCE_ID", os.getenv("HOSTNAME", "instance-0"))
    resource = Resource.create(res_attrs)

    meter = None
    tracer = None
    logging_handler: Optional[logging.Handler] = None

    metric_endpoint = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
    if metric_endpoint:
        try:
            metric_exporter = _get_metric_exporter()
            metric_reader = PeriodicExportingMetricReader(metric_exporter)
            otel_metrics.set_meter_provider(
                MeterProvider(resource=resource, metric_readers=[metric_reader])
            )
            meter = otel_metrics.get_meter("sglang")
            global _GLOBAL_METER
            _GLOBAL_METER = meter
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to initialize OTEL metrics: %s", e)

    log_endpoint = os.getenv("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
    if log_endpoint:
        try:
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(_get_log_exporter()))
            logging_handler = LoggingHandler(level=logging.INFO, logger_provider=logger_provider)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to initialize OTEL logs: %s", e)
            logging_handler = None
    else:
        logging_handler = logging.StreamHandler(sys.stdout)

    if logging_handler is not None:
        try:
            logging_handler = _wrap_with_queue(logging_handler)
        except Exception:
            pass
    # Print diagnostic info similar to vLLM and emit a test log
    try:
        initialized_components = []
        if metric_endpoint:
            initialized_components.append(f"metrics -> {metric_endpoint}")
        if log_endpoint:
            initialized_components.append(f"logs -> {log_endpoint}")

        if initialized_components:
            msg = (
                "OpenTelemetry initialized successfully!\n"
                f"  Service: {res_attrs.get('service.name', 'sglang')} "
                f"(instance: {res_attrs.get('service.instance.id', 'instance-0')})\n"
                "  Components:\n"
            )
            for comp in initialized_components:
                msg += f"    - {comp}\n"

            # Try to send a test log to verify connectivity
            if logging_handler is not None and log_endpoint:
                try:
                    test_logger = logging.getLogger("sglang.otel_test")
                    test_logger.propagate = False
                    test_logger.setLevel(logging.INFO)
                    test_logger.addHandler(logging_handler)
                    test_logger.info(
                        "OTEL initialization test message",
                        extra={"otel.test": "true", "initialization": "success"},
                    )
                    msg += "Test log sent successfully to collector\n"
                except Exception as e:
                    msg += f"Test log failed (but handler is attached): {e}\n"

            print(msg, file=sys.stderr, flush=True)
        else:
            print(
                "OpenTelemetry initialized but no exporters configured "
                "(set OTEL_EXPORTER_OTLP_*_ENDPOINT env vars)",
                file=sys.stderr,
                flush=True,
            )
    except Exception:
        # Never fail initialization due to diagnostics
        pass

    return meter, tracer, logging_handler


def get_otel_meter():
    return _GLOBAL_METER


def _sanitize_metric_name(name: str) -> str:
    return name.replace(":", "_")


def _labels_to_attributes(labels: Dict[str, str]) -> Dict[str, str]:
    return {str(k): str(v) for k, v in labels.items()}


def start_prom_to_otel_bridge(scrape_url: str, interval_seconds: float = 10.0) -> None:
    global _PROM_BRIDGE_THREAD
    if _PROM_BRIDGE_THREAD is not None:
        return
    meter = get_otel_meter()
    if meter is None:
        return

    def _ensure_gauge(name: str):
        if name in _PROM_GAUGES:
            return
        values_ref = _PROM_GAUGE_VALUES.setdefault(name, {})

        def _callback(observer):  # type: ignore[no-redef]
            obs_list = []
            try:
                from opentelemetry.metrics import Observation
            except Exception:
                return []
            for label_items, val in list(values_ref.items()):
                attrs = {k: v for k, v in label_items}
                obs_list.append(Observation(val, attrs))
            return obs_list

        try:
            inst = meter.create_observable_gauge(name, callbacks=[_callback])
            _PROM_GAUGES[name] = inst
        except Exception:
            _PROM_GAUGES[name] = None

    def _ensure_counter(name: str):
        if name in _PROM_COUNTERS:
            return
        try:
            _PROM_COUNTERS[name] = meter.create_counter(name)
        except Exception:
            _PROM_COUNTERS[name] = None

    def _add_counter_delta(name: str, labels: Dict[str, str], value: float, prev_map: Dict[Tup[str, Tup[Tup[str, str], ...]], float]):
        key = (name, tuple(sorted(labels.items())))
        prev = prev_map.get(key, 0.0)
        delta = value - prev
        if delta < 0:
            delta = value
        prev_map[key] = value
        if delta <= 0:
            return
        inst = _PROM_COUNTERS.get(name)
        if inst is None:
            return
        try:
            inst.add(delta, attributes=_labels_to_attributes(labels))
        except Exception:
            return

    def _scrape_once():
        try:
            with urllib.request.urlopen(scrape_url, timeout=5) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception:
            return
        for family in text_string_to_metric_families(text):
            ftype = family.type or ""
            for sample in family.samples:
                raw_name = sample.name
                name = _sanitize_metric_name(raw_name)
                labels = sample.labels or {}
                value = float(sample.value or 0.0)
                if ftype == "counter":
                    _ensure_counter(name)
                    _add_counter_delta(name, labels, value, _PROM_COUNTER_PREV)
                elif ftype == "gauge":
                    _ensure_gauge(name)
                    if name in _PROM_GAUGE_VALUES:
                        _PROM_GAUGE_VALUES[name][tuple(sorted(labels.items()))] = value
                elif ftype == "histogram":
                    if raw_name.endswith("_count"):
                        _ensure_counter(name)
                        _add_counter_delta(name, labels, value, _PROM_COUNTER_PREV)
                    elif raw_name.endswith("_sum"):
                        _ensure_counter(name)
                        _add_counter_delta(name, labels, value, _PROM_COUNTER_PREV)
                    elif raw_name.endswith("_bucket"):
                        _ensure_counter(name)
                        _add_counter_delta(name, labels, value, _PROM_BUCKET_PREV)
                elif ftype == "summary":
                    if raw_name.endswith("_count") or raw_name.endswith("_sum"):
                        _ensure_counter(name)
                        _add_counter_delta(name, labels, value, _PROM_COUNTER_PREV)

    def _run():
        while True:
            try:
                _scrape_once()
            except Exception:
                pass
            time.sleep(interval_seconds)

    _PROM_BRIDGE_THREAD = threading.Thread(target=_run, name="sglang-prom-bridge", daemon=True)
    _PROM_BRIDGE_THREAD.start()


