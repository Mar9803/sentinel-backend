"""
Dashboard FastHTML montata su FastAPI sotto /dashboard.

Simulazione a flusso continuo con HTMX polling.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

from fastcore.xml import to_xml
from fasthtml.common import *
from starlette.responses import HTMLResponse

from src.engine.wrapper import FraudWrapper
from src.web.stream_generator import TransactionStreamGenerator
from src.web.stream_metrics import stream_metrics

# Host backend per HTMX quando la dashboard è embeddata in Astro (:4321).
# Path relativi risolverebbero sulla origine della pagina host → richieste sulla porta sbagliata.
BACKEND_BASE_URL = os.getenv("SENTINEL_BACKEND_URL", "http://localhost:8000").rstrip("/")
DASHBOARD_PREFIX = "/dashboard"

TAILWIND_CDN = Link(
    rel="stylesheet",
    href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css",
)

HTMX_INDICATOR_CSS = Style(
    """
    .htmx-indicator { opacity: 0; transition: opacity 300ms ease-in; }
    .htmx-request .htmx-indicator,
    .htmx-request.htmx-indicator { opacity: 1; }
    """
)

dashboard_app, rt = fast_app(
    hdrs=(TAILWIND_CDN, HTMX_INDICATOR_CSS),
    pico=False,
    title="SentinelGraph Live Simulation",
)

stream_generator = TransactionStreamGenerator()
fraud_wrapper = FraudWrapper()


def _p(path: str) -> str:
    """URL assoluto verso FastAPI (es. http://127.0.0.1:8000/dashboard/stream/tick)."""
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{BACKEND_BASE_URL}{DASHBOARD_PREFIX}{path}"


def _html_response(*parts: Any) -> HTMLResponse:
    return HTMLResponse("".join(to_xml(p) for p in parts), media_type="text/html")


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if not str(k).startswith("_")}


def run_inference(payload: dict[str, Any]) -> dict[str, Any]:
    return fraud_wrapper.predict_all(_clean_payload(payload))


# ---------------------------------------------------------------------------
# KPI + controlli + feed
# ---------------------------------------------------------------------------


def _kpi_card(kpi_id: str, label: str, value: str, accent: str) -> Any:
    return Div(
        id=kpi_id,
        cls="rounded-xl border border-slate-600/60 bg-slate-800/90 p-4 text-center",
    )(
        P(label, cls="text-xs font-bold uppercase tracking-wider text-slate-400"),
        P(value, cls=f"mt-2 text-2xl font-black {accent}"),
    )


def _kpi_panel() -> Any:
    m = stream_metrics
    return Div(
        id="kpi-panel",
        cls="mb-6 grid grid-cols-2 gap-3 md:grid-cols-4",
        hx_swap_oob="true",
    )(
        _kpi_card("kpi-analyzed", "Transactions analyzed", str(m.transactions_analyzed), "text-white"),
        _kpi_card(
            "kpi-losses-avoided",
            "Losses avoided",
            f"€ {m.losses_avoided_eur:,.2f}",
            "text-emerald-400",
        ),
        _kpi_card("kpi-damages", "Losses incurred", f"€ {m.damages_eur:,.2f}", "text-red-400"),
        _kpi_card(
            "kpi-vanity-accuracy",
            "Vanity accuracy",
            f"{m.vanity_accuracy_pct:.1f}%",
            "text-amber-300",
        ),
    )


def _stream_controls(running: bool) -> Any:
    if running:
        buttons = Div(cls="mt-4 flex flex-wrap gap-3")(
            Button(
                "⏹ Stop stream",
                cls=(
                    "rounded-xl bg-red-600 px-5 py-2.5 text-sm font-semibold text-white "
                    "shadow-md hover:bg-red-700 transition"
                ),
                hx_post=_p("/stream/stop"),
                hx_target="#stream-controls-slot",
                hx_swap="innerHTML",
            ),
            Span(
                "Active polling · 1 tick/s",
                cls="inline-flex items-center rounded-full bg-emerald-900/50 px-3 py-1 text-xs text-emerald-300",
            ),
        )
    else:
        buttons = Div(cls="mt-4 flex flex-wrap gap-3")(
            Button(
                "▶ Start stream",
                cls=(
                    "rounded-xl bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white "
                    "shadow-md hover:bg-emerald-700 transition"
                ),
                hx_post=_p("/stream/start"),
                hx_target="#stream-controls-slot",
                hx_swap="innerHTML",
                hx_indicator="#stream-boot-loading",
            ),
            Button(
                "↻ Reset metrics",
                cls=(
                    "rounded-xl border border-slate-500 px-5 py-2.5 text-sm "
                    "font-semibold text-slate-300 hover:bg-slate-700 transition"
                ),
                hx_post=_p("/stream/stop"),
                hx_target="#stream-controls-slot",
                hx_swap="innerHTML",
            ),
        )

    return Div(
        cls="rounded-2xl border border-slate-600/50 bg-slate-800/80 p-6 backdrop-blur-sm",
    )(
        H2("Live transaction stream", cls="text-lg font-bold text-white"),
        P(
            "HTMX polling across 3 walls — Rules, XGBoost, Anomaly Detection.",
            cls="mt-2 text-sm text-slate-400",
        ),
        buttons,
    )


def _transaction_feed_container(running: bool) -> Any:
    base_cls = (
        "mt-3 max-h-96 overflow-y-auto rounded-xl border border-slate-600 "
        "bg-slate-900/50 p-3 space-y-2"
    )
    if running:
        return Div(
            id="transaction-feed",
            cls=base_cls,
            hx_get=_p("/stream/tick"),
            hx_trigger="every 1s",
            hx_swap="afterbegin",
            hx_swap_oob="true",
        )()
    return Div(
        id="transaction-feed",
        cls=base_cls + " min-h-[120px] flex items-center justify-center",
        hx_swap_oob="true",
    )(
        P("Waiting for the first tick…", cls="text-sm text-slate-500"),
    )


def _feed_row(payload: dict[str, Any], result: dict[str, Any]) -> Any:
    meta = payload.get("_stream_meta", {})
    profile = meta.get("label", "Transaction")
    decision = result.get("decision", "PASS")
    amount = payload.get("amount", 0)
    is_block = decision == "BLOCK"

    row_cls = (
        "rounded-lg border-l-4 p-3 shadow-sm "
        + ("border-red-500 bg-red-950/40" if is_block else "border-emerald-500 bg-emerald-950/30")
    )
    badge_cls = (
        "rounded-full px-2 py-0.5 text-xs font-bold "
        + ("bg-red-600 text-white" if is_block else "bg-emerald-600 text-white")
    )

    return Div(cls=row_cls)(
        Div(cls="flex flex-wrap items-center justify-between gap-2")(
            Div(cls="flex items-center gap-2")(
                Span(decision, cls=badge_cls),
                Span(profile, cls="text-xs text-slate-400"),
            ),
            Span(f"€ {amount:,.2f}", cls="font-mono text-sm font-semibold text-white"),
        ),
        P(
            f"{result.get('transaction_id', '—')} · score {result.get('final_score', 0):.4f}",
            cls="mt-1 font-mono text-xs text-slate-400",
        ),
    )


# ---------------------------------------------------------------------------
# Componenti UI dettaglio (ultima analisi)
# ---------------------------------------------------------------------------


def _format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _score_bar_colors(value: float | None) -> tuple[str, str, int]:
    if value is None:
        return "bg-gray-300", "bg-gray-200", 0
    pct = int(min(100, max(0, round(value * 100))))
    if value < 0.3:
        return "bg-green-500", "bg-gray-200", pct
    if value < 0.7:
        return "bg-orange-500", "bg-gray-200", pct
    return "bg-red-500", "bg-gray-200", pct


def _score_progress_bar(value: float | None) -> Any:
    fill_cls, track_cls, pct = _score_bar_colors(value)
    if value is None:
        return Div(cls=f"mt-3 h-2.5 w-full rounded-full {track_cls} opacity-40")
    return Div(cls=f"mt-3 h-2.5 w-full overflow-hidden rounded-full {track_cls}")(
        Div(
            cls=f"h-full rounded-full transition-all duration-700 ease-out {fill_cls}",
            style=f"width: {pct}%",
        ),
    )


def _loading_indicator() -> Any:
    return Div(
        id="sim-loading",
        cls=(
            "htmx-indicator rounded-2xl border border-slate-600/60 "
            "bg-slate-800/95 p-10 text-center shadow-xl backdrop-blur-sm"
        ),
    )(
        Div(
            cls=(
                "mx-auto h-14 w-14 animate-spin rounded-full border-4 "
                "border-slate-600 border-t-emerald-400"
            ),
        ),
        P(
            "🛡️ The 3 walls are analyzing the transaction…",
            cls="mt-6 animate-pulse text-lg font-semibold text-emerald-300",
        ),
    )


def _stream_boot_loader() -> Any:
    return Div(
        id="stream-boot-loading",
        cls=(
            "htmx-indicator rounded-xl border border-slate-600/60 "
            "bg-slate-800/90 px-4 py-3 shadow-md"
        ),
    )(
        Div(cls="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-4")(
            P(
                "Request sent — waking up the anti-fraud engine…",
                cls="text-sm font-medium text-emerald-300 sm:shrink-0",
            ),
            Div(cls="h-2.5 min-w-[120px] flex-1 overflow-hidden rounded-full bg-gray-700")(
                Div(cls="h-full w-full rounded-full bg-emerald-400 animate-pulse"),
            ),
        ),
    )


def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    main_keys = ("transaction_id", "user_id", "amount", "country", "timestamp")
    serialized: dict[str, Any] = {}
    for key in main_keys:
        if key in payload:
            val = payload[key]
            serialized[key] = val.isoformat() if isinstance(val, datetime) else val
    for key in sorted(k for k in payload if k.startswith("V")):
        serialized[key] = payload[key]
    for key, val in payload.items():
        if str(key).startswith("_"):
            continue
        if key not in serialized:
            serialized[key] = val.isoformat() if isinstance(val, datetime) else val
    return serialized


def _payload_inspector(payload: dict[str, Any]) -> Any:
    formatted = json.dumps(_serialize_payload(payload), indent=2, ensure_ascii=False)
    return Details(cls="mt-6 overflow-hidden rounded-xl border border-gray-200 bg-gray-50")(
        Summary(
            "🔍 Inspect transaction technical payload",
            cls="cursor-pointer select-none px-5 py-4 text-sm font-semibold text-gray-700",
        ),
        Pre(
            cls="overflow-x-auto border-t border-gray-700 bg-gray-900 p-5 font-mono text-xs text-green-400",
        )(formatted),
    )


def _decision_badge(decision: str) -> Any:
    if decision == "PASS":
        return Div(cls="rounded-2xl bg-green-500 px-8 py-5 text-center shadow-lg")(
            P("🟢 PASS", cls="text-3xl font-black text-white"),
        )
    return Div(cls="rounded-2xl bg-red-600 px-8 py-5 text-center shadow-lg")(
        P("🔴 BLOCK", cls="text-3xl font-black text-white"),
    )


def _rules_panel(rules: dict) -> Any:
    triggered = rules.get("triggered") or []
    ml_bypassed = rules.get("ml_bypassed", False)
    rules_list = (
        Ul(cls="mt-2 list-disc list-inside text-sm text-gray-700")(
            *[Li(n.replace("_", " ").title()) for n in triggered]
        )
        if triggered
        else P("No rules triggered.", cls="mt-2 text-sm text-gray-500")
    )
    return Div(cls="rounded-xl border border-gray-200 bg-gray-50 p-5")(
        H4("Rule details", cls="text-sm font-bold uppercase text-gray-500"),
        P(f"ml_bypassed: {ml_bypassed}", cls="mt-2 font-mono text-sm"),
        rules_list,
    )


def _model_score_card(label: str, value: float | None, accent: str) -> Any:
    return Div(cls="rounded-xl border border-gray-200 bg-white p-4 shadow-sm")(
        P(label, cls="text-xs font-bold uppercase text-gray-500"),
        P(_format_score(value), cls=f"mt-2 text-2xl font-mono font-bold {accent}"),
        _score_progress_bar(value),
    )


def _models_grid(models: dict[str, Any] | None) -> Any:
    models = models or {}
    return Div(cls="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3")(
        _model_score_card("XGBoost", models.get("xgb"), "text-amber-600"),
        _model_score_card("Isolation Forest", models.get("isolation_forest"), "text-violet-600"),
        _model_score_card("Autoencoder", models.get("autoencoder"), "text-indigo-600"),
    )


def render_transaction_result(label: str, result: dict, payload: dict[str, Any]) -> Any:
    return Div(
        id="simulation-results",
        cls="rounded-2xl border border-white/20 bg-white p-6 shadow-2xl",
        hx_swap_oob="true",
    )(
        Div(cls="border-b border-gray-100 pb-4")(
            H3(f"Latest analysis — {label}", cls="text-xl font-bold text-gray-900"),
            P(
                f"{result.get('transaction_id')} · score {result.get('final_score', 0):.4f}",
                cls="font-mono text-xs text-gray-500",
            ),
        ),
        Div(cls="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3")(
            Div(cls="lg:col-span-1")(_decision_badge(result.get("decision", "PASS"))),
            Div(cls="lg:col-span-2")(
                _rules_panel(result.get("rules") or {}),
                _models_grid(result.get("models")),
            ),
        ),
        _payload_inspector(_clean_payload(payload)),
    )


def _dashboard_core() -> Any:
    return Div(id="sentinel-simulation-widget", cls="w-full")(
        _kpi_panel(),
        Div(id="stream-controls-slot")(_stream_controls(running=False)),
        _stream_boot_loader(),
        Div(cls="mt-6")(
            H3("Transaction feed", cls="text-sm font-bold uppercase tracking-wider text-slate-400"),
            _transaction_feed_container(running=False),
        ),
        Div(cls="mt-10 space-y-4")(
            _loading_indicator(),
            Div(id="simulation-results")(
                P(
                    "Latest transaction details will appear here after each tick.",
                    cls="text-center text-sm text-slate-500 py-6",
                ),
            ),
        ),
    )


def _embed_assets() -> Any:
    return Div(id="sentinel-embed-assets", cls="hidden")(
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
        HTMX_INDICATOR_CSS,
    )


# ---------------------------------------------------------------------------
# Rotte pagina
# ---------------------------------------------------------------------------


@rt("/")
def dashboard_home():
    return Titled(
        "SentinelGraph Live Simulation",
        Div(cls="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900")(
            Header(cls="border-b border-white/10 bg-black/20 backdrop-blur-sm")(
                Div(cls="container mx-auto px-4 py-8 text-center")(
                    H1(
                        "🛡️ SentinelGraph Live Simulation",
                        cls="text-3xl font-extrabold text-white md:text-4xl",
                    ),
                    P(
                        "Continuous stream · HTMX polling · 3 defensive walls",
                        cls="mx-auto mt-3 max-w-2xl text-sm text-slate-300",
                    ),
                ),
            ),
            Main(cls="container mx-auto px-4 py-10")(_dashboard_core()),
        ),
    )


@rt("/embed")
def dashboard_embed():
    """
    Fragment HTML per Astro. Tutti gli attributi HTMX usano URL assoluti via _p()
    (BACKEND_BASE_URL + /dashboard/...) per evitare richieste verso :4321.
    """
    fragment = Div(cls="w-full")(_embed_assets(), _dashboard_core())
    return HTMLResponse(content=to_xml(fragment), media_type="text/html")


# ---------------------------------------------------------------------------
# Rotte stream (HTMX)
# ---------------------------------------------------------------------------


def _reinit_fraud_wrapper() -> None:
    global fraud_wrapper
    fraud_wrapper = FraudWrapper()


@rt("/stream/start", methods=["POST"])
def stream_start():
    global fraud_wrapper

    stream_metrics.reset()
    stream_generator.reset()
    _reinit_fraud_wrapper()
    stream_generator.start()

    return _html_response(
        _stream_controls(running=True),
        _kpi_panel(),
        _transaction_feed_container(running=True),
    )


@rt("/stream/tick", methods=["GET"])
def stream_tick():
    if not stream_generator.is_running:
        return HTMLResponse("", media_type="text/html")

    payload = stream_generator.next_transaction()
    meta = payload.get("_stream_meta", {})
    profile = meta.get("profile", "unknown")
    label = meta.get("label", "Live")

    try:
        result = run_inference(payload)
    except Exception as exc:
        return _html_response(
            Div(cls="rounded-lg bg-red-900/50 p-3 text-red-200 text-sm")(str(exc)),
            _kpi_panel(),
        )

    stream_metrics.apply_tick(
        amount=float(payload.get("amount", 0)),
        decision=result.get("decision", "PASS"),
        profile=profile,
    )

    clean = _clean_payload(payload)
    return _html_response(
        _feed_row(payload, result),
        _kpi_panel(),
        render_transaction_result(label, result, clean),
    )


@rt("/stream/stop", methods=["POST"])
def stream_stop():
    stream_generator.stop()
    stream_metrics.reset()

    return _html_response(
        _stream_controls(running=False),
        _kpi_panel(),
        _transaction_feed_container(running=False),
    )
