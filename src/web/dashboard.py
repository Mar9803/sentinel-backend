"""
Dashboard FastHTML montata su FastAPI sotto /dashboard.

One-Click Scenario Simulation — interfaccia live collegata a FraudWrapper.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastcore.xml import to_xml
from fasthtml.common import *
from starlette.responses import HTMLResponse

from src.engine.wrapper import FraudWrapper, InMemoryTransactionStore

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

BASE_TS = datetime(2026, 5, 31, 12, 0, 0, tzinfo=timezone.utc)
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "creditcard.csv"

SCENARIOS = (
    {
        "id": "a",
        "label": "Scenario A",
        "title": "Utente Legittimo",
        "description": (
            "Transazione ordinaria in Italia. Nessun muro difensivo "
            "dovrebbe bloccare l'operazione."
        ),
        "accent": "border-emerald-400",
        "button": "bg-emerald-600 hover:bg-emerald-700",
        "badge": "bg-emerald-100 text-emerald-800",
        "endpoint": "/dashboard/simulate/a",
    },
    {
        "id": "b",
        "label": "Scenario B",
        "title": "Frode Sofisticata",
        "description": (
            "Pattern di frode nota con feature V alterate. "
            "XGBoost dovrebbe riconoscere l'impronta storica e bloccare."
        ),
        "accent": "border-amber-400",
        "button": "bg-amber-500 hover:bg-amber-600",
        "badge": "bg-amber-100 text-amber-800",
        "endpoint": "/dashboard/simulate/b",
    },
    {
        "id": "c",
        "label": "Scenario C",
        "title": "Attacco Zero-Day / Geo-Velocity",
        "description": (
            "Pagamento legittimo a Roma pochi minuti prima, poi transazione "
            "da Singapore. Le regole statiche bloccano con Impossible Travel."
        ),
        "accent": "border-violet-400",
        "button": "bg-violet-600 hover:bg-violet-700",
        "badge": "bg-violet-100 text-violet-800",
        "endpoint": "/dashboard/simulate/c",
    },
)


# ---------------------------------------------------------------------------
# FraudWrapper dedicato alla dashboard (store con seed per Scenario C)
# ---------------------------------------------------------------------------


def _load_fraud_v_features() -> dict[str, float]:
    """Feature V1-V28 da una transazione fraudolenta reale (creditcard.csv)."""
    try:
        df = pd.read_csv(DATA_PATH)
        row = df.loc[df["Class"] == 1].iloc[0]
        return {f"V{i}": float(row[f"V{i}"]) for i in range(1, 29)}
    except Exception:
        return {f"V{i}": float((i % 7) * 1.8 - 4.0) for i in range(1, 29)}


def _build_dashboard_store() -> InMemoryTransactionStore:
    store = InMemoryTransactionStore()
    # Seed Scenario C: titolare legittimo paga a Roma ~5 min prima dell'attacco SG
    store.seed_transaction(
        transaction_id="seed-user-c-rome",
        user_id="user-c",
        amount=45.0,
        country="IT",
        timestamp=BASE_TS,
    )
    return store


FRAUD_V_FEATURES = _load_fraud_v_features()
fraud_wrapper = FraudWrapper(store=_build_dashboard_store())


def _payload_scenario_a() -> dict[str, Any]:
    return {
        "transaction_id": "sim-tx-a",
        "user_id": "user-onesto",
        "amount": 45.50,
        "country": "IT",
        "timestamp": BASE_TS.replace(hour=14, minute=0),
    }


def _payload_scenario_b() -> dict[str, Any]:
    payload = {
        "transaction_id": "sim-tx-b",
        "user_id": "user-b",
        "amount": 80.0,
        "country": "IT",
        "timestamp": BASE_TS.replace(hour=15, minute=0),
    }
    payload.update(FRAUD_V_FEATURES)
    return payload


def _payload_scenario_c() -> dict[str, Any]:
    return {
        "transaction_id": "sim-tx-c",
        "user_id": "user-c",
        "amount": 100.0,
        "country": "SG",
        "timestamp": BASE_TS.replace(minute=5),
    }


SCENARIO_PAYLOADS = {
    "a": _payload_scenario_a,
    "b": _payload_scenario_b,
    "c": _payload_scenario_c,
}


# ---------------------------------------------------------------------------
# Componenti UI (FastHTML = HTML-as-Python via tag functions)
# ---------------------------------------------------------------------------


def _format_score(value: float | None) -> str:
    return "N/D" if value is None else f"{value:.4f}"


def _score_bar_colors(value: float | None) -> tuple[str, str, int]:
    """Ritorna (colore fill, colore track, larghezza %) per la micro-barra."""
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
    """Spinner HTMX — visibile durante hx-post, nascosto al completamento."""
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
            "🛡️ I 3 Muri stanno analizzando la transazione...",
            cls="mt-6 animate-pulse text-lg font-semibold text-emerald-300",
        ),
        P(
            "Regole statiche  →  XGBoost  →  Anomaly Detection",
            cls="mt-2 text-sm tracking-wide text-slate-400",
        ),
    )


def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Converte datetime e ordina le chiavi per una lettura più chiara."""
    main_keys = ("transaction_id", "user_id", "amount", "country", "timestamp")
    serialized: dict[str, Any] = {}
    for key in main_keys:
        if key in payload:
            val = payload[key]
            serialized[key] = val.isoformat() if isinstance(val, datetime) else val
    v_keys = sorted(k for k in payload if k.startswith("V"))
    for key in v_keys:
        serialized[key] = payload[key]
    for key, val in payload.items():
        if key not in serialized:
            serialized[key] = val.isoformat() if isinstance(val, datetime) else val
    return serialized


def _payload_inspector(payload: dict[str, Any]) -> Any:
    formatted = json.dumps(_serialize_payload(payload), indent=2, ensure_ascii=False)
    return Details(cls="mt-6 overflow-hidden rounded-xl border border-gray-200 bg-gray-50")(
        Summary(
            "🔍 Ispeziona Payload Tecnico Transazione",
            cls=(
                "cursor-pointer select-none px-5 py-4 text-sm font-semibold "
                "text-gray-700 transition hover:bg-gray-100"
            ),
        ),
        Pre(
            cls=(
                "overflow-x-auto border-t border-gray-700 bg-gray-900 p-5 "
                "font-mono text-xs leading-relaxed text-green-400"
            ),
        )(formatted),
    )


def _decision_badge(decision: str) -> Any:
    if decision == "PASS":
        return Div(cls="rounded-2xl bg-green-500 px-8 py-5 text-center shadow-lg")(
            P("🟢 PASS", cls="text-3xl font-black tracking-wide text-white"),
            P("Transazione approvata", cls="mt-1 text-sm text-green-100"),
        )
    return Div(cls="rounded-2xl bg-red-600 px-8 py-5 text-center shadow-lg")(
        P("🔴 BLOCK", cls="text-3xl font-black tracking-wide text-white"),
        P("Transazione bloccata", cls="mt-1 text-sm text-red-100"),
    )


def _rules_panel(rules: dict) -> Any:
    triggered = rules.get("triggered") or []
    ml_bypassed = rules.get("ml_bypassed", False)

    if triggered:
        rules_list = Ul(cls="mt-2 list-inside list-disc space-y-1 text-sm text-gray-700")(
            *[Li(name.replace("_", " ").title()) for name in triggered]
        )
    else:
        rules_list = P("Nessuna regola scattata.", cls="mt-2 text-sm text-gray-500")

    bypass_cls = (
        "mt-3 inline-block rounded-full px-3 py-1 text-xs font-semibold "
        + ("bg-orange-100 text-orange-800" if ml_bypassed else "bg-blue-100 text-blue-800")
    )
    bypass_label = "ML bypassato (short-circuit)" if ml_bypassed else "ML eseguito"

    return Div(cls="rounded-xl border border-gray-200 bg-gray-50 p-5")(
        H4("Dettagli Regole", cls="text-sm font-bold uppercase tracking-wider text-gray-500"),
        P(
            Span("ml_bypassed: ", cls="font-mono text-gray-600"),
            Span(str(ml_bypassed), cls="font-mono font-bold text-gray-900"),
            cls="mt-2 text-sm",
        ),
        Span(bypass_label, cls=bypass_cls),
        Div(cls="mt-4")(
            P("Regole triggerate:", cls="text-xs font-semibold uppercase text-gray-500"),
            rules_list,
        ),
    )


def _model_score_card(label: str, value: float | None, accent: str) -> Any:
    display = _format_score(value)
    is_active = value is not None
    opacity = "" if is_active else "opacity-60"
    risk_label = "N/D"
    if value is not None:
        if value < 0.3:
            risk_label = "Rischio basso"
        elif value < 0.7:
            risk_label = "Rischio medio"
        else:
            risk_label = "Rischio alto"

    return Div(
        cls=f"rounded-xl border border-gray-200 bg-white p-4 shadow-sm {opacity}",
    )(
        P(label, cls="text-xs font-bold uppercase tracking-wider text-gray-500"),
        P(display, cls=f"mt-2 text-2xl font-mono font-bold {accent}"),
        _score_progress_bar(value),
        P(
            risk_label if is_active else "Non eseguito",
            cls="mt-2 text-xs text-gray-400",
        ),
    )


def _models_grid(models: dict[str, Any] | None) -> Any:
    if models is None:
        models = {}
    return Div(cls="mt-4")(
        H4("Punteggi Modelli", cls="text-sm font-bold uppercase tracking-wider text-gray-500"),
        Div(cls="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3")(
            _model_score_card("XGBoost", models.get("xgb"), "text-amber-600"),
            _model_score_card("Isolation Forest", models.get("isolation_forest"), "text-violet-600"),
            _model_score_card("Autoencoder", models.get("autoencoder"), "text-indigo-600"),
        ),
    )


def _render_simulation_result(
    scenario_title: str,
    result: dict,
    payload: dict[str, Any],
) -> Any:
    """Assembla il pannello risultati iniettato da HTMX in #simulation-results."""
    decision = result.get("decision", "PASS")
    final_score = result.get("final_score", 0.0)
    rules = result.get("rules") or {}
    models = result.get("models")

    return Div(cls="rounded-2xl border border-white/20 bg-white p-6 shadow-2xl")(
        Div(cls="flex flex-col gap-2 border-b border-gray-100 pb-4 sm:flex-row sm:items-center sm:justify-between")(
            Div()(
                H3(f"Risultato — {scenario_title}", cls="text-xl font-bold text-gray-900"),
                P(
                    f"transaction_id: {result.get('transaction_id', '—')} · "
                    f"final_score: {final_score:.4f}",
                    cls="font-mono text-xs text-gray-500",
                ),
            ),
        ),
        Div(cls="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3")(
            Div(cls="lg:col-span-1")(_decision_badge(decision)),
            Div(cls="lg:col-span-2")(
                _rules_panel(rules),
                _models_grid(models),
            ),
        ),
        _payload_inspector(payload),
    )


def _scenario_card(scenario: dict) -> Any:
    return Div(
        cls=(
            "flex flex-col rounded-2xl border-2 bg-white p-6 shadow-lg "
            f"transition hover:shadow-xl {scenario['accent']}"
        ),
    )(
        Span(
            scenario["label"],
            cls=f"text-xs font-bold uppercase tracking-wider px-2 py-1 rounded-full w-fit {scenario['badge']}",
        ),
        H2(scenario["title"], cls="mt-4 text-2xl font-bold text-gray-900"),
        P(scenario["description"], cls="mt-3 flex-1 text-sm leading-relaxed text-gray-600"),
        Button(
            "Avvia simulazione",
            cls=(
                "mt-6 w-full rounded-xl py-3 text-sm font-semibold text-white "
                f"shadow-md transition {scenario['button']}"
            ),
            hx_post=scenario["endpoint"],
            hx_target="#simulation-results",
            hx_swap="innerHTML",
            hx_indicator="#sim-loading",
        ),
    )


def _dashboard_core() -> Any:
    """
    Blocco principale riutilizzabile: griglia scenari + loading + area risultati.
    Usato sia dalla pagina standalone che dalla rotta /embed per Astro.
    """
    return Div(id="sentinel-simulation-widget", cls="w-full")(
        Div(cls="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3")(
            *[_scenario_card(s) for s in SCENARIOS],
        ),
        Div(cls="mt-10 space-y-4")(
            _loading_indicator(),
            Div(id="simulation-results"),
        ),
    )


def _embed_assets() -> Any:
    """
    Asset minimi da includere una volta nella pagina Astro host
    (Tailwind CDN + CSS indicatore HTMX). HTMX è già iniettato da FastHTML sul backend.
    """
    return Div(id="sentinel-embed-assets", cls="hidden")(
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
        HTMX_INDICATOR_CSS,
    )


# ---------------------------------------------------------------------------
# Rotte
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
                        cls="text-3xl font-extrabold tracking-tight text-white md:text-4xl",
                    ),
                    P(
                        "One-Click Scenario Simulation — Regole, XGBoost e Anomaly Detection in tempo reale.",
                        cls="mx-auto mt-3 max-w-2xl text-sm text-slate-300 md:text-base",
                    ),
                ),
            ),
            Main(cls="container mx-auto px-4 py-10")(_dashboard_core()),
            Footer(cls="border-t border-white/10 py-6 text-center text-xs text-slate-500")(
                P("Sentinel MLOps · FastAPI + FastHTML · API /api/v1/predict"),
            ),
        ),
    )


@rt("/embed")
def dashboard_embed():
    """
    Fragment HTML per embedding in Astro — nessun <html>/<body>, solo il widget.

    In Astro (es. mio-portfolio):
      const html = await fetch('http://127.0.0.1:8000/dashboard/embed').then(r => r.text());
      document.getElementById('sentinel-root').innerHTML = html;

    Assicurarsi che la pagina host carichi HTMX (unpkg.com/htmx.org) e Tailwind CDN.
    """
    fragment = Div(cls="w-full")(_embed_assets(), _dashboard_core())
    return HTMLResponse(content=to_xml(fragment), media_type="text/html")


def _run_simulation(scenario_id: str):
    labels = {s["id"]: s["title"] for s in SCENARIOS}
    title = labels.get(scenario_id.lower(), scenario_id.upper())

    builder = SCENARIO_PAYLOADS.get(scenario_id.lower())
    if builder is None:
        return Div(cls="rounded-xl bg-red-100 p-4 text-red-800")(
            P(f"Scenario sconosciuto: {scenario_id}")
        )

    try:
        payload = builder()
        result = fraud_wrapper.predict_all(payload)
    except Exception as exc:
        return Div(cls="rounded-xl bg-red-100 p-4 text-red-800")(
            H4("Errore simulazione", cls="font-bold"),
            P(str(exc), cls="mt-2 font-mono text-sm"),
        )

    return _render_simulation_result(title, result, payload)


@rt("/simulate/a")
def simulate_a():
    return _run_simulation("a")


@rt("/simulate/b")
def simulate_b():
    return _run_simulation("b")


@rt("/simulate/c")
def simulate_c():
    return _run_simulation("c")
