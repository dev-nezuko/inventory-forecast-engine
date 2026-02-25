"""
Forecast desk: a Streamlit view over the pipeline outputs in DATA_DIR.
Run with: streamlit run streamlit_demo.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import DATA_DIR
from src.dashboard import (
    backtest_note,
    call_forecast_api,
    load_artifacts,
    pipeline_stages,
    series_view,
)

INK, PINE, OCHRE, BRICK, MUTED, RULE = (
    "#1f2a26",
    "#2e5e4e",
    "#b8791f",
    "#a4452c",
    "#5c625d",
    "#d9d2c3",
)
CHART_CONFIG = {"displayModeBar": False}
API_URL = os.environ.get("FORECAST_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Inventory forecast desk", layout="wide")

st.html(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, p, li, label, input, td, th, .stMarkdown {{
  font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
h1, h2, h3, h1 span, h2 span, h3 span {{ font-family: 'Fraunces', Georgia, serif !important; color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-size: 2.1rem !important; margin-bottom: 0 !important; }}
.desk-kicker {{ font: 500 0.78rem 'IBM Plex Mono', monospace; text-transform: uppercase;
  letter-spacing: 0.12em; color: {PINE}; margin: 0 0 0.25rem; }}
.desk-lede {{ color: {MUTED}; max-width: 62ch; margin: 0.35rem 0 1.2rem; }}
.desk-note {{ border-left: 3px solid {OCHRE}; background: #fbf6ea; padding: 0.6rem 0.9rem;
  color: {INK}; font-size: 0.92rem; margin: 0.2rem 0 1rem; }}
[data-testid="stMetricValue"] {{ font-family: 'IBM Plex Mono', monospace; color: {INK}; }}
[data-testid="stMetric"] {{ border-top: 2px solid {INK}; padding-top: 0.4rem; }}
.stage-table td, .stage-table th {{ padding: 0.35rem 0.75rem 0.35rem 0; border-bottom: 1px solid {RULE};
  text-align: left; vertical-align: top; }}
.stage-table th {{ font: 500 0.75rem 'IBM Plex Mono', monospace; text-transform: uppercase; color: {MUTED}; }}
.stage-done {{ color: {PINE}; font-weight: 600; }}
.stage-missing {{ color: {BRICK}; font-weight: 600; }}
.stage-stub {{ color: {OCHRE}; font-weight: 600; }}
</style>
"""
)


@st.cache_data(show_spinner="Reading pipeline outputs...")
def _artifacts(data_dir: str, stamp: float):
    return load_artifacts(Path(data_dir))


def _stamp(data_dir: Path) -> float:
    files = list(data_dir.glob("*/*.parquet")) + list(data_dir.glob("*/*.json"))
    return max((f.stat().st_mtime for f in files), default=0.0)


def _chart(hist: pd.DataFrame, fc: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist["week"],
            y=hist["units_sold"],
            name="Actual units",
            mode="lines+markers",
            line=dict(color=INK, width=2),
            marker=dict(size=4),
        )
    )
    if not fc.empty:
        fig.add_trace(
            go.Scatter(
                x=fc["date"],
                y=fc["q0.9"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc["date"],
                y=fc["q0.1"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(184,121,31,0.22)",
                name="10-90% band",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fc["date"],
                y=fc["q0.5"],
                name="Median forecast",
                mode="lines+markers",
                line=dict(color=OCHRE, width=2.5, dash="dot"),
                marker=dict(size=5),
            )
        )
    fig.update_layout(
        title=dict(
            text=title, font=dict(family="Fraunces, Georgia, serif", size=18, color=INK)
        ),
        font=dict(family="IBM Plex Sans, sans-serif", color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
        yaxis=dict(title="Units per week", gridcolor=RULE, rangemode="tozero"),
        xaxis=dict(gridcolor=RULE),
    )
    return fig


def _backtest_section(bt: dict | None) -> None:
    st.subheader("Holdout backtest")
    if not bt:
        st.info(
            "No backtest yet. Run `python -m src.models.train_gbm_quantile` to create one."
        )
        return
    st.html(f'<p class="desk-note">{backtest_note(bt)}</p>')
    labels = {
        "gbm_median": "GBM median",
        "trailing_4w_mean": "Mean of last 4 weeks",
        "same_week_last_year": "Same week last year",
    }
    wape = pd.DataFrame(
        [(labels.get(k, k), v) for k, v in bt["wape"].items()],
        columns=["Forecast", "WAPE"],
    )
    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure(
            go.Bar(
                x=wape["WAPE"],
                y=wape["Forecast"],
                orientation="h",
                marker_color=[
                    PINE if f == "GBM median" else RULE for f in wape["Forecast"]
                ],
                text=[f"{v:.3f}" for v in wape["WAPE"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            title=dict(
                text="WAPE on the holdout (lower is better)", font=dict(size=15)
            ),
            font=dict(family="IBM Plex Sans, sans-serif", color=INK),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=230,
            margin=dict(l=10, r=40, t=40, b=10),
            xaxis=dict(gridcolor=RULE, range=[0, float(wape["WAPE"].max()) * 1.25]),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width="stretch", config=CHART_CONFIG)
    with right:
        cov = bt["interval_coverage_q0.1_q0.9"]
        st.metric(
            "10-90% band coverage",
            f"{cov:.0%}",
            delta=f"{(cov - 0.8) * 100:+.0f} pts vs 80% target",
            delta_color="off",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Quantile": list(bt["pinball_loss"]),
                    "Pinball loss": list(bt["pinball_loss"].values()),
                }
            ),
            hide_index=True,
            width="stretch",
        )
    if cov < 0.75:
        st.caption(
            "The band is too narrow: the recursive forecast feeds back medians, so later "
            "weeks do not carry the uncertainty of earlier ones."
        )


def _pipeline_section(a) -> None:
    st.subheader("Pipeline outputs")
    body = "".join(
        f'<tr><td>{stage}</td><td class="stage-{status}">{status}</td><td>{detail}</td></tr>'
        for stage, status, detail in pipeline_stages(a)
    )
    st.html(
        '<table class="stage-table"><thead><tr><th>Stage</th><th>Status</th><th>Output</th>'
        f"</tr></thead><tbody>{body}</tbody></table>"
    )
    if a.weighted is not None and "weight" in a.weighted.columns:
        st.caption(
            "Raking weights use illustrative population targets and do not feed the forecast yet."
        )


def _api_section(sku: str, region: str) -> None:
    st.subheader("Serving API")
    st.write(
        "The FastAPI service returns the same stored quantiles. Point "
        "`FORECAST_API_URL` at it and request the selected series."
    )
    url = st.text_input("API base URL", API_URL)
    weeks = st.slider("Weeks to request", 1, 12, 4)
    if st.button("Request forecast", type="primary"):
        rows, error = call_forecast_api(url, sku, region, weeks)
        if error:
            st.error(error)
        else:
            st.success(f"{len(rows)} rows from POST {url.rstrip('/')}/forecast")
            st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")


def main() -> None:
    data_dir = Path(DATA_DIR)
    a = _artifacts(str(data_dir), _stamp(data_dir))

    st.html(
        '<p class="desk-kicker">Weekly demand · SKU × region</p>'
        "<h1>Inventory forecast desk</h1>"
        '<p class="desk-lede">Recent weekly units and the stored quantile forecast for one '
        "series, the holdout check behind it, and which pipeline outputs exist. The bundled "
        "sample is synthetic.</p>"
    )
    for problem in a.problems:
        st.error(problem)

    pairs = a.series()
    if pairs.empty:
        st.warning(
            f"No weekly features found under {data_dir}. Run the pipeline "
            "(ingest, link, features, train) or set DATA_DIR to a folder that has them."
        )
        _pipeline_section(a)
        return

    with st.sidebar:
        st.markdown("### Series")
        top_sku, top_region = a.busiest_series()
        skus = sorted(pairs["sku"].unique())
        sku = st.selectbox("SKU", skus, index=skus.index(top_sku))
        regions = sorted(pairs.loc[pairs["sku"] == sku, "region"])
        region = st.selectbox(
            "Region",
            regions,
            index=regions.index(top_region) if top_region in regions else 0,
        )
        history = st.slider("Weeks of history", 8, 104, 26, step=2)
        st.caption(f"Data folder: {data_dir.name}")
        if a.forecast_source:
            st.caption(f"Forecast file: {a.forecast_source}")

    hist, fc = series_view(a, sku, region, history)
    k1, k2, k3 = st.columns(3)
    recent = hist["units_sold"].tail(4)
    k1.metric("Last 4 weeks, mean", f"{recent.mean():.1f}" if len(recent) else "n/a")
    if fc.empty:
        k2.metric("Next 4 weeks, median", "n/a")
        k3.metric("Next 4 weeks, 10-90%", "n/a")
    else:
        nxt = fc.head(4)
        k2.metric("Next 4 weeks, median", f"{nxt['q0.5'].mean():.1f}")
        k3.metric(
            "Next 4 weeks, 10-90%", f"{nxt['q0.1'].mean():.0f}–{nxt['q0.9'].mean():.0f}"
        )

    st.plotly_chart(
        _chart(hist, fc, f"{sku} in {region}"), width="stretch", config=CHART_CONFIG
    )
    if fc.empty:
        st.info(
            "No stored forecast for this series. Run `python -m src.models.train_gbm_quantile` "
            "and then the ensemble step."
        )

    _backtest_section(a.backtest)
    _pipeline_section(a)
    _api_section(sku, region)


main()
