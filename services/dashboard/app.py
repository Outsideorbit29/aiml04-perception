"""F7 analyst dashboard (Streamlit).

Reads the case store written by the consumer (cases_db), the fraud rings from F4 (rings_path) and
the scorer's /model and /health endpoints. Analysts confirm or dismiss alerts here; the decision
is written with CaseStore.set_status and becomes the feedback label F6 uses.

Run from the repo root:  streamlit run services/dashboard/app.py
Config: configs/serving.yaml + services/dashboard/dashboard.yaml, env overrides (CASES_DB, ...).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from fraud import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="AIML-01 Fraud Analyst", layout="wide")


@st.cache_data(ttl=60)
def get_config() -> dict[str, Any]:
    return dd.dashboard_config(
        os.environ.get("SERVING_CONFIG", ROOT / "configs" / "serving.yaml"),
        os.environ.get("DASHBOARD_CONFIG", ROOT / "services" / "dashboard" / "dashboard.yaml"),
    )


@st.cache_data(show_spinner=False)
def get_rings(path: str, mtime: float) -> pd.DataFrame:  # mtime busts the cache on change
    return dd.load_rings(path)


def rings_frame(cfg: dict[str, Any]) -> pd.DataFrame:
    p = Path(cfg["rings_path"])
    return get_rings(str(p), p.stat().st_mtime if p.exists() else 0.0)


def fetch_json(url: str, timeout: float) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:  # scorer down is a normal state for the dashboard
        return None


def fmt(v: float | None, spec: str = ".1f", none: str = "–") -> str:
    return none if v is None else format(v, spec)


cfg = get_config()
st.title("Real-time fraud detection: analyst dashboard")
st.caption(f"Case DB `{cfg['cases_db']}` · scorer `{cfg['scorer_url']}`")


@st.fragment(run_every=cfg["refresh_s"])
def live_overview() -> None:
    conn = dd.connect(cfg["cases_db"])
    try:
        ov = dd.overview(
            conn,
            cfg["alert_budget"],
            cfg["throughput_window_s"],
            cfg["latency_last_n"],
            cfg.get("expected_alert_rate"),
        )
        tl = dd.timeline(conn, cfg["timeline_bucket_s"], cfg["timeline_last_s"])
        sc = dd.status_counts(conn)
    finally:
        if conn is not None:
            conn.close()
    if ov["scored"] == 0:
        st.info("No scored transactions yet. Start the consumer and the replay producer.")
        return
    c = st.columns(6)
    c[0].metric("Transactions scored", f"{ov['scored']:,}")
    c[1].metric("Alerts", f"{ov['alerts']:,}")
    c[2].metric(
        "Alert rate",
        f"{ov['alert_rate']:.2%}",
        delta=f"{ov['alert_rate'] - ov['expected_alert_rate']:+.2%} vs "
        f"{ov['expected_alert_rate']:.1%} expected",
        delta_color="inverse",
    )
    c[3].metric(
        "Latency p50 / p95", f"{fmt(ov['latency_p50_ms'])} / {fmt(ov['latency_p95_ms'])} ms"
    )
    c[4].metric("Throughput", f"{ov['throughput_tps']:.1f} tx/s")
    c[5].metric(
        "Reviewed precision",
        fmt(sc["reviewed_precision"], ".1%"),
        help="confirmed / (confirmed + dismissed), from analyst decisions only",
    )
    age = time.time() - ov["last_scored_at"]
    st.caption(
        f"Last transaction scored {age:.0f} s ago. Budget: the threshold is set for at most "
        f"{ov['alert_budget']:.0%} of legitimate transactions flagged; with ~3.5% fraud and ~50% "
        f"recall the expected overall alert rate is {ov['expected_alert_rate']:.1%}."
    )
    if not tl.empty:
        st.line_chart(tl.set_index("t")[["tps", "alerts"]], height=180)


def queue_row(row: pd.Series) -> None:
    tid = int(row["transaction_id"])
    cols = st.columns([1.1, 1.0, 1.6, 2.6, 0.9, 0.9])
    cols[0].markdown(f"**#{tid}**  \nscore **{row['score']:.3f}**")
    gnn = "n/a" if pd.isna(row["gnn_risk"]) else f"{row['gnn_risk']:.3f}"
    cols[1].markdown(f"XGB {row['p_xgb']:.3f}  \nGNN {gnn}")
    ring = "" if pd.isna(row.get("ring_id", float("nan"))) else f"  \nring {int(row['ring_id'])}"
    amount = "?" if pd.isna(row["amount"]) else f"${row['amount']:,.2f}"
    cols[2].markdown(f"{amount} · product {row['product']}{ring}")
    cols[3].markdown(
        f"client `{row['client_key']}`  \ncard `{row['card_key']}` · device `{row['device_key']}`"
    )
    if cols[4].button("Confirm fraud", key=f"c{tid}", type="primary"):
        dd.set_case_status(cfg["cases_db"], tid, "confirmed")
        st.toast(f"#{tid} confirmed as fraud")
        st.rerun()
    if cols[5].button("Dismiss", key=f"d{tid}"):
        dd.set_case_status(cfg["cases_db"], tid, "dismissed")
        st.toast(f"#{tid} dismissed")
        st.rerun()


def alert_queue_tab() -> None:
    left, mid, right = st.columns([1, 1, 1])
    status = left.selectbox("Status", ["open", "confirmed", "dismissed", "all"], index=0)
    min_score = mid.slider("Minimum score", 0.0, 1.0, 0.0, 0.05)
    n_buttons = right.number_input("Rows with actions", 5, 100, 20, 5)
    right.button("Refresh queue")
    conn = dd.connect(cfg["cases_db"])
    try:
        q = dd.alert_queue(
            conn,
            None if status == "all" else status,
            cfg["queue_limit"],
            min_score,
            rings_frame(cfg),
        )
    finally:
        if conn is not None:
            conn.close()
    if q.empty:
        st.info("No alerts with this filter.")
        return
    st.caption(
        "Score = blend of XGBoost (p_xgb) and the client's GNN graph risk; gnn_shift is how much "
        "the graph moved the score. Unseen clients use XGBoost alone."
    )
    if status == "open":
        for _, row in q.head(int(n_buttons)).iterrows():
            queue_row(row)
            st.divider()
    st.dataframe(
        q.drop(columns=["latency_ms"]),
        hide_index=True,
        use_container_width=True,
        column_config={
            "score": st.column_config.ProgressColumn("score", min_value=0.0, max_value=1.0),
            "amount": st.column_config.NumberColumn("amount", format="$%.2f"),
        },
    )
    with st.expander("Change the status of any alert"):
        c = st.columns([2, 2, 1])
        tid = c[0].number_input("TransactionID", min_value=0, step=1)
        new = c[1].selectbox("New status", ["confirmed", "dismissed", "open"])
        if c[2].button("Apply"):
            try:
                dd.set_case_status(cfg["cases_db"], int(tid), new)
                st.success(f"#{int(tid)} -> {new}")
            except KeyError as e:
                st.error(str(e))


def cases_tab() -> None:
    conn = dd.connect(cfg["cases_db"])
    try:
        sc = dd.status_counts(conn)
        ev = dd.evaluation(conn)
    finally:
        if conn is not None:
            conn.close()
    c = st.columns(4)
    c[0].metric("Open", f"{sc['open']:,}")
    c[1].metric("Confirmed fraud", f"{sc['confirmed']:,}")
    c[2].metric("Dismissed", f"{sc['dismissed']:,}")
    c[3].metric("Live precision (reviewed)", fmt(sc["reviewed_precision"], ".1%"))
    st.caption(
        "Analyst decisions are stored as feedback labels (confirmed = 1, dismissed = 0); F6 uses "
        "them to track live precision and trigger retraining."
    )
    with st.expander("Evaluation only: true labels from the replay (hidden from analysts)"):
        st.warning(
            "These numbers use isFraud from the replayed test period. A real analyst would not see "
            "them; they are here to check the model and the analysts' decisions."
        )
        e = st.columns(5)
        e[0].metric("Alert precision", fmt(ev["precision"], ".1%"))
        e[1].metric("Recall", fmt(ev["recall"], ".1%"))
        e[2].metric("False-positive rate", fmt(ev["fpr"], ".2%"))
        e[3].metric("Frauds in stream", "–" if ev["frauds"] is None else f"{ev['frauds']:,}")
        e[4].metric("Analyst agreement", fmt(ev["analyst_agreement"], ".1%"))


def rings_tab() -> None:
    rings = rings_frame(cfg)
    if rings.empty:
        st.info(f"No rings file at `{cfg['rings_path']}`.")
        return
    only_alerts = st.toggle("Only rings with alerts", value=True)
    conn = dd.connect(cfg["cases_db"])
    try:
        summary = dd.ring_summary(conn, rings, only_alerts)
        st.caption(
            f"{rings['ring_id'].nunique():,} candidate rings, {len(rings):,} clients (clients "
            "sharing a specific device fingerprint; see docs/GRAPH.md). Very large rings come from "
            "chains of coarse fingerprints and are weaker evidence."
        )
        if summary.empty:
            st.info("No ring members in the live stream yet.")
            return
        show_eval = st.checkbox("Show true fraud share (evaluation only)", value=False)
        view = summary if show_eval else summary.drop(columns=["fraud_share_eval"])
        st.dataframe(
            view,
            hide_index=True,
            use_container_width=True,
            column_config={
                "alert_share": st.column_config.ProgressColumn(
                    "alert share", min_value=0.0, max_value=1.0, format="%.2f"
                ),
            },
        )
        ring_id = st.selectbox("Ring network", summary["ring_id"].tolist())
        if ring_id is not None:
            dot = dd.ring_graph_dot(conn, rings, int(ring_id), cfg["ring_graph_max_clients"])
            st.graphviz_chart(dot, use_container_width=True)
            st.caption("Ellipses: clients (red = has alerts) · boxes: devices · diamonds: cards.")
    finally:
        if conn is not None:
            conn.close()


def model_tab() -> None:
    base = cfg["scorer_url"].rstrip("/")
    timeout = float(cfg.get("http_timeout_s", 5))
    info, health = fetch_json(f"{base}/model", timeout), fetch_json(f"{base}/health", timeout)
    if info is None:
        st.error(f"Scorer not reachable at {base}.")
        return
    c = st.columns(4)
    c[0].metric("Model", str(info.get("model", "?")))
    c[1].metric("Alert threshold", fmt(info.get("threshold"), ".4f"))
    c[2].metric("XGBoost weight in blend", fmt(info.get("weight_xgb"), ".2f"))
    c[3].metric("Clients with GNN risk", f"{info.get('clients_with_risk', 0):,}")
    st.json({"model": info, "health": health})
    stats = dd.load_consumer_stats(cfg["stats_path"])
    if stats:
        with st.expander("Consumer stage latencies (consumer_stats.json)"):
            st.json(stats)


live_overview()
tabs = st.tabs(["Alert queue", "Cases", "Fraud rings", "Model"])
with tabs[0]:
    alert_queue_tab()
with tabs[1]:
    cases_tab()
with tabs[2]:
    rings_tab()
with tabs[3]:
    model_tab()
