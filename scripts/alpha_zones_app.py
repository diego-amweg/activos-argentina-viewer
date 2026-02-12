# /scripts/alpha_zones_app.py
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Config
# =============================
st.set_page_config(page_title="Activos Argentina — Zonas + Ranking", layout="wide")

# Data paths (en el repo viewer)
DATA_DIR = os.path.join("data", "processed")
ZONES_PATH = os.path.join(DATA_DIR, "signals_zones_latest.csv")
RANKING_PATH = os.path.join(DATA_DIR, "multifactor_ranking_v0.csv")

APP_TITLE = "Activos Argentina — Zonas + Ranking (Viewer)"
DISCLAIMER = (
    "⚠️ **IMPORTANTE:** Este dashboard **NO es una recomendación** ni asesoramiento financiero. "
    "Es un **orden de prioridad para mirar tickers** y entender contexto (riesgo/tendencia/zonas). "
    "Toda decisión debe validarse con análisis adicional y tu propio criterio."
)

# =============================
# Helpers
# =============================
def _file_info(path: str) -> str:
    if not os.path.exists(path):
        return "No encontrado"
    ts = os.path.getmtime(path)
    # Mostrar en horario local (Streamlit Cloud suele estar en UTC; igual es informativo)
    return f"Existe • mtime={datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')}"

@st.cache_data(show_spinner=False)
def _read_csv_cached(path: str) -> pd.DataFrame:
    # Cachea lectura; se limpia con el botón.
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")

def _as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _add_basic_types_zones(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "date" in df.columns:
        df["date"] = _safe_to_datetime(df["date"])
    for col in ["dist_MA60_pct", "score_total", "VAT3_norm", "discount_pct", "discount_risk_ratio"]:
        if col in df.columns:
            df[col] = _as_float(df[col])
    return df

def _add_basic_types_ranking(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "date" in df.columns:
        df["date"] = _safe_to_datetime(df["date"])
    # numéricos típicos del ranking
    num_cols = [
        "rank_v0", "score_total_v0", "score_salida_v0",
        "trend_score_raw", "trend_score_pos",
        "risk_score_100", "tails_score_100", "liquidity_proxy_score_100",
        "price_position_score_100", "history_quality_score_100",
        "close_target", "days_since_same_level",
        "pos_in_range_60", "pos_in_range_60_score_100",
        "dist_MA20_pct", "dist_MA60_pct", "dist_MA252_pct",
        "slope_20_log", "slope_60_log", "slope_252_log",
        "score_total", "score_vol", "score_tails", "score_illiq", "VAT3_norm",
        "vol_20d_annual", "vol_60d_annual", "vol_252d_annual", "vol_adaptativa_annual_v3",
        "pct_days_with_returns",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = _as_float(df[c])
    # last_same_level_date
    if "last_same_level_date" in df.columns:
        df["last_same_level_date"] = _safe_to_datetime(df["last_same_level_date"])
    return df

def _download_button(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        st.caption("Sin datos para descargar.")
        return
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
    )

# =============================
# UI: Header
# =============================
st.title(APP_TITLE)
st.info(DISCLAIMER)

col_a, col_b, col_c = st.columns([2, 2, 1])
with col_a:
    st.caption(f"Zonas: `{ZONES_PATH}` → {_file_info(ZONES_PATH)}")
with col_b:
    st.caption(f"Ranking v0: `{RANKING_PATH}` → {_file_info(RANKING_PATH)}")
with col_c:
    if st.button("🔄 Recargar datos (borrar caché)", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# =============================
# Load data
# =============================
zones_raw = _read_csv_cached(ZONES_PATH)
ranking_raw = _read_csv_cached(RANKING_PATH)

zones = _add_basic_types_zones(zones_raw)
ranking = _add_basic_types_ranking(ranking_raw)

tabs = st.tabs(["📍 Zonas (signals_zones_latest)", "🏁 Ranking multifactor v0"])

# =============================
# TAB 1: Zonas
# =============================
with tabs[0]:
    st.subheader("Zonas (signals_zones_latest.csv)")

    if zones.empty:
        st.warning("No hay datos de zonas. Verificá que el archivo exista y tenga contenido.")
    else:
        # Controles
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])

        with c1:
            dates = sorted(zones["date"].dropna().dt.date.unique()) if "date" in zones.columns else []
            date_choice = st.selectbox("Fecha", options=dates[::-1] if dates else [], index=0)
        with c2:
            zone_options = ["(todas)"] + sorted([z for z in zones["zone_label"].dropna().unique()]) if "zone_label" in zones.columns else ["(todas)"]
            zone_choice = st.selectbox("Zona", options=zone_options, index=0)
        with c3:
            trend_options = ["(todas)"] + sorted([t for t in zones["trend_label"].dropna().unique()]) if "trend_label" in zones.columns else ["(todas)"]
            trend_choice = st.selectbox("Trend", options=trend_options, index=0)
        with c4:
            search = st.text_input("Buscar ticker (contiene)", value="")

        df = zones.copy()

        if "date" in df.columns and dates:
            df = df[df["date"].dt.date == date_choice]

        if "zone_label" in df.columns and zone_choice != "(todas)":
            df = df[df["zone_label"] == zone_choice]

        if "trend_label" in df.columns and trend_choice != "(todas)":
            df = df[df["trend_label"] == trend_choice]

        if search.strip():
            if "ticker" in df.columns:
                df = df[df["ticker"].astype(str).str.contains(search.strip(), case=False, na=False)]

        # Orden sugerido
        sort_cols = []
        if "discount_pct" in df.columns:
            sort_cols.append("discount_pct")
        if "score_total" in df.columns:
            sort_cols.append("score_total")
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        st.caption(f"Filas: {len(df)}")
        st.dataframe(df, use_container_width=True, hide_index=True)

        _download_button(df, "signals_zones_filtered.csv", "⬇️ Descargar CSV filtrado")

# =============================
# TAB 2: Ranking v0
# =============================
with tabs[1]:
    st.subheader("Ranking multifactor v0 (multifactor_ranking_v0.csv)")

    if ranking.empty:
        st.warning("No hay datos de ranking. Verificá que el archivo exista y tenga contenido.")
    else:
        # Filtro por fecha (debe ser “del día” = última fecha disponible)
        dates = sorted(ranking["date"].dropna().dt.date.unique()) if "date" in ranking.columns else []
        default_date = dates[-1] if dates else None

        c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 2])

        with c1:
            date_choice = st.selectbox("Fecha", options=dates[::-1] if dates else [], index=0)
        with c2:
            top_n = st.selectbox("Top N", options=[20, 50, 100, 200, 500], index=1)
        with c3:
            only_ok = st.checkbox("Sólo status OK", value=True)
        with c4:
            exclude_nd = st.checkbox("Excluir trend ND", value=True)
        with c5:
            search = st.text_input("Buscar ticker (contiene)", value="")

        df = ranking.copy()

        if "date" in df.columns and dates:
            df = df[df["date"].dt.date == date_choice]

        # status OK
        if only_ok and "status" in df.columns:
            df = df[df["status"] == "OK"]

        # excluir ND
        if exclude_nd and "trend_label" in df.columns:
            df = df[df["trend_label"] != "ND"]

        # búsqueda ticker
        if search.strip() and "ticker" in df.columns:
            df = df[df["ticker"].astype(str).str.contains(search.strip(), case=False, na=False)]

        # ordenar por score_total_v0 desc si existe; si no, por rank_v0 asc
        if "score_total_v0" in df.columns:
            df = df.sort_values(["score_total_v0"], ascending=False)
        elif "rank_v0" in df.columns:
            df = df.sort_values(["rank_v0"], ascending=True)

        # top N
        df = df.head(int(top_n))

        # Columnas recomendadas para visualización (si existen)
        preferred_cols = [
            "date", "ticker", "rank_v0",
            "score_total_v0", "score_salida_v0",
            "trend_label", "trend_score_raw", "trend_score_pos",
            "risk_score_100", "tails_score_100", "liquidity_proxy_score_100",
            "price_position_score_100", "history_quality_score_100",
            "zone_label",
            "close_target", "last_same_level_date", "days_since_same_level",
            "pos_in_range_60", "pos_in_range_60_score_100",
            "dist_MA20_pct", "dist_MA60_pct", "dist_MA252_pct",
            "VAT3_norm", "regime_label",
            "vol_20d_annual", "vol_60d_annual", "vol_252d_annual", "vol_adaptativa_annual_v3",
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        df_view = df[cols].copy() if cols else df

        st.caption(f"Fecha: {date_choice} • Filas mostradas: {len(df_view)}")
        st.dataframe(df_view, use_container_width=True, hide_index=True)

        _download_button(df_view, "multifactor_ranking_v0_top.csv", "⬇️ Descargar CSV (Top filtrado)")
