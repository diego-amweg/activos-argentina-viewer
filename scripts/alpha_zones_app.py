from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Paths (repo viewer)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_SIGNALS = BASE_DIR / "data" / "processed" / "signals_zones_latest.csv"
DATA_RANKING_V0 = BASE_DIR / "data" / "processed" / "multifactor_ranking_v0.csv"

APP_TITLE = "ACTIVOS-ARGENTINA — Viewer de Zonas Precio / Riesgo"


# -----------------------------
# Cacheable loaders
# -----------------------------
@st.cache_data(show_spinner=False)
def load_signals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("signals_zones_latest.csv: falta columna obligatoria 'date'")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # dist_MA60_pct viene como proporción (0.34 = 34%). Pasamos a %.
    if "dist_MA60_pct" in df.columns:
        df["dist_MA60_pct"] = pd.to_numeric(df["dist_MA60_pct"], errors="coerce") * 100.0
    else:
        df["dist_MA60_pct"] = np.nan

    # Placeholder de descuento: si el precio está por debajo de MA60, lo usamos como "descuento"
    df["discount_pct"] = np.where(df["dist_MA60_pct"] < 0, -df["dist_MA60_pct"], 0.0)

    # Ratio descuento / riesgo (placeholder)
    if "score_total" in df.columns:
        denom = pd.to_numeric(df["score_total"], errors="coerce").replace(0, np.nan)
        df["discount_risk_ratio"] = np.where(df["discount_pct"] > 0, df["discount_pct"] / denom, np.nan)
    else:
        df["discount_risk_ratio"] = np.nan

    # Buckets de riesgo por score_total
    def bucket(score: float) -> str:
        if pd.isna(score):
            return "Sin dato"
        if score < 40:
            return "Bajo"
        if score < 60:
            return "Medio"
        if score < 80:
            return "Alto"
        return "Muy alto"

    if "score_total" in df.columns:
        df["score_bucket"] = pd.to_numeric(df["score_total"], errors="coerce").apply(bucket)
    else:
        df["score_bucket"] = "Sin dato"

    return df


@st.cache_data(show_spinner=False)
def load_ranking_v0(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("multifactor_ranking_v0.csv: falta columna obligatoria 'date'")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Aseguramos tipos numéricos típicos (sin romper si faltan)
    for col in [
        "rank_v0",
        "score_total_v0",
        "score_salida_v0",
        "close_target",
        "days_since_same_level",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# -----------------------------
# Helpers
# -----------------------------
def safe_readable_date_options(dts: pd.Series) -> list[pd.Timestamp]:
    dts = pd.to_datetime(dts, errors="coerce").dropna().sort_values().unique()
    return list(dts)


def base_filter_signals(
    df: pd.DataFrame,
    target_date: pd.Timestamp,
    bucket_filter: str,
    min_discount: float,
) -> pd.DataFrame:
    out = df[df["date"].dt.date == target_date.date()].copy()

    if bucket_filter != "Todos":
        out = out[out["score_bucket"] == bucket_filter]

    if min_discount > 0:
        out = out[out["discount_pct"] >= float(min_discount)]

    return out


def prepare_signals_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df_disp = df.copy()

    # Formateo de porcentajes
    if "dist_MA60_pct" in df_disp.columns:
        df_disp["dist_MA60_pct"] = df_disp["dist_MA60_pct"].map(lambda x: f"{x:.2f} %" if pd.notna(x) else "")
    if "discount_pct" in df_disp.columns:
        df_disp["discount_pct"] = df_disp["discount_pct"].map(lambda x: f"{x:.2f} %" if pd.notna(x) else "")

    cols = [
        "date",
        "ticker",
        "trend_label",
        "score_total",
        "VAT3_norm",
        "dist_MA60_pct",
        "discount_pct",
        "discount_risk_ratio",
        "score_bucket",
        "signal_type",
        "signal_label",
    ]
    cols = [c for c in cols if c in df_disp.columns]
    return df_disp[cols]


def render_disclaimer() -> None:
    st.warning(
        "⚠️ **Aviso importante**: este viewer NO da recomendaciones de inversión. "
        "El **Ranking v0** es solo un **orden de prioridad para mirar tickers**. "
        "Toda decisión requiere validación adicional (riesgo, liquidez, fundamentales, contexto)."
    )


def render_help() -> None:
    with st.expander("¿Cómo usar este viewer? Guía rápida", expanded=False):
        st.markdown(
            """
### Idea general
Este viewer muestra **artefactos pre-calculados** del proyecto *ACTIVOS-ARGENTINA*.

- **Zonas/Señales**: mapa precio/riesgo para monitoreo.
- **Ranking v0**: lista ordenada para **priorizar análisis**, no para operar.

### Botón “Recargar datos”
Si actualizás CSVs en `data/processed/`, apretá **🔄 Recargar datos** para limpiar caché y volver a cargar.
"""
        )


def render_summary(signals_day: pd.DataFrame) -> None:
    if signals_day.empty:
        st.info("No hay señales con los filtros actuales.")
        return

    c1, c2, c3, c4 = st.columns(4)

    tickers = signals_day["ticker"].nunique() if "ticker" in signals_day.columns else len(signals_day)
    c1.metric("Activos distintos con señal", f"{tickers}")

    c2.metric("Cantidad total de señales", f"{len(signals_day)}")

    if "score_total" in signals_day.columns:
        c3.metric("Mediana score_total", f"{float(pd.to_numeric(signals_day['score_total'], errors='coerce').median()):.1f}")
    else:
        c3.metric("Mediana score_total", "ND")

    if "VAT3_norm" in signals_day.columns:
        c4.metric("Mediana VAT3_norm", f"{float(pd.to_numeric(signals_day['VAT3_norm'], errors='coerce').median()):.2f}")
    else:
        c4.metric("Mediana VAT3_norm", "ND")

    # Tendencias
    if "trend_label" in signals_day.columns:
        st.markdown("**Tendencias entre activos con señal:**")
        tbl = signals_day["trend_label"].fillna("ND").value_counts().reset_index()
        tbl.columns = ["trend_label", "count"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)


def render_signals_tab(df_signals: pd.DataFrame) -> None:
    st.subheader("Zonas / Señales")

    if df_signals.empty:
        st.info("No hay señales con los filtros actuales.")
        return

    st.dataframe(prepare_signals_table(df_signals), use_container_width=True, hide_index=True)


def render_ranking_tab(df_rank_all: pd.DataFrame) -> None:
    st.subheader("Ranking v0 (prioridad para análisis)")

    if df_rank_all.empty:
        st.info("No hay datos de ranking disponibles.")
        return

    dates = safe_readable_date_options(df_rank_all["date"])
    if not dates:
        st.info("Ranking: no hay fechas válidas.")
        return

    # SELECTBOX con key único (evita StreamlitDuplicateElementId)
    selected_dt = st.selectbox(
        "Fecha (ranking)",
        options=dates,
        index=len(dates) - 1,
        format_func=lambda x: x.strftime("%Y-%m-%d"),
        key="ranking_date_selectbox",  # << clave única
    )

    # Top N
    top_n = st.slider("Top N", min_value=10, max_value=200, value=50, step=10, key="ranking_top_n_slider")

    day = df_rank_all[df_rank_all["date"].dt.date == selected_dt.date()].copy()
    if day.empty:
        st.info("No hay filas para la fecha seleccionada.")
        return

    # Orden por rank_v0 si existe, si no por score_total_v0 desc
    if "rank_v0" in day.columns:
        day = day.sort_values(["rank_v0"], ascending=True)
    elif "score_total_v0" in day.columns:
        day = day.sort_values(["score_total_v0"], ascending=False)

    day = day.head(int(top_n)).copy()

    # Columnas “amigables” (si existen)
    cols_pref = [
        "date",
        "ticker",
        "rank_v0",
        "score_total_v0",
        "score_salida_v0",
        "trend_label",
        "risk_score_100",
        "tails_score_100",
        "price_position_score_100",
        "close_target",
        "last_same_level_date",
        "days_since_same_level",
        "pos_in_range_60_score_100",
        "dist_MA20_pct",
        "dist_MA60_pct",
        "dist_MA252_pct",
        "VAT3_norm",
        "regime_label",
        "vol_adaptativa_annual_v3",
        "vol_regime_label",
        "pct_days_with_returns",
    ]
    cols = [c for c in cols_pref if c in day.columns]
    st.dataframe(day[cols], use_container_width=True, hide_index=True)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    render_disclaimer()

    # Sidebar: acciones + filtros
    with st.sidebar:
        st.header("Parámetros")

        if st.button("🔄 Recargar datos (borrar caché)", key="btn_clear_cache"):
            st.cache_data.clear()
            st.rerun()

    # Cargas (con manejo de faltantes)
    signals_df = pd.DataFrame()
    ranking_df = pd.DataFrame()

    # Signals
    if DATA_SIGNALS.exists():
        try:
            signals_df = load_signals(str(DATA_SIGNALS))
        except Exception as e:
            st.error(f"Error cargando signals_zones_latest.csv: {e}")
    else:
        st.warning(f"No existe: {DATA_SIGNALS}")

    # Ranking
    if DATA_RANKING_V0.exists():
        try:
            ranking_df = load_ranking_v0(str(DATA_RANKING_V0))
        except Exception as e:
            st.error(f"Error cargando multifactor_ranking_v0.csv: {e}")
    else:
        st.info("Ranking v0: aún no está disponible en el repo viewer (faltaría copiar multifactor_ranking_v0.csv).")

    # Filtros de señales (en sidebar) con keys únicos
    with st.sidebar:
        st.subheader("Zonas / señales")

        if not signals_df.empty and "date" in signals_df.columns:
            available_dates = safe_readable_date_options(signals_df["date"])
        else:
            available_dates = []

        if available_dates:
            default_idx = len(available_dates) - 1
            selected_date = st.selectbox(
                "Fecha objetivo (señales)",
                options=available_dates,
                index=default_idx,
                format_func=lambda x: x.strftime("%Y-%m-%d"),
                key="signals_date_selectbox",  # << clave única
            )
        else:
            selected_date = None
            st.caption("No hay fechas disponibles en señales.")

        bucket_filter = st.selectbox(
            "Filtro de bucket (riesgo)",
            options=["Todos", "Bajo", "Medio", "Alto", "Muy alto", "Sin dato"],
            index=0,
            key="signals_bucket_selectbox",  # << clave única
        )

        min_discount = st.slider(
            "Mínimo descuento (%)",
            min_value=0.0,
            max_value=50.0,
            value=0.0,
            step=0.25,
            key="signals_min_discount_slider",  # << clave única
        )

    # Aplicar filtros de señales (si hay fecha)
    filtered_signals = pd.DataFrame()
    if selected_date is not None and not signals_df.empty:
        filtered_signals = base_filter_signals(
            signals_df,
            target_date=selected_date,
            bucket_filter=bucket_filter,
            min_discount=min_discount,
        )

    # Tabs
    tab_zonas, tab_ranking, tab_help = st.tabs(["Señales / Zonas", "Ranking v0", "Guía"])

    with tab_zonas:
        st.subheader("1. Resumen general")
        render_summary(filtered_signals)

        st.subheader("2. Señales / Zonas a la fecha seleccionada")
        render_signals_tab(filtered_signals)

    with tab_ranking:
        render_ranking_tab(ranking_df)

    with tab_help:
        render_help()


if __name__ == "__main__":
    main()
