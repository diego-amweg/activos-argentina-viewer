# scripts/alpha_zones_app.py
#
# ACTIVOS-ARGENTINA — Viewer de Zonas Precio / Riesgo
# ---------------------------------------------------
# Esta app:
# - Lee SOLO el archivo data/processed/signals_zones_latest.csv
# - NO depende de trend_metrics.csv, risk_scores_v3.csv ni vat3_metrics_v3.csv
# - Muestra:
#     * Resumen general de señales en la fecha seleccionada
#     * Pestañas por tipo de señal (descuento PREMIUM, SEMI, violento, sobre-extensión, todas)
#     * Filtros por fecha, bucket de riesgo (score_total), mínimo descuento estimado y umbral de sobre-extensión
#
# IMPORTANTE:
# - Este viewer muestra estadísticas sobre señales PRE-CALCULADAS.
# - El motor de cálculo permanece en el repo principal (no visible aquí).

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------------
# Configuración básica de la página
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ACTIVOS-ARGENTINA — Viewer de Zonas Precio / Riesgo",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Utilidades de carga y preparación de datos
# ------------------------------------------------------------------
DATA_PATH = Path("data") / "processed" / "signals_zones_latest.csv"


@st.cache_data(show_spinner=True)
def load_signals(path: Path) -> pd.DataFrame:
    """Carga el archivo de señales pre-calculadas.

    Se espera un CSV con columnas:
    - date
    - ticker
    - trend_label
    - dist_MA60_pct  (distancia vs MA60, en fracción: 0.12 = 12%)
    - score_total
    - VAT3_norm
    - signal_type
    """
    df = pd.read_csv(path)

    # Normalizamos tipos
    df["date"] = pd.to_datetime(df["date"])
    numeric_cols = ["dist_MA60_pct", "score_total", "VAT3_norm"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Definimos un "bucket" de riesgo en base a score_total
    def score_bucket(score: float) -> str:
        if pd.isna(score):
            return "ND"
        if score < 20:
            return "Muy bajo (0–20)"
        if score < 40:
            return "Bajo (20–40)"
        if score < 60:
            return "Medio (40–60)"
        if score < 80:
            return "Alto (60–80)"
        return "Muy alto (≥80)"

    df["score_bucket"] = df["score_total"].apply(score_bucket)

    # Estimamos un "descuento_pct" usando la distancia a MA60:
    # - Si dist_MA60_pct < 0 → precio por debajo de MA60: tomamos |dist| como descuento.
    # - Si dist_MA60_pct ≥ 0 → sobre-extensión alcista: descuento = 0.
    df["discount_pct"] = np.where(
        df["dist_MA60_pct"] < 0,
        -df["dist_MA60_pct"] * 100.0,
        0.0,
    )

    # También dejamos dist_MA60_pct en porcentaje para mostrarlo más cómodo
    df["dist_MA60_pct"] = df["dist_MA60_pct"] * 100.0

    return df


def filter_by_date_and_bucket(
    df: pd.DataFrame, target_date: pd.Timestamp, bucket: str
) -> pd.DataFrame:
    """Filtra por fecha objetivo y bucket de riesgo."""
    df_date = df[df["date"].dt.date == target_date.date()].copy()

    if bucket != "Todos":
        df_date = df_date[df_date["score_bucket"] == bucket].copy()

    return df_date


def apply_additional_filters(
    df: pd.DataFrame,
    min_discount_pct: float,
    overext_threshold_pct: float,
) -> pd.DataFrame:
    """Aplica filtros adicionales comunes.

    - min_discount_pct: descuento mínimo estimado (solo relevante para zonas de descuento).
    - overext_threshold_pct: umbral mínimo de sobre-extensión vs MA60
      (relevante para la pestaña de sobre-extensión).
    """
    df = df.copy()

    # Filtro de descuento mínimo: dejamos pasar señales cuyo "descuento_pct"
    # sea al menos el valor definido.
    if "discount_pct" in df.columns:
        df = df[df["discount_pct"] >= min_discount_pct]

    # El filtro de sobre-extensión se aplicará después por pestaña, para no
    # eliminar señales de otras categorías que no dependen de MA60.
    return df, overext_threshold_pct


def format_percent(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:.2f} %"


def format_number(x: float) -> str:
    if pd.isna(x):
        return "-"
    return f"{x:,.2f}"


def prepare_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara el DataFrame para mostrar en la tabla principal."""
    if df.empty:
        return df

    df = df.copy()

    # Orden sugerida de columnas si existen
    col_order = [
        "ticker",
        "trend_label",
        "signal_type",
        "score_total",
        "VAT3_norm",
        "dist_MA60_pct",
        "discount_pct",
    ]
    cols_present = [c for c in col_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols_present]

    df = df[cols_present + other_cols]

    # Formateo amigable para porcentajes y números
    if "dist_MA60_pct" in df.columns:
        df["dist_MA60_pct"] = df["dist_MA60_pct"].apply(format_percent)
    if "discount_pct" in df.columns:
        df["discount_pct"] = df["discount_pct"].apply(format_percent)
    if "score_total" in df.columns:
        df["score_total"] = df["score_total"].apply(format_number)
    if "VAT3_norm" in df.columns:
        df["VAT3_norm"] = df["VAT3_norm"].apply(format_number)

    return df


# ------------------------------------------------------------------
# Interfaz
# ------------------------------------------------------------------
def sidebar_controls(df: pd.DataFrame) -> Tuple[pd.Timestamp, str, float, float]:
    """Construye los controles de la barra lateral y devuelve sus valores."""
    st.sidebar.header("Parámetros")

    # 1) Fecha objetivo
    all_dates = sorted(df["date"].dt.date.unique())
    default_date = all_dates[-1] if all_dates else None

    selected_date = st.sidebar.selectbox(
        "Fecha objetivo",
        options=all_dates,
        index=len(all_dates) - 1 if all_dates else 0,
        format_func=lambda d: d.strftime("%Y-%m-%d"),
    )
    selected_date = pd.to_datetime(selected_date)

    # 2) Filtro de riesgo (score_total → bucket)
    st.sidebar.subheader("Filtro de riesgo\n(score_total)")
    bucket_options = ["Todos"] + sorted(df["score_bucket"].unique().tolist())
    selected_bucket = st.sidebar.selectbox("Filtro de bucket:", options=bucket_options)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtros adicionales")

    # 3) Mínimo descuento estimado
    min_discount = st.sidebar.slider(
        "Mínimo descuento (%)",
        min_value=0.0,
        max_value=10.0,
        value=0.0,
        step=0.25,
        help="Descuento estimado mínimo vs MA60 para considerar una señal (solo aplica a zonas de descuento).",
    )

    # 4) Umbral de sobre-extensión vs MA60
    overext_threshold = st.sidebar.slider(
        "Umbral de sobre-extensión vs MA60 (%)",
        min_value=8.0,
        max_value=20.0,
        value=12.0,
        step=0.5,
        help="Distancia mínima vs MA60 para considerar una sobre-extensión alcista.",
    )

    return selected_date, selected_bucket, min_discount, overext_threshold


def main() -> None:
    st.title("ACTIVOS-ARGENTINA — Viewer de Zonas Precio / Riesgo")
    st.caption(
        "Visualización privada de señales estadísticas pre-calculadas. "
        "El motor de cálculo permanece en el repositorio principal (no visible aquí)."
    )

    # ------------------------------------------------------------------
    # Carga de datos
    # ------------------------------------------------------------------
    try:
        df_signals = load_signals(DATA_PATH)
    except FileNotFoundError:
        st.error(
            f"Error al cargar archivos: No se encontró el archivo: {DATA_PATH.as_posix()}"
        )
        st.stop()

    if df_signals.empty:
        st.warning("No hay datos de señales disponibles en el archivo actual.")
        st.stop()

    # ------------------------------------------------------------------
    # Controles de la barra lateral
    # ------------------------------------------------------------------
    (
        selected_date,
        selected_bucket,
        min_discount_pct,
        overext_threshold_pct,
    ) = sidebar_controls(df_signals)

    # ------------------------------------------------------------------
    # Filtros principales
    # ------------------------------------------------------------------
    df_filtered = filter_by_date_and_bucket(df_signals, selected_date, selected_bucket)
    df_filtered, overext_threshold_pct = apply_additional_filters(
        df_filtered, min_discount_pct, overext_threshold_pct
    )

    # Si no hay señales después de filtros, avisamos y salimos
    if df_filtered.empty:
        st.info(
            "No hay señales para los filtros actuales. "
            "Ajustá la fecha, el bucket de riesgo o el mínimo de descuento para ver resultados."
        )
        st.stop()

    # ------------------------------------------------------------------
    # 1. Resumen general
    # ------------------------------------------------------------------
    st.markdown("### 1. Resumen general")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        activos_distintos = df_filtered["ticker"].nunique()
        st.metric("Activos distintos con señal en la fecha", activos_distintos)

    with col2:
        total_senales = len(df_filtered)
        st.metric("Cantidad total de señales en la fecha", total_senales)

    with col3:
        med_score = float(df_filtered["score_total"].median())
        st.metric(
            "Mediana score_total (sobre universo con señal)",
            f"{med_score:,.1f}",
        )

    with col4:
        med_vat3 = float(df_filtered["VAT3_norm"].median())
        st.metric(
            "Mediana VAT3_norm (sobre universo con señal)",
            f"{med_vat3:,.2f}",
        )

    # Tabla de tendencias entre activos con señal
    st.markdown("")
    st.markdown("Tendencias entre activos con señal:")

    trend_counts = (
        df_filtered.groupby("trend_label")["ticker"].nunique().reset_index(name="count")
    )
    st.dataframe(trend_counts, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ------------------------------------------------------------------
    # 2. Señales / Zonas a la fecha seleccionada
    # ------------------------------------------------------------------
    st.markdown("### 2. Señales / Zonas a la fecha seleccionada")

    tab_premium, tab_semi, tab_violento, tab_overext, tab_todas = st.tabs(
        [
            "Descuento tranquilo PREMIUM",
            "Descuento tranquilo SEMI",
            "Descuento violento (alto riesgo)",
            "Sobre-extensión alcista",
            "Todas las señales",
        ]
    )

    # Helper para mostrar tabla + mensaje vacío
    def show_signals_tab(df_tab: pd.DataFrame, descripcion: str) -> None:
        st.markdown(descripcion)
        if df_tab.empty:
            st.info("No hay activos en esta categoría con los filtros actuales.")
            return
        st.dataframe(
            prepare_df_for_display(df_tab),
            use_container_width=True,
            hide_index=True,
        )

    # 2.1 Descuento tranquilo PREMIUM
    with tab_premium:
        df_prem = df_filtered[df_filtered["signal_type"] == "descuento_tranquilo_premium"]
        desc = (
            "**Descuento tranquilo PREMIUM**\n\n"
            "Alcista + por debajo de MA60 (descuento moderado/alto) + "
            "score_total elevado (bajo riesgo relativo) + colas suaves. "
            "Ordenadas por mejor relación descuento / riesgo."
        )
        show_signals_tab(df_prem, desc)

    # 2.2 Descuento tranquilo SEMI
    with tab_semi:
        df_semi = df_filtered[df_filtered["signal_type"] == "descuento_tranquilo_semi"]
        desc = (
            "**Descuento tranquilo SEMI**\n\n"
            "Alcista + por debajo de MA60 + score_total medio/alto. "
            "Condiciones algo más laxas que la categoría PREMIUM."
        )
        show_signals_tab(df_semi, desc)

    # 2.3 Descuento violento (alto riesgo)
    with tab_violento:
        df_viol = df_filtered[
            df_filtered["signal_type"] == "descuento_violento_alto_riesgo"
        ]
        desc = (
            "**Descuento violento con riesgo elevado**\n\n"
            "Alcista + fuerte descuento vs MA60 + score_total muy alto "
            "(riesgo elevado / volatilidad relevante)."
        )
        show_signals_tab(df_viol, desc)

    # 2.4 Sobre-extensión alcista
    with tab_overext:
        df_over = df_filtered[df_filtered["signal_type"] == "sobre_extension_alcista"]

        # Aquí sí aplicamos el umbral de sobre-extensión vs MA60:
        df_over = df_over[df_over["dist_MA60_pct"] >= overext_threshold_pct]

        desc = (
            "**Sobre-extensión alcista**\n\n"
            "Alcista + muy por encima de MA60 (umbral configurable) + "
            "riesgo medio/alto. Posibles candidatos a toma de ganancias "
            "o monitoreo cercano."
        )
        show_signals_tab(df_over, desc)

    # 2.5 Todas las señales
    with tab_todas:
        desc = (
            "**Todas las señales combinadas**\n\n"
            "Se muestran todas las señales para la fecha y filtros actuales, "
            "independientemente del tipo de zona."
        )
        show_signals_tab(df_filtered, desc)

    st.markdown("---")
    st.caption(
        "Notas: los parámetros y percentiles de riesgo se calculan sobre el universo "
        "de señales disponibles en el archivo actual. Este viewer no reemplaza el "
        "análisis fundamental ni constituye recomendación de inversión."
    )


if __name__ == "__main__":
    main()
