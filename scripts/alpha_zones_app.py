#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alpha_zones_app.py

Aplicación Streamlit para visualizar:
- Tendencia (trend_metrics.csv)
- Riesgo (risk_scores_v3.csv + vat3_metrics_v3.csv)
- Zonas/Señales:
    * Descuento tranquilo PREMIUM (adaptativo)
    * Descuento tranquilo SEMI (adaptativo)
    * Descuento violento con riesgo elevado
    * Sobre-extensión alcista
- Métrica discount_risk_ratio (descuento / riesgo)
- Pequeños knobs de ajuste:
    * Mínimo descuento vs MA60
    * Umbral de sobre-extensión vs MA60
    * Relajar umbrales de score_total y VAT3_norm

Pensado para correr desde la raíz del proyecto:

    cd /home/diego/projects/activos-argentina-viewer
    streamlit run scripts/alpha_zones_app.py

Requisitos:
    pip install streamlit pandas numpy
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# CONFIGURACIÓN BÁSICA
# --------------------------------------------------------------------

# Cambiá esta contraseña si querés proteger el acceso local.
# En Streamlit Cloud, la seguridad real la da la allow-list de emails.
APP_PASSWORD = "activos2025"

TREND_PATH = "data/processed/trend_metrics.csv"
RISK_PATH = "data/processed/risk_scores_v3.csv"
VAT_PATH = "data/processed/vat3_metrics_v3.csv"


# --------------------------------------------------------------------
# UTILIDADES
# --------------------------------------------------------------------

def check_password() -> bool:
    """
    Pequeño control de acceso.
    No es seguridad fuerte, pero evita acceso casual.
    """
    def password_entered():
        if st.session_state["password"] == APP_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text("Acceso restringido")
        st.text_input(
            "Ingresá la contraseña:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False

    if not st.session_state["password_correct"]:
        st.text("Contraseña incorrecta. Volvé a intentar.")
        st.text_input(
            "Ingresá la contraseña:",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False

    return True


@st.cache_data
def load_csv_with_date(path: str, date_col: str = "date") -> pd.DataFrame:
    """Leer CSV y parsear la columna de fecha con cache."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"El archivo {path} no tiene columna '{date_col}'.")

    df[date_col] = pd.to_datetime(df[date_col])
    return df


def get_common_dates(dfs) -> list:
    """Obtener lista de fechas comunes a todos los dataframes (como datetime.date)."""
    common = None
    for df in dfs:
        dates = set(df["date"].dropna().dt.date.unique())
        if common is None:
            common = dates
        else:
            common = common.intersection(dates)

    if not common:
        return []
    return sorted(common)


def merge_risk_vat(risk_df: pd.DataFrame, vat_df: pd.DataFrame) -> pd.DataFrame:
    """Unir riesgo + VAT3."""
    merged = pd.merge(
        risk_df,
        vat_df[["date", "ticker", "VAT3_norm"]],
        on=["date", "ticker"],
        how="left"
    )
    return merged


def build_latest_signals(
    trend_df: pd.DataFrame,
    risk_vat_df: pd.DataFrame,
    target_date: pd.Timestamp,
    min_discount_pct: float,
    overext_threshold_pct: float,
    relax_score: float,
    relax_vat: float,
) -> pd.DataFrame:
    """
    Construir señales para la fecha objetivo con lógica adaptativa y knobs:

    - descuento_tranquilo_premium
    - descuento_tranquilo_semi
    - descuento_violento_alto_riesgo
    - sobre_extension_alcista

    Parámetros:
    - min_discount_pct: mínimo descuento vs MA60 (ej. 0.0 → cualquier valor < 0).
    - overext_threshold_pct: umbral de sobre-extensión (> este %).
    - relax_score: puntos adicionales para score_p50/p75.
    - relax_vat: extra permitido en VAT3_norm p50/p75.

    Incluye discount_pct y discount_risk_ratio.
    """
    trend_last = trend_df[trend_df["date"] == target_date].copy()
    risk_last = risk_vat_df[
        (risk_vat_df["date"] == target_date) & (risk_vat_df["status"] == "OK")
    ].copy()

    if len(trend_last) == 0 or len(risk_last) == 0:
        return pd.DataFrame()

    # Umbrales adaptativos
    score_series = risk_last["score_total"].dropna()
    vat_series = risk_last["VAT3_norm"].dropna()
    if len(score_series) == 0 or len(vat_series) == 0:
        return pd.DataFrame()

    score_p50, score_p75 = np.percentile(score_series, [50, 75])
    vat_p50, vat_p75 = np.percentile(vat_series, [50, 75])

    # Aplicar relajación de umbrales
    score_p50_adj = score_p50 + relax_score
    score_p75_adj = score_p75 + relax_score
    vat_p50_adj = vat_p50 + relax_vat
    vat_p75_adj = vat_p75 + relax_vat

    merged_last = pd.merge(
        trend_last,
        risk_last[["date", "ticker", "score_total", "VAT3_norm", "status"]],
        on=["date", "ticker"],
        how="inner",
    )

    if len(merged_last) == 0:
        return pd.DataFrame()

    # Descuento y ratio descuento/riesgo
    min_disc_dec = min_discount_pct / 100.0
    overext_dec = overext_threshold_pct / 100.0

    merged_last["discount_pct"] = np.where(
        merged_last["dist_MA60_pct"] <= -min_disc_dec,
        -merged_last["dist_MA60_pct"],
        0.0,
    )
    eps = 1e-6
    merged_last["risk_norm"] = merged_last["score_total"] / 100.0
    merged_last["discount_risk_ratio"] = np.where(
        merged_last["discount_pct"] > 0,
        merged_last["discount_pct"] / (merged_last["risk_norm"] + eps),
        np.nan,
    )

    base_descuento = (
        (merged_last["trend_label"] == "Alcista") &
        (merged_last["dist_MA60_pct"] <= -min_disc_dec)
    )

    mask_desc_tranquilo_premium = (
        base_descuento &
        (merged_last["score_total"] <= score_p50_adj) &
        (merged_last["VAT3_norm"] <= vat_p50_adj)
    )

    mask_desc_tranquilo_semi = (
        base_descuento &
        (merged_last["score_total"] <= score_p75_adj) &
        (merged_last["VAT3_norm"] <= vat_p75_adj) &
        (~mask_desc_tranquilo_premium)
    )

    mask_desc_violento = (
        base_descuento &
        (merged_last["score_total"] >= 60) &
        (merged_last["VAT3_norm"] >= 2.0)
    )

    mask_sobre_ext = (
        (merged_last["trend_label"] == "Alcista") &
        (merged_last["dist_MA60_pct"] > overext_dec) &
        (merged_last["score_total"] >= 55)
    )

    signals = []

    def _add(mask, label):
        df_sig = merged_last[mask].copy()
        if len(df_sig) == 0:
            return
        df_sig["signal_type"] = label
        signals.append(df_sig)

    _add(mask_desc_tranquilo_premium, "descuento_tranquilo_premium")
    _add(mask_desc_tranquilo_semi, "descuento_tranquilo_semi")
    _add(mask_desc_violento, "descuento_violento_alto_riesgo")
    _add(mask_sobre_ext, "sobre_extension_alcista")

    if not signals:
        return pd.DataFrame()

    signals_all = pd.concat(signals, ignore_index=True)
    # Guardar (por compatibilidad con el pipeline principal)
    outdir = "data/processed"
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "signals_zones_latest.csv")
    cols_out = [
        "date",
        "ticker",
        "trend_label",
        "dist_MA60_pct",
        "score_total",
        "VAT3_norm",
        "discount_pct",
        "discount_risk_ratio",
        "signal_type",
    ]
    cols_out = [c for c in cols_out if c in signals_all.columns]
    signals_all[cols_out].to_csv(out_path, index=False)

    return signals_all


def format_pct(x: float) -> str:
    """Formato porcentaje simple."""
    if pd.isna(x):
        return ""
    return f"{x*100:.2f} %"


def format_ratio(x: float) -> str:
    """Formato ratio simple."""
    if pd.isna(x):
        return ""
    return f"{x:.2f}"


# --------------------------------------------------------------------
# APP PRINCIPAL
# --------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="ACTIVOS-ARGENTINA — Zonas Precio/Riesgo",
        layout="wide",
    )

    if not check_password():
        st.stop()

    st.title("ACTIVOS-ARGENTINA — Mapa de Zonas Precio / Riesgo")
    st.caption("Visualización privada de riesgo, tendencia y señales estadísticas.")

    # Carga de datos
    try:
        trend_df = load_csv_with_date(TREND_PATH, date_col="date")
        risk_df = load_csv_with_date(RISK_PATH, date_col="date")
        vat_df = load_csv_with_date(VAT_PATH, date_col="date")
    except Exception as e:
        st.error(f"Error al cargar archivos: {e}")
        st.stop()

    risk_vat_df = merge_risk_vat(risk_df, vat_df)

    # Fechas comunes
    common_dates = get_common_dates([trend_df, risk_vat_df])
    if not common_dates:
        st.error("No se encontraron fechas comunes entre los datasets.")
        st.stop()

    # Sidebar — selección de fecha, filtro de riesgo y knobs
    with st.sidebar:
        st.header("Parámetros")

        date_selected = st.selectbox(
            "Fecha objetivo",
            options=common_dates,
            index=len(common_dates) - 1,
            format_func=lambda d: d.strftime("%Y-%m-%d"),
        )

        st.subheader("Filtro de riesgo (score_total)")
        risk_filter = st.selectbox(
            "Filtro de bucket:",
            options=[
                "Todos",
                "Muy bajo (0–20)",
                "Moderado (20–40)",
                "Elevado (40–60)",
                "Muy elevado (>=60)",
            ],
        )

        st.markdown("---")
        st.subheader("Ajustes finos (no rompen el modelo)")

        min_discount_pct = st.slider(
            "Mínimo descuento vs MA60 (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="Requiere que el precio esté al menos este % por debajo de MA60 para considerarse 'en descuento'.",
        )

        overext_threshold_pct = st.slider(
            "Umbral de sobre-extensión vs MA60 (%)",
            min_value=8.0,
            max_value=20.0,
            value=12.0,
            step=0.5,
            help="Precio por encima de MA60 a partir del cual se considera 'sobre-extensión alcista'.",
        )

        relax_score = st.slider(
            "Relajar umbrales de riesgo (score_total, puntos)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=1.0,
            help="Permite incluir activos con score_total algo mayor que los percentiles P50/P75 base.",
        )

        relax_vat = st.slider(
            "Relajar umbrales de colas (VAT3_norm)",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.05,
            help="Permite incluir activos con VAT3_norm algo mayor que los percentiles P50/P75 base.",
        )

    target_date = pd.to_datetime(date_selected)

    # Resumen general
    st.markdown("### 1. Resumen general")

    col1, col2, col3 = st.columns(3)

    df_trend_last = trend_df[trend_df["date"] == target_date].copy()
    df_risk_last = risk_vat_df[
        (risk_vat_df["date"] == target_date) & (risk_vat_df["status"] == "OK")
    ].copy()

    with col1:
        st.metric("Activos con tendencia calculada", df_trend_last["ticker"].nunique())
        st.metric("Activos con riesgo OK", df_risk_last["ticker"].nunique())

    with col2:
        if len(df_risk_last) > 0:
            score_mediana = df_risk_last["score_total"].median()
            st.metric("Mediana score_total", f"{score_mediana:.1f}")
        if len(df_risk_last) > 0:
            vat_mediana = df_risk_last["VAT3_norm"].median()
            st.metric("Mediana VAT3_norm", f"{vat_mediana:.2f}")

    with col3:
        trend_counts = df_trend_last["trend_label"].value_counts()
        st.write("Tendencias en la fecha:")
        st.write(trend_counts)

    # Construcción de señales con knobs
    signals = build_latest_signals(
        trend_df,
        risk_vat_df,
        target_date,
        min_discount_pct=min_discount_pct,
        overext_threshold_pct=overext_threshold_pct,
        relax_score=relax_score,
        relax_vat=relax_vat,
    )

    st.markdown("### 2. Señales / Zonas a la fecha seleccionada")

    if signals.empty:
        st.info("No se detectaron señales para esta fecha con los umbrales actuales.")
    else:
        # Filtro de riesgo
        if risk_filter != "Todos":
            if "Muy bajo" in risk_filter:
                signals = signals[signals["score_total"] < 20]
            elif "Moderado" in risk_filter:
                signals = signals[(signals["score_total"] >= 20) & (signals["score_total"] < 40)]
            elif "Elevado" in risk_filter:
                signals = signals[(signals["score_total"] >= 40) & (signals["score_total"] < 60)]
            elif "Muy elevado" in risk_filter:
                signals = signals[signals["score_total"] >= 60]

        tabs = st.tabs(
            [
                "Descuento tranquilo PREMIUM",
                "Descuento tranquilo SEMI",
                "Descuento violento (alto riesgo)",
                "Sobre-extensión alcista",
                "Todas las señales",
            ]
        )

        def _prepare_df(df: pd.DataFrame) -> pd.DataFrame:
            """Preparar tabla amigable."""
            if df.empty:
                return df
            df_show = df.copy()

            # Ordenar por ratio en descuentos; por dist_MA60 en sobre-extensión
            if (df_show["signal_type"] == "sobre_extension_alcista").all():
                df_show = df_show.sort_values("dist_MA60_pct", ascending=False)
            elif (df_show["signal_type"] == "descuento_violento_alto_riesgo").all():
                df_show = df_show.sort_values("dist_MA60_pct", ascending=True)
            else:
                if "discount_risk_ratio" in df_show.columns:
                    df_show = df_show.sort_values("discount_risk_ratio", ascending=False)

            cols = [
                "ticker",
                "trend_label",
                "dist_MA60_pct",
                "score_total",
                "VAT3_norm",
                "discount_pct",
                "discount_risk_ratio",
                "signal_type",
            ]
            cols = [c for c in cols if c in df_show.columns]
            df_show = df_show[cols]

            # Formatear porcentaje y ratio
            if "dist_MA60_pct" in df_show.columns:
                df_show["dist_MA60_pct"] = df_show["dist_MA60_pct"].apply(format_pct)
            if "discount_pct" in df_show.columns:
                df_show["discount_pct"] = df_show["discount_pct"].apply(format_pct)
            if "discount_risk_ratio" in df_show.columns:
                df_show["discount_risk_ratio"] = df_show["discount_risk_ratio"].apply(format_ratio)

            return df_show

        with tabs[0]:
            df_tranq_p = signals[signals["signal_type"] == "descuento_tranquilo_premium"]
            st.subheader("Descuento tranquilo PREMIUM")
            st.caption(
                "Alcista + por debajo de MA60 (>= mínimo descuento configurado) + "
                "score_total <= P50 ajustado + VAT3_norm <= P50 ajustado (adaptativo por fecha)."
            )
            if df_tranq_p.empty:
                st.info("No hay activos en esta categoría para la fecha seleccionada.")
            else:
                st.dataframe(_prepare_df(df_tranq_p), width="stretch")

        with tabs[1]:
            df_tranq_s = signals[signals["signal_type"] == "descuento_tranquilo_semi"]
            st.subheader("Descuento tranquilo SEMI")
            st.caption(
                "Alcista + por debajo de MA60 (>= mínimo descuento configurado) + "
                "score_total <= P75 ajustado + VAT3_norm <= P75 ajustado, "
                "excluyendo la categoría PREMIUM (adaptativo por fecha)."
            )
            if df_tranq_s.empty:
                st.info("No hay activos en esta categoría para la fecha seleccionada.")
            else:
                st.dataframe(_prepare_df(df_tranq_s), width="stretch")

        with tabs[2]:
            df_viol = signals[signals["signal_type"] == "descuento_violento_alto_riesgo"]
            st.subheader("Descuento violento con riesgo elevado")
            st.caption(
                "Alcista + por debajo de MA60 (>= mínimo descuento configurado) + "
                "score_total ≥ 60 + VAT3_norm ≥ 2.0."
            )
            if df_viol.empty:
                st.info("No hay activos en esta categoría para la fecha seleccionada.")
            else:
                st.dataframe(_prepare_df(df_viol), width="stretch")

        with tabs[3]:
            df_ext = signals[signals["signal_type"] == "sobre_extension_alcista"]
            st.subheader("Sobre-extensión alcista")
            st.caption(
                "Alcista + muy por encima de MA60 (umbral configurable) + score_total ≥ 55."
            )
            if df_ext.empty:
                st.info("No hay activos en esta categoría para la fecha seleccionada.")
            else:
                st.dataframe(_prepare_df(df_ext), width="stretch")

        with tabs[4]:
            st.subheader("Todas las señales combinadas")
            st.dataframe(_prepare_df(signals), width="stretch")

    st.markdown("### 3. Vista de depuración (opcional)")

    with st.expander("Ver datos crudos combinados (riesgo + tendencia)", expanded=False):
        merged_last = pd.merge(
            df_trend_last,
            df_risk_last[["date", "ticker", "score_total", "VAT3_norm", "status"]],
            on=["date", "ticker"],
            how="left",
        )
        st.dataframe(merged_last.head(100), width="stretch")


if __name__ == "__main__":
    main()
