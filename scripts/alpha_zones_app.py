from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Ruta al CSV dentro del repo del viewer
DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "signals_zones_latest.csv"
)


# ---------------------------------------------------------------------------
# Carga y preparación de datos
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Carga el CSV de señales y agrega columnas auxiliares para el viewer."""
    df = pd.read_csv(path)

    # Tipos
    df["date"] = pd.to_datetime(df["date"])

    # dist_MA60_pct viene como proporción (ej: 0.34 = 34%)
    # Lo pasamos a porcentaje directo para trabajar más intuitivamente
    df["dist_MA60_pct"] = df["dist_MA60_pct"] * 100.0

    # Por ahora no tenemos verdaderas señales de "descuento".
    # Definimos discount_pct = 0 para mantener la estructura del viewer.
    df["discount_pct"] = np.where(df["dist_MA60_pct"] < 0, -df["dist_MA60_pct"], 0.0)

    # Ratio descuento / riesgo (placeholder, útil a futuro)
    df["discount_risk_ratio"] = np.where(
        df["discount_pct"] > 0,
        df["discount_pct"] / df["score_total"].replace(0, np.nan),
        np.nan,
    )

    # Buckets de riesgo aproximados, solo para filtro de alto nivel
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

    df["score_bucket"] = df["score_total"].apply(bucket)

    return df


def base_filter(
    df: pd.DataFrame,
    target_date,
    bucket_filter: str,
    min_discount: float,
) -> pd.DataFrame:
    """Aplica filtros comunes: fecha, bucket de riesgo y mínimo descuento."""
    out = df[df["date"].dt.date == target_date].copy()

    if bucket_filter != "Todos":
        out = out[out["score_bucket"] == bucket_filter]

    if min_discount > 0:
        out = out[out["discount_pct"] >= min_discount]

    return out


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ordena y formatea columnas para mostrarlas en la tabla principal."""
    if df.empty:
        return df

    df_disp = df.copy()

    # Formateo de porcentajes
    df_disp["dist_MA60_pct"] = df_disp["dist_MA60_pct"].map(
        lambda x: f"{x:.2f} %"
    )
    df_disp["discount_pct"] = df_disp["discount_pct"].map(
        lambda x: f"{x:.2f} %"
    )

    # Orden sugerido de columnas
    cols = [
        "ticker",
        "trend_label",
        "signal_type",
        "score_total",
        "VAT3_norm",
        "dist_MA60_pct",
        "discount_pct",
        "date",
        "discount_risk_ratio",
        "score_bucket",
    ]

    # Nos quedamos solo con las que existan (por si el CSV cambia en el futuro)
    cols = [c for c in cols if c in df_disp.columns]

    df_disp = df_disp[cols]

    return df_disp


# ---------------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="ACTIVOS-ARGENTINA — Viewer de Zonas Precio / Riesgo",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ACTIVOS-ARGENTINA — Viewer de Zonas Precio / Riesgo")
    st.caption(
        "Visualización privada de señales estadísticas pre-calculadas. "
        "El motor de cálculo permanece en el repositorio principal (no visible aquí)."
    )

    # Carga de datos
    try:
        df = load_data(str(DATA_FILE))
    except FileNotFoundError:
        st.error(f"Error al cargar archivos: No se encontró el archivo: {DATA_FILE}")
        st.stop()
    except Exception as e:
        st.error(f"Error al cargar archivos: {e}")
        st.stop()

    # ------------------------------------------------------------------
    # Sidebar: parámetros y filtros
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Parámetros")

        # Fechas disponibles (ordenadas de más reciente a más antigua)
        available_dates = sorted(df["date"].dt.date.unique(), reverse=True)
        target_date = st.selectbox(
            "Fecha objetivo",
            options=available_dates,
            format_func=lambda d: d.isoformat(),
        )

        st.subheader("Filtro de riesgo\n(score_total)")
        bucket_options = ["Todos", "Bajo", "Medio", "Alto", "Muy alto"]
        bucket_filter = st.selectbox("Filtro de bucket:", options=bucket_options)

        st.subheader("Filtros adicionales")

        min_discount = st.slider(
            "Mínimo descuento (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.25,
            help=(
                "Filtra solo activos con un descuento mínimo respecto a su nivel "
                "de referencia (cuando esa métrica esté disponible)."
            ),
        )

        ma60_threshold = st.slider(
            "Umbral de sobre-extensión vs MA60 (%)",
            min_value=8.0,
            max_value=20.0,
            value=12.0,
            step=0.25,
            help=(
                "Define qué tan por encima de la MA60 debe estar el precio para "
                "que una señal se considere 'sobre-extensión alcista'."
            ),
        )

    # Aplicamos filtros base (fecha, bucket, descuento)
    df_filtered = base_filter(df, target_date, bucket_filter, min_discount)

    # ------------------------------------------------------------------
    # 1. Resumen general
    # ------------------------------------------------------------------
    st.markdown("### 1. Resumen general")

    if df_filtered.empty:
        st.info(
            "No hay señales para los filtros actuales. Ajustá la fecha, "
            "el bucket de riesgo o el mínimo de descuento para ver resultados."
        )
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Activos distintos con señal en la fecha",
                int(df_filtered["ticker"].nunique()),
            )
            st.metric(
                "Cantidad total de señales en la fecha",
                int(len(df_filtered)),
            )

        with col2:
            st.metric(
                "Mediana score_total (sobre universo con señal)",
                round(float(df_filtered["score_total"].median()), 1),
            )
            st.metric(
                "Mediana VAT3_norm (sobre universo con señal)",
                round(float(df_filtered["VAT3_norm"].median()), 2),
            )

        with col3:
            st.write("Tendencias entre activos con señal:")
            trend_counts = (
                df_filtered.groupby("trend_label")["ticker"]
                .nunique()
                .reset_index(name="count")
            )
            st.dataframe(trend_counts, use_container_width=True)

    # ------------------------------------------------------------------
    # 2. Señales / Zonas a la fecha seleccionada
    # ------------------------------------------------------------------
    st.markdown("### 2. Señales / Zonas a la fecha seleccionada")

    (
        tab_desc_premium,
        tab_desc_semi,
        tab_desc_violento,
        tab_sobre_ext,
        tab_all,
    ) = st.tabs(
        [
            "Descuento tranquilo PREMIUM",
            "Descuento tranquilo SEMI",
            "Descuento violento (alto riesgo)",
            "Sobre-extensión alcista",
            "Todas las señales",
        ]
    )

    # Helper para renderizar cada tab
    def render_tab(tab, title: str, description: str, df_tab: pd.DataFrame) -> None:
        with tab:
            st.subheader(title)
            st.write(description)
            if df_tab.empty:
                st.info("No hay activos en esta categoría con los filtros actuales.")
            else:
                st.dataframe(prepare_df(df_tab), use_container_width=True)

    # Mapeo de tipos de señal (a futuro el motor puede generar más tipos)
    df_premium = df_filtered[df_filtered["signal_type"] == "descuento_tranquilo_premium"]
    df_semi = df_filtered[df_filtered["signal_type"] == "descuento_tranquilo_semi"]
    df_violento = df_filtered[
        df_filtered["signal_type"] == "descuento_violento_alto_riesgo"
    ]

    df_sobre_ext = df_filtered[df_filtered["signal_type"] == "sobre_extension_alcista"]
    df_sobre_ext = df_sobre_ext[df_sobre_ext["dist_MA60_pct"] >= ma60_threshold]

    df_all = df_filtered.copy()

    # Render de cada tab
    render_tab(
        tab_desc_premium,
        "Descuento tranquilo PREMIUM",
        (
            "Alcista + por debajo de MA60 (descuento moderado/alto) + score_total "
            "elevado (bajo riesgo relativo) + colas suaves. Ordenadas por mejor "
            "relación descuento / riesgo."
        ),
        df_premium,
    )

    render_tab(
        tab_desc_semi,
        "Descuento tranquilo SEMI",
        (
            "Señales etiquetadas como descuento tranquilo SEMI en el motor principal. "
            "También ordenadas por mejor relación descuento / riesgo, pero con "
            "condiciones algo más laxas."
        ),
        df_semi,
    )

    render_tab(
        tab_desc_violento,
        "Descuento violento con riesgo elevado",
        (
            "Alcista + fuerte descuento vs MA60 + riesgo muy alto. Escenario "
            "especulativo y de alta volatilidad."
        ),
        df_violento,
    )

    render_tab(
        tab_sobre_ext,
        "Sobre-extensión alcista",
        (
            "Alcista + muy por encima de MA60 (umbral configurable) + riesgo "
            "medio/alto. Posibles candidatos a toma de ganancias o monitoreo cercano."
        ),
        df_sobre_ext.sort_values("dist_MA60_pct", ascending=False),
    )

    render_tab(
        tab_all,
        "Todas las señales combinadas",
        (
            "Todas las señales de la fecha seleccionada, aplicando los filtros de "
            "riesgo y descuento configurados en la barra lateral."
        ),
        df_all.sort_values("score_total", ascending=False),
    )

    st.markdown(
        "---\n"
        "Notas: los parámetros y percentiles de riesgo se calculan sobre el universo "
        "de señales disponibles en el archivo actual. Este viewer no reemplaza el "
        "análisis fundamental ni constituye recomendación de inversión."
    )


if __name__ == "__main__":
    main()
