from __future__ import annotations

from pathlib import Path
from typing import Optional

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


def render_help() -> None:
    """Guía rápida de interpretación del viewer (se muestra en un expander)."""
    with st.expander("¿Cómo usar este viewer? Guía rápida", expanded=False):
        st.markdown(
            """
### 1. Idea general

Este viewer muestra **señales estadísticas pre-calculadas** para el universo
de activos de *ACTIVOS-ARGENTINA*.  
El motor de cálculo vive en otro proyecto y **no está expuesto aquí**.

Pensalo como un mapa de “zonas de precio / riesgo” que te ayuda a:

- Detectar **sobre-extensiones alcistas** (posibles candidatos a toma de ganancias
  o monitoreo cercano).
- Ver **descuentos tranquilos** (escenarios de posible entrada, siempre
  sujetos a validación fundamental).
- Filtrar por **riesgo total** y por **condiciones mínimas** de descuento.

> Nada de lo que ves aquí constituye una recomendación de inversión.

---

### 2. Barra lateral (filtros principales)

1. **Fecha objetivo**  
   - Elegís la fecha para la que querés ver las señales.
   - Solo se muestran fechas donde el motor generó señales.

2. **Filtro de bucket de riesgo (score_total)**  
   - *Todos*: muestra todo el universo con señal.
   - *Bajo / Medio / Alto / Muy alto*: restringe las señales según el
     bucket de `score_total`.  
       - *Bajo*: escenarios más tranquilos.
       - *Muy alto*: escenarios más especulativos / volátiles.

3. **Mínimo descuento (%)**  
   - Cuando en el futuro exista una métrica de descuento explícita,
     este slider filtrará solo activos con **descuento mínimo** respecto
     a su referencia.
   - Mientras tanto, podés dejarlo en `0.00%` para no filtrar nada
     adicional.

4. **Umbral de sobre-extensión vs MA60 (%)**  
   - Define qué tan por encima de la media móvil de 60 ruedas (MA60)
     debe estar el precio para que consideremos una **sobre-extensión
     alcista**.
   - Valores sugeridos:
     - `12%` (default): sobre-extensión moderada.
     - `15–20%`: sobre-extensión extrema (bordes).

---

### 3. Bloque “1. Resumen general”

Te da una foto rápida del universo filtrado:

- **Activos distintos con señal**: cuántos tickers tienen al menos una señal.
- **Cantidad total de señales**: algunas reglas pueden generar
  más de una señal por ticker.
- **Mediana de score_total y VAT3_norm**: te ayudan a entender si la
  fecha está “tranquila” o “cargada” de riesgo.
- **Tabla de tendencias**: cuántos activos están etiquetados como
  *Alcista* (u otras tendencias, si se agregan a futuro).

Si este bloque muestra un mensaje azul informando que no hay señales,
revisá los filtros: tal vez elegiste una fecha sin señales o
pusiste filtros demasiado restrictivos.

---

### 4. Bloque “2. Señales / Zonas a la fecha seleccionada”

Hay 5 pestañas:

1. **Descuento tranquilo PREMIUM**  
   - Escenarios de *descuento moderado/alto* + *buen perfil de riesgo*.
   - Son candidatos naturales para mirar con calma y luego validar
     con análisis fundamental.

2. **Descuento tranquilo SEMI**  
   - Parecido al anterior, pero con condiciones algo más laxas.
   - Útil para ampliar el radar cuando el universo está muy filtrado.

3. **Descuento violento (alto riesgo)**  
   - Descuentos fuertes en activos con **riesgo muy elevado**.
   - Territorio especulativo. Se mira, pero no implica acción.

4. **Sobre-extensión alcista**  
   - Activos *alcistas* y **muy por encima de la MA60** (según el
     umbral que definas en el slider).
   - Pueden ser candidatos a:
     - Toma parcial de ganancias.
     - Ajuste de stop.
     - Monitoreo más cercano.

5. **Todas las señales**  
   - Muestra el conjunto completo para la fecha, ordenado por
     `score_total` (de mayor a menor).
   - Útil como “tablero completo” después de explorar las pestañas
     anteriores.

En todas las pestañas:

- Las tablas muestran:
  - `ticker`
  - `trend_label` (por ahora “Alcista”)
  - `signal_type`
  - `score_total`
  - `VAT3_norm`
  - `dist_MA60_pct` (distancia a MA60 en %)
  - `discount_pct` (por ahora placeholder)
  - `discount_risk_ratio` (placeholder para análisis futuro)

Si una pestaña no tiene resultados, se muestra un mensaje azul aclarando
que no hay activos para esa categoría con los filtros actuales.

---

### 5. Cómo usar los “bordes” de señal en la práctica

- **Para buscar entradas posibles**  
  - Mirar primero *Descuento tranquilo PREMIUM* y *SEMI* con:
    - Bucket de riesgo en `Bajo` o `Medio`.
    - Umbral de MA60 en su valor por defecto.
  - Luego, validar esos tickers con tu análisis fundamental.

- **Para controlar toma de ganancias / sobre-extensión**  
  - Ir a *Sobre-extensión alcista*.
  - Subir el umbral de MA60 al rango `15–20%` para ver los casos más
    extremos.
  - Revisar esos nombres dentro de tu cartera o watchlist.

Siempre el orden es:
1. Señal cuantitativa → 2. Validación cualitativa → 3. Decisión.

---

### 6. Recordatorio legal

Este viewer:

- No reemplaza el análisis fundamental.
- No considera tu perfil de riesgo ni tu situación patrimonial.
- No constituye recomendación de inversión ni asesoramiento financiero.

Tomá todas las salidas como **insumos de análisis**, no como órdenes
de compra/venta.
            """
        )


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

    # Guía de uso dentro de la app
    render_help()

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

        # 🔄 Botón para recargar datos (borrar caché)
        if st.button("🔄 Recargar datos (borrar caché)"):
            st.cache_data.clear()
            st.experimental_rerun()

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
