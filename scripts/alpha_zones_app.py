# scripts/alpha_zones_app.py
import os
from pathlib import Path
import pandas as pd
import streamlit as st


# =========================
# Config
# =========================
APP_TITLE = "Activos Argentina — Zonas + Ranking (Viewer)"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

SIGNALS_CSV = DATA_DIR / "signals_zones_latest.csv"
RANKING_CSV = DATA_DIR / "multifactor_ranking_v0.csv"

DATE_COL = "date"

# Nota: el ranking NO es recomendación. Es prioridad para mirar tickers.
DISCLAIMER_RANKING = (
    "⚠️ **IMPORTANTE**: Este *ranking* **NO es una recomendación de inversión** ni una señal automática. "
    "Es un **orden de prioridad para revisar tickers** con análisis adicional (cualitativo + contexto + liquidez real)."
)

DISCLAIMER_GENERAL = (
    "Este viewer es una herramienta de *análisis* (CFA-friendly + extensiones propias del sistema). "
    "No constituye asesoramiento financiero."
)


# =========================
# Helpers
# =========================
def _file_mtime(path: Path) -> float:
    if not path.exists():
        return 0.0
    return float(path.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    # mtime se usa SOLO para invalidar cache cuando el archivo cambia
    return pd.read_csv(path_str)


def load_csv(path: Path) -> pd.DataFrame:
    return load_csv_cached(str(path), _file_mtime(path))


def normalize_date_col(df: pd.DataFrame) -> pd.DataFrame:
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.date
    return df


def safe_read_signals() -> pd.DataFrame:
    if not SIGNALS_CSV.exists():
        return pd.DataFrame()
    df = load_csv(SIGNALS_CSV)
    df = normalize_date_col(df)
    return df


def safe_read_ranking() -> pd.DataFrame:
    if not RANKING_CSV.exists():
        return pd.DataFrame()
    df = load_csv(RANKING_CSV)
    df = normalize_date_col(df)
    return df


def render_zones(df: pd.DataFrame) -> None:
    st.subheader("Zonas (signals_zones_latest.csv)")

    if df.empty:
        st.warning(f"No se encontró o está vacío: {SIGNALS_CSV}")
        st.stop()

    # Filtros básicos
    dates = sorted([d for d in df[DATE_COL].dropna().unique().tolist()])
    default_date = dates[-1] if dates else None

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        selected_date = st.selectbox("Fecha", options=dates, index=len(dates) - 1 if dates else 0)
    with col2:
        zone = st.selectbox(
            "Zona",
            options=["(todas)"] + sorted([z for z in df.get("zone_label", pd.Series(dtype=str)).dropna().unique().tolist()]),
            index=0,
        )
    with col3:
        ticker_q = st.text_input("Buscar ticker (contiene)", value="").strip().upper()

    dff = df.copy()
    if selected_date:
        dff = dff[dff[DATE_COL] == selected_date]

    if zone != "(todas)" and "zone_label" in dff.columns:
        dff = dff[dff["zone_label"] == zone]

    if ticker_q and "ticker" in dff.columns:
        dff = dff[dff["ticker"].astype(str).str.upper().str.contains(ticker_q, na=False)]

    # Orden recomendado: por label y luego por dist/riesgo si existen
    sort_cols = []
    if "discount_pct" in dff.columns:
        sort_cols.append("discount_pct")
    if "score_total" in dff.columns:
        sort_cols.append("score_total")
    if "VAT3_norm" in dff.columns:
        sort_cols.append("VAT3_norm")

    if sort_cols:
        dff = dff.sort_values(by=sort_cols, ascending=False)

    st.caption(f"Filas: {len(dff)}")
    st.dataframe(dff, use_container_width=True)


def render_ranking(df: pd.DataFrame) -> None:
    st.subheader("Ranking v0 (multifactor_ranking_v0.csv)")
    st.info(DISCLAIMER_RANKING)

    if df.empty:
        st.warning(f"No se encontró o está vacío: {RANKING_CSV}")
        st.stop()

    # Fecha (por defecto última)
    dates = sorted([d for d in df[DATE_COL].dropna().unique().tolist()])
    default_date = dates[-1] if dates else None

    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
    with col1:
        selected_date = st.selectbox("Fecha", options=dates, index=len(dates) - 1 if dates else 0)
    with col2:
        top_n = st.selectbox("Top N", options=[20, 50, 100, 200, 500], index=1)
    with col3:
        only_ok = st.checkbox("Solo status=OK (si existe)", value=True)
    with col4:
        ticker_q = st.text_input("Buscar ticker (contiene)", value="").strip().upper()

    dff = df.copy()
    if selected_date:
        dff = dff[dff[DATE_COL] == selected_date]

    if only_ok and "status" in dff.columns:
        dff = dff[dff["status"].astype(str).str.upper() == "OK"]

    if ticker_q and "ticker" in dff.columns:
        dff = dff[dff["ticker"].astype(str).str.upper().str.contains(ticker_q, na=False)]

    # Orden por score_total_v0 desc si existe; si no, rank_v0 asc
    if "score_total_v0" in dff.columns:
        dff = dff.sort_values(by="score_total_v0", ascending=False)
    elif "rank_v0" in dff.columns:
        dff = dff.sort_values(by="rank_v0", ascending=True)

    dff_head = dff.head(int(top_n)) if top_n else dff
    st.caption(f"Filas (post-filtros): {len(dff)} | Mostrando: {len(dff_head)}")

    # Columnas “de cabecera” sugeridas si existen
    preferred = [
        "date",
        "ticker",
        "rank_v0",
        "score_total_v0",
        "score_salida_v0",
        "trend_label",
        "risk_score_100",
        "tails_score_100",
        "liquidity_proxy_score_100",
        "price_position_score_100",
        "pos_in_range_60_score_100",
        "close_target",
        "last_same_level_date",
        "days_since_same_level",
        "zone_label",
    ]
    cols = [c for c in preferred if c in dff_head.columns]
    # si faltan, mostramos todo
    if cols:
        st.dataframe(dff_head[cols], use_container_width=True)
        with st.expander("Ver todas las columnas"):
            st.dataframe(dff_head, use_container_width=True)
    else:
        st.dataframe(dff_head, use_container_width=True)


# =========================
# App
# =========================
st.set_page_config(page_title=APP_TITLE, layout="wide")

st.title(APP_TITLE)
st.caption(DISCLAIMER_GENERAL)

# Botón para limpiar cache
colA, colB = st.columns([1, 5])
with colA:
    if st.button("🔄 Recargar datos (borrar caché)"):
        st.cache_data.clear()
        st.success("Caché borrado. Releyendo datos...")

# Lectura datos (con invalidación automática por mtime)
signals_df = safe_read_signals()
ranking_df = safe_read_ranking()

tab1, tab2 = st.tabs(["Zonas", "Ranking v0"])

with tab1:
    render_zones(signals_df)

with tab2:
    render_ranking(ranking_df)
