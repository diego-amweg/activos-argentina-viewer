# ACTIVOS-ARGENTINA — Viewer de zonas precio/riesgo

Repositorio público con la app Streamlit para visualizar:

- Señales de descuento / sobre-extensión.
- Métrica `discount_risk_ratio` (descuento / riesgo).
- Resumen de tendencia y riesgo (si están disponibles los CSV completos).

Este repo **no** contiene la lógica interna de cálculo de métricas;
solo lee los CSV ya procesados desde `data/processed`.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run scripts/alpha_zones_app.py
