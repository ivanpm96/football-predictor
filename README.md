# ⚽ Football Predictor — FixtureDownload.com

Sistema autónomo de predicción de fútbol usando Poisson + Gradient Boosting.

## Archivos
| Archivo | Descripción |
|---|---|
| `app.py` | Dashboard Streamlit (interfaz visual) |
| `predictor.py` | Motor de predicción (Poisson + ML) |
| `requirements.txt` | Dependencias Python |
| `colab_launcher.ipynb` | Notebook para Google Colab |

## Opción 1 — Streamlit Community Cloud (URL permanente, GRATIS)
1. Sube estos archivos a un repositorio GitHub
2. Ve a https://share.streamlit.io/
3. Conecta tu GitHub y selecciona `app.py`
4. ¡Listo! URL pública gratuita

## Opción 2 — Google Colab (rápido, URL temporal)
Abre `colab_launcher.ipynb` en Google Colab y ejecuta todas las celdas.

## Opción 3 — Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Uso del dashboard
1. **Sidebar**: selecciona competición + temporadas → clic en **Cargar & Entrenar**
2. **Tab Predicción**: selecciona local y visitante → **Predecir**
3. **Tab Próximos partidos**: predice todos los fixtures pendientes de una vez
4. **Tab Análisis**: historial visual de cualquier equipo

## Competiciones disponibles
| Label | IDs FixtureDownload |
|---|---|
| Premier League | `epl-2023`, `epl-2024`, `epl-2025` |
| La Liga | `la-liga-2023`, `la-liga-2024`, `la-liga-2025` |
| Bundesliga | `bundesliga-2023`, `bundesliga-2024`, `bundesliga-2025` |
| Serie A | `serie-a-2023`, `serie-a-2024` |
| Ligue 1 | `ligue-1-2023`, `ligue-1-2024` |
| Champions League | `uefa-champions-league-2024` |
