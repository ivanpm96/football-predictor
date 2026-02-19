# Football Predictor Pro — Guia de Implementacion

## Estructura del proyecto
```
football_predictor/
├── app.py                  # UI Streamlit principal
├── predictor.py            # Motor Poisson + GBM
├── catalog_builder.py      # Catalogo dinamico de ligas
├── auto_updater.py         # Actualizacion automatica en background
├── requirements.txt        # Dependencias
└── fd_cache/               # Cache JSON (se crea automaticamente)
    ├── competitions.json   # Catalogo de ligas
    ├── bundesliga-2025.json
    ├── bundesliga-2024.json
    └── ...
```

## Tipos de temporada
| Tipo       | Ejemplo         | slug-2025 significa | Label |
|------------|-----------------|---------------------|-------|
| SPLIT      | Bundesliga, EPL | 2025/26             | 2025/26 |
| SINGLE     | MLS             | 2025                | 2025  |
| TOURNAMENT | Copa America    | ano del evento      | 2025  |

## Paso 1 — Instalar dependencias
```bash
pip install -r requirements.txt
```

## Paso 2 — Ejecutar la app
```bash
streamlit run app.py
```

## Paso 3 — Primera vez (catalogo)
Al iniciar, la app carga el catalogo fallback automaticamente.
Para construir el catalogo completo verificado:
```bash
python catalog_builder.py --force
```
Esto toma ~5-10 min (HEAD requests a la API).

## Paso 4 — Usar la app
1. Seleccionar Region -> Competicion en el sidebar
2. Las temporadas de entrenamiento y prediccion se sugieren automaticamente
   - Bundesliga muestra: entrena 2018/19...2024/25 | predice 2025/26
   - MLS muestra:        entrena 2019...2025       | predice 2026
3. Click en 'Cargar & Entrenar'
4. Ir a Tab 'Proximos partidos' para ver todos los pronosticos

## Despliegue en Streamlit Cloud
1. Subir los 4 archivos .py + requirements.txt a GitHub
2. Conectar repo en share.streamlit.io
3. Main file: app.py
4. NO subir fd_cache/ (se crea en runtime)

## Despliegue local con Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py .
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
```
```bash
docker build -t football-predictor .
docker run -p 8501:8501 -v $(pwd)/fd_cache:/app/fd_cache football-predictor
```
