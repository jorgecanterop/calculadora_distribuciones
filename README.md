# Calculadora interactiva de distribuciones — Streamlit

Conversión de la notebook `Distribuciones_Especiales_eventos_y_probabilidades` a una aplicación Streamlit autónoma.

## Funcionalidades conservadas

- 13 distribuciones discretas y continuas.
- Cálculo de probabilidades para eventos de cola, igualdad e intervalo.
- Cálculo inverso de límites a partir de una probabilidad objetivo.
- Tratamiento correcto de límites no enteros en distribuciones discretas.
- Gráfico de la función de masa/densidad y de la función de distribución.
- Media, varianza y desviación estándar.
- Validaciones de parámetros y mensajes de error legibles.

## Estructura

- `app.py`: interfaz Streamlit.
- `distribution_engine.py`: lógica estadística independiente de la interfaz.
- `plotting.py`: generación de gráficos Matplotlib.
- `.streamlit/config.toml`: apariencia fija y configuración de ejecución.
- `requirements.txt`: dependencias.
- `run_app.bat`: inicio rápido en Windows.
- `tests/test_engine.py`: pruebas básicas del motor estadístico.

## Ejecución

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux/macOS:

```bash
source .venv/bin/activate
```

Luego:

```bash
pip install -r requirements.txt
streamlit run app.py
```

En Windows también puede iniciarse con doble clic en `run_app.bat` después de instalar las dependencias.
