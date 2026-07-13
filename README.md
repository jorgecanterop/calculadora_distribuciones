# Calculadora interactiva de distribuciones — Streamlit

Conversión de la notebook `Distribuciones_Especiales_eventos_y_probabilidades` a una aplicación Streamlit autónoma, preparada para utilizarse directamente o incrustarse en Google Sites.

## Funcionalidades conservadas

- 13 distribuciones discretas y continuas.
- Cálculo de probabilidades para eventos de cola, igualdad e intervalo.
- Cálculo inverso de límites a partir de una probabilidad objetivo.
- Tratamiento correcto de límites no enteros en distribuciones discretas.
- Gráfico de la función de densidad y de la función de distribución.
- El resultado solicitado se destaca primero: probabilidad calculada o cuantil/límite obtenido.
- Media y varianza como información de la distribución.
- Validaciones de parámetros y mensajes de error legibles.

## Adaptaciones para Google Sites

- La configuración general ya no depende de la barra lateral de Streamlit.
- Distribución y modo de cálculo se encuentran en un cuadro expandible dentro de la página principal.
- El formulario y los resultados utilizan todo el ancho disponible.
- La salida prioriza el valor solicitado; la media y la varianza se presentan después como información secundaria.
- Los gráficos se muestran uno encima del otro para mejorar su tamaño y legibilidad.
- Los parámetros numéricos continuos se presentan con dos decimales por defecto.
- Las listas desplegables solo permiten seleccionar opciones existentes; la escritura y el filtrado por teclado están desactivados.
- Al cambiar el tipo de evento, los campos de límites o probabilidad se actualizan inmediatamente sin tener que calcular primero.
- La interfaz incluye ajustes responsivos básicos para marcos estrechos.

## Estructura

- `app.py`: interfaz Streamlit.
- `distribution_engine.py`: lógica estadística independiente de la interfaz.
- `plotting.py`: generación de gráficos Matplotlib.
- `.streamlit/config.toml`: apariencia fija y configuración de ejecución.
- `requirements.txt`: dependencias; requiere Streamlit 1.58 o posterior para desactivar la entrada por teclado en las listas.
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
