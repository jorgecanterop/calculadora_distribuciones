"""Aplicación Streamlit: calculadora interactiva de distribuciones."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from distribution_engine import (
    CALCULATION_MODES,
    DIRECT_EVENTS,
    DISTRIBUTIONS,
    INVERSE_EVENTS,
    calculate_event_from_probability,
    calculate_probability,
    distribution_moments,
    event_to_latex,
    format_moment,
    normalize_parameters,
)
from plotting import create_distribution_figure

st.set_page_config(
    page_title="Calculadora de distribuciones",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #F7F9FC; }
      [data-testid="stSidebar"] { background: #EEF3F8; }
      .block-container { max-width: 1480px; padding-top: 1.5rem; padding-bottom: 2.5rem; }
      .hero {
          background: white; border: 1px solid #D9E2EC; border-radius: 16px;
          padding: 1.15rem 1.35rem; margin-bottom: 1rem;
          box-shadow: 0 4px 18px rgba(31, 41, 55, 0.05);
      }
      .hero h1 { margin: 0 0 .25rem 0; color: #1F2937; font-size: 2rem; }
      .hero p { margin: 0; color: #5D6878; line-height: 1.55; }
      .distribution-card {
          background: white; border: 1px solid #D9E2EC; border-radius: 14px;
          padding: .9rem 1rem; margin: .4rem 0 1rem 0;
      }
      .type-discrete { color: #3D8BB8; font-weight: 700; }
      .type-continuous { color: #9B5E86; font-weight: 700; }
      [data-testid="stMetric"] {
          background: white; border: 1px solid #D9E2EC; border-radius: 12px;
          padding: .55rem .75rem;
      }
      div[data-testid="stAlert"] { border-radius: 12px; }
      .stButton > button, .stFormSubmitButton > button {
          border-radius: 10px; min-height: 2.75rem; font-weight: 700;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def clear_result() -> None:
    st.session_state.pop("calculation_result", None)


st.markdown(
    """
    <div class="hero">
      <h1>Calculadora interactiva de distribuciones</h1>
      <p>Calcule probabilidades a partir de eventos o determine límites de eventos a partir de una probabilidad. La aplicación representa simultáneamente la función de masa o densidad y la función de distribución acumulada.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuración")
    selected_distribution = st.selectbox(
        "Distribución",
        options=list(DISTRIBUTIONS),
        index=list(DISTRIBUTIONS).index("Normal"),
        key="selected_distribution",
        on_change=clear_result,
    )
    mode = st.radio(
        "Modo de cálculo",
        options=CALCULATION_MODES,
        key="calculation_mode",
        on_change=clear_result,
    )

spec = DISTRIBUTIONS[selected_distribution]
type_class = "type-discrete" if spec.kind == "discreta" else "type-continuous"
type_label = "● DISCRETA" if spec.kind == "discreta" else "○ CONTINUA"

st.markdown(
    f"""
    <div class="distribution-card">
      <span class="{type_class}">{type_label}</span>
      <span style="color:#5D6878; margin-left:.6rem;">{spec.description}</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.latex(spec.formula)

left_column, right_column = st.columns([0.95, 1.05], gap="large")

with left_column:
    st.subheader("Parámetros de la distribución")
    with st.form("distribution_calculator", clear_on_submit=False, border=True):
        raw_parameters: dict[str, float] = {}
        for parameter in spec.parameters:
            widget_key = f"parameter::{spec.name}::{parameter.key}"
            if parameter.kind == "int":
                raw_parameters[parameter.key] = st.number_input(
                    parameter.label,
                    value=int(parameter.default),
                    step=1,
                    format="%d",
                    help=f"Restricción: {parameter.constraint}",
                    key=widget_key,
                )
            else:
                raw_parameters[parameter.key] = st.number_input(
                    parameter.label,
                    value=float(parameter.default),
                    step=0.1,
                    format="%.6f",
                    help=f"Restricción: {parameter.constraint}",
                    key=widget_key,
                )

        st.divider()
        inverse_mode = mode == CALCULATION_MODES[1]
        event_options = INVERSE_EVENTS if inverse_mode else DIRECT_EVENTS
        event = st.selectbox(
            "Evento",
            options=event_options,
            key=f"event::{mode}",
        )

        value_1 = None
        value_2 = None
        target_probability = None
        known_limit = None

        if not inverse_mode:
            if event == "P(a ≤ X ≤ b)":
                interval_columns = st.columns(2)
                with interval_columns[0]:
                    value_1 = st.number_input("a", value=0.0, step=0.1, format="%.8f", key=f"a::{spec.name}::{event}")
                with interval_columns[1]:
                    value_2 = st.number_input("b", value=1.0, step=0.1, format="%.8f", key=f"b::{spec.name}::{event}")
            else:
                value_1 = st.number_input("x", value=0.0, step=0.1, format="%.8f", key=f"x::{spec.name}::{event}")
        else:
            target_probability = st.number_input(
                "Probabilidad objetivo p",
                min_value=0.000000001,
                max_value=0.999999999,
                value=0.95,
                step=0.01,
                format="%.9f",
                key=f"p_target::{spec.name}::{event}",
                help="Debe cumplirse 0 < p < 1.",
            )
            if event.startswith("P(a ≤ X ≤ b)"):
                known_label = "Límite conocido a" if "a conocido" in event else "Límite conocido b"
                known_limit = st.number_input(
                    known_label,
                    value=0.0,
                    step=0.1,
                    format="%.8f",
                    key=f"known_limit::{spec.name}::{event}",
                )

        submit_label = "Calcular evento y graficar" if inverse_mode else "Calcular y graficar"
        submitted = st.form_submit_button(submit_label, use_container_width=True, type="primary")

    if submitted:
        try:
            parameters = normalize_parameters(spec, raw_parameters)
            distribution = spec.build(parameters)
            if inverse_mode:
                calculated_event, calculated_value_1, calculated_value_2, achieved_probability = calculate_event_from_probability(
                    distribution,
                    spec.kind,
                    event,
                    float(target_probability),
                    float(known_limit) if known_limit is not None else None,
                )
                result_latex = (
                    rf"p_{{\mathrm{{objetivo}}}}={float(target_probability):.8g}"
                    rf"\;\Longrightarrow\;"
                    rf"{event_to_latex(calculated_event, calculated_value_1, calculated_value_2)}"
                    rf"={achieved_probability:.10f}"
                )
                st.session_state["calculation_result"] = {
                    "distribution_name": selected_distribution,
                    "mode": mode,
                    "parameters": parameters,
                    "event": calculated_event,
                    "value_1": calculated_value_1,
                    "value_2": calculated_value_2,
                    "probability": achieved_probability,
                    "target_probability": float(target_probability),
                    "result_latex": result_latex,
                }
            else:
                probability = calculate_probability(
                    distribution,
                    spec.kind,
                    event,
                    float(value_1),
                    float(value_2) if value_2 is not None else None,
                )
                st.session_state["calculation_result"] = {
                    "distribution_name": selected_distribution,
                    "mode": mode,
                    "parameters": parameters,
                    "event": event,
                    "value_1": float(value_1),
                    "value_2": float(value_2) if value_2 is not None else None,
                    "probability": probability,
                    "target_probability": None,
                    "result_latex": None,
                }
        except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
            st.session_state.pop("calculation_result", None)
            st.error(f"Revisemos los datos: {error}")
        except Exception as error:
            st.session_state.pop("calculation_result", None)
            st.error(f"No fue posible completar el cálculo: {error}")

with right_column:
    st.subheader("Resultado")
    result = st.session_state.get("calculation_result")
    if result is None or result.get("distribution_name") != selected_distribution or result.get("mode") != mode:
        st.info("Configure los parámetros y el evento, luego presione el botón de cálculo.")
    else:
        result_spec = DISTRIBUTIONS[result["distribution_name"]]
        result_distribution = result_spec.build(result["parameters"])
        mean, variance, deviation = distribution_moments(result_distribution)
        metric_columns = st.columns(3)
        metric_columns[0].metric("Media", format_moment(mean))
        metric_columns[1].metric("Desviación estándar", format_moment(deviation))
        metric_columns[2].metric("Varianza", format_moment(variance))

        if result["result_latex"]:
            st.latex(result["result_latex"])
        else:
            st.latex(
                rf"{event_to_latex(result['event'], result['value_1'], result['value_2'])}"
                rf"\;=\;{result['probability']:.10f}"
            )

        if result_spec.kind == "discreta" and result["target_probability"] is not None:
            difference = abs(result["probability"] - result["target_probability"])
            if difference > 1e-10:
                st.caption(
                    "La distribución es discreta: no siempre existe un límite que produzca exactamente p. "
                    f"Se muestra el evento alcanzable más cercano (diferencia absoluta = {difference:.3g})."
                )

        figure, outside_range = create_distribution_figure(
            result_spec,
            result["parameters"],
            result["event"],
            result["value_1"],
            result["value_2"],
            result["probability"],
            result_latex=result["result_latex"],
        )
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)
        if outside_range:
            st.caption(
                "Uno o más límites quedan fuera del rango central usado para visualizar la distribución. "
                "La probabilidad se calculó con la distribución completa, no con el rango del gráfico."
            )
