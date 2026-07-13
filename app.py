"""Aplicación Streamlit: calculadora interactiva de distribuciones."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: #F7F9FC; }
      .block-container {
          max-width: 1240px;
          padding-top: 1.25rem;
          padding-bottom: 2.5rem;
      }
      .hero {
          background: white;
          border: 1px solid #D9E2EC;
          border-radius: 16px;
          padding: 1.15rem 1.35rem;
          margin-bottom: 1rem;
          box-shadow: 0 4px 18px rgba(31, 41, 55, 0.05);
      }
      .hero h1 { margin: 0 0 .25rem 0; color: #1F2937; font-size: 2rem; }
      .hero p { margin: 0; color: #5D6878; line-height: 1.55; }
      .distribution-card {
          background: white;
          border: 1px solid #D9E2EC;
          border-radius: 14px;
          padding: .9rem 1rem;
          margin: .4rem 0 1rem 0;
      }
      .type-discrete { color: #3D8BB8; font-weight: 700; }
      .type-continuous { color: #9B5E86; font-weight: 700; }
      [data-testid="stMetric"] {
          background: white;
          border: 1px solid #D9E2EC;
          border-radius: 12px;
          padding: .55rem .75rem;
      }
      .primary-result {
          background: #F1F8F4;
          border: 2px solid #3E8B62;
          border-radius: 15px;
          padding: .95rem 1.1rem .8rem 1.1rem;
          margin: .25rem 0 1rem 0;
      }
      .primary-result-title {
          color: #2F6F4E;
          font-size: .9rem;
          font-weight: 800;
          letter-spacing: .035em;
          text-transform: uppercase;
          margin-bottom: .25rem;
      }
      .primary-result-value {
          color: #1F2937;
          font-size: 2rem;
          line-height: 1.15;
          font-weight: 800;
          margin-bottom: .15rem;
          overflow-wrap: anywhere;
      }
      .primary-result-note {
          color: #586474;
          font-size: .93rem;
          margin: 0;
      }
      [data-testid="stExpander"] {
          background: white;
          border: 1px solid #D9E2EC;
          border-radius: 14px;
          overflow: hidden;
      }
      div[data-testid="stAlert"] { border-radius: 12px; }
      .stButton > button {
          border-radius: 10px;
          min-height: 2.75rem;
          font-weight: 700;
      }
      @media (max-width: 700px) {
          .block-container { padding-left: .75rem; padding-right: .75rem; }
          .hero h1 { font-size: 1.55rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def clear_result() -> None:
    """Descarta resultados que ya no corresponden a la configuración visible."""
    st.session_state.pop("calculation_result", None)


def strict_selectbox(label: str, options, **kwargs):
    """Crea una lista desplegable que solo admite selección con el puntero.

    ``filter_mode=None`` desactiva la escritura y el filtrado por teclado, mientras
    que ``accept_new_options=False`` impide agregar valores ajenos a la lista.
    Estas opciones están disponibles a partir de Streamlit 1.58.
    """
    return st.selectbox(
        label,
        options=options,
        accept_new_options=False,
        filter_mode=None,
        **kwargs,
    )


st.markdown(
    """
    <div class="hero">
      <h1>Calculadora interactiva de distribuciones</h1>
      <p>Calcule probabilidades a partir de eventos o determine límites de eventos a partir de una probabilidad. La aplicación representa la función de densidad y la función de distribución acumulada.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# La configuración general se ubica en el contenido principal para que siga
# siendo accesible cuando la aplicación se incrusta en Google Sites mediante iframe.
with st.expander("Configuración general", expanded=True):
    configuration_columns = st.columns(2, gap="large")
    with configuration_columns[0]:
        selected_distribution = strict_selectbox(
            "Distribución",
            options=list(DISTRIBUTIONS),
            index=list(DISTRIBUTIONS).index("Normal"),
            key="selected_distribution",
            on_change=clear_result,
        )
    with configuration_columns[1]:
        mode = strict_selectbox(
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

st.subheader("Parámetros y evento")

# Los controles no se agrupan en st.form. De esta manera Streamlit vuelve a
# ejecutar la página inmediatamente al cambiar el evento y puede mostrar los
# campos que corresponden a la nueva selección. El cálculo continúa ejecutándose
# solamente cuando se presiona el botón principal.
with st.container(border=True):
    st.markdown("#### Parámetros de la distribución")
    raw_parameters: dict[str, float] = {}

    parameter_columns = st.columns(min(3, len(spec.parameters)), gap="large")
    for index, parameter in enumerate(spec.parameters):
        widget_key = f"parameter::{spec.name}::{parameter.key}"
        with parameter_columns[index % len(parameter_columns)]:
            if parameter.kind == "int":
                raw_parameters[parameter.key] = st.number_input(
                    parameter.label,
                    value=int(parameter.default),
                    step=1,
                    format="%d",
                    help=f"Restricción: {parameter.constraint}",
                    key=widget_key,
                    on_change=clear_result,
                )
            else:
                raw_parameters[parameter.key] = st.number_input(
                    parameter.label,
                    value=float(parameter.default),
                    step=0.01,
                    format="%.2f",
                    help=f"Restricción: {parameter.constraint}",
                    key=widget_key,
                    on_change=clear_result,
                )

    st.divider()
    st.markdown("#### Evento")
    inverse_mode = mode == CALCULATION_MODES[1]
    event_options = INVERSE_EVENTS if inverse_mode else DIRECT_EVENTS
    event = strict_selectbox(
        "Tipo de evento",
        options=event_options,
        key=f"event::{mode}",
        on_change=clear_result,
    )

    value_1 = None
    value_2 = None
    target_probability = None
    known_limit = None

    if not inverse_mode:
        if event == "P(a ≤ X ≤ b)":
            interval_columns = st.columns(2, gap="large")
            with interval_columns[0]:
                value_1 = st.number_input(
                    "a",
                    value=0.0,
                    step=0.01,
                    format="%.2f",
                    key=f"a::{spec.name}::{event}",
                    on_change=clear_result,
                )
            with interval_columns[1]:
                value_2 = st.number_input(
                    "b",
                    value=1.0,
                    step=0.01,
                    format="%.2f",
                    key=f"b::{spec.name}::{event}",
                    on_change=clear_result,
                )
        else:
            value_1 = st.number_input(
                "x",
                value=0.0,
                step=0.01,
                format="%.2f",
                key=f"x::{spec.name}::{event}",
                on_change=clear_result,
            )
    else:
        target_probability = st.number_input(
            "Probabilidad objetivo p",
            min_value=0.000000001,
            max_value=0.999999999,
            value=0.95,
            step=0.01,
            format="%.4f",
            key=f"p_target::{spec.name}::{event}",
            help="Debe cumplirse 0 < p < 1.",
            on_change=clear_result,
        )
        if event.startswith("P(a ≤ X ≤ b)"):
            known_label = "Límite conocido a" if "a conocido" in event else "Límite conocido b"
            known_limit = st.number_input(
                known_label,
                value=0.0,
                step=0.01,
                format="%.2f",
                key=f"known_limit::{spec.name}::{event}",
                on_change=clear_result,
            )

    submit_label = "Calcular evento y graficar" if inverse_mode else "Calcular y graficar"
    submitted = st.button(
        submit_label,
        width="stretch",
        type="primary",
        key="calculate_button",
    )

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
                "inverse_event_request": event,
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
                "inverse_event_request": None,
                "result_latex": None,
            }
    except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
        st.session_state.pop("calculation_result", None)
        st.error(f"Revisemos los datos: {error}")
    except Exception as error:
        st.session_state.pop("calculation_result", None)
        st.error(f"No fue posible completar el cálculo: {error}")

st.subheader("Resultado")
result = st.session_state.get("calculation_result")
if result is None or result.get("distribution_name") != selected_distribution or result.get("mode") != mode:
    st.info("Configure los parámetros y el evento, luego presione el botón de cálculo.")
else:
    result_spec = DISTRIBUTIONS[result["distribution_name"]]
    result_distribution = result_spec.build(result["parameters"])
    mean, variance = distribution_moments(result_distribution)

    if result["target_probability"] is None:
        primary_title = "Probabilidad calculada"
        primary_value = f"{result['probability']:.10f}"
        primary_note = f"Equivale aproximadamente a {100 * result['probability']:.4f} %."
    else:
        inverse_request = result.get("inverse_event_request", "")
        if "a conocido" in inverse_request:
            primary_title = "Límite superior calculado (b)"
            calculated_limit = result["value_2"]
        elif "b conocido" in inverse_request:
            primary_title = "Límite inferior calculado (a)"
            calculated_limit = result["value_1"]
        else:
            primary_title = "Cuantil calculado (x)"
            calculated_limit = result["value_1"]

        if result_spec.kind == "discreta" and np.isclose(calculated_limit, round(calculated_limit)):
            primary_value = str(int(round(calculated_limit)))
        elif calculated_limit == 0:
            primary_value = "0"
        elif abs(calculated_limit) >= 1e6 or abs(calculated_limit) < 1e-5:
            primary_value = f"{calculated_limit:.6e}"
        else:
            primary_value = f"{calculated_limit:.6f}".rstrip("0").rstrip(".")

        primary_note = (
            f"Probabilidad objetivo: {result['target_probability']:.8g}. "
            f"Probabilidad obtenida: {result['probability']:.10f}."
        )

    st.markdown(
        f"""
        <div class="primary-result">
          <div class="primary-result-title">{primary_title}</div>
          <div class="primary-result-value">{primary_value}</div>
          <p class="primary-result-note">{primary_note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    st.markdown("#### Información adicional de la distribución")
    information_columns = st.columns(2, gap="large")
    information_columns[0].metric("Media", format_moment(mean))
    information_columns[1].metric("Varianza", format_moment(variance))

    figure, outside_range = create_distribution_figure(
        result_spec,
        result["parameters"],
        result["event"],
        result["value_1"],
        result["value_2"],
        result["probability"],
        result_latex=result["result_latex"],
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    if outside_range:
        st.caption(
            "Uno o más límites quedan fuera del rango central usado para visualizar la distribución. "
            "La probabilidad se calculó con la distribución completa, no con el rango del gráfico."
        )
