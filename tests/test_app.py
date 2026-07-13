"""Pruebas de interacción para la interfaz Streamlit."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def run_app() -> AppTest:
    app = AppTest.from_file(str(APP_PATH), default_timeout=20).run()
    assert not app.exception
    return app


def number_input_labels(app: AppTest) -> list[str]:
    return [widget.label for widget in app.number_input]


def test_direct_event_fields_update_immediately():
    app = run_app()
    assert number_input_labels(app)[-1:] == ["x"]

    app.selectbox[2].set_value("P(a ≤ X ≤ b)").run()
    assert not app.exception
    labels = number_input_labels(app)
    assert "a" in labels
    assert "b" in labels
    assert "x" not in labels

    app.selectbox[2].set_value("P(X ≥ x)").run()
    assert not app.exception
    labels = number_input_labels(app)
    assert labels[-1:] == ["x"]
    assert "a" not in labels
    assert "b" not in labels


def test_inverse_event_fields_update_immediately():
    app = run_app()
    app.selectbox[1].set_value("Evento a partir de una probabilidad").run()
    assert not app.exception
    assert "Probabilidad objetivo p" in number_input_labels(app)

    app.selectbox[2].set_value("P(a ≤ X ≤ b) = p, con a conocido").run()
    assert not app.exception
    labels = number_input_labels(app)
    assert "Probabilidad objetivo p" in labels
    assert "Límite conocido a" in labels
    assert "Límite conocido b" not in labels

    app.selectbox[2].set_value("P(a ≤ X ≤ b) = p, con b conocido").run()
    assert not app.exception
    labels = number_input_labels(app)
    assert "Probabilidad objetivo p" in labels
    assert "Límite conocido b" in labels
    assert "Límite conocido a" not in labels


def test_selectboxes_are_strict_and_form_batching_is_not_used():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "filter_mode=None" in source
    assert "accept_new_options=False" in source
    assert "with st.form(" not in source
    assert "st.form_submit_button(" not in source


def test_calculation_still_runs_after_reactive_update():
    app = run_app()
    app.selectbox[2].set_value("P(a ≤ X ≤ b)").run()
    assert not app.exception

    inputs = {widget.label: widget for widget in app.number_input}
    inputs["a"].set_value(-1.0)
    inputs["b"].set_value(1.0)
    app.button[0].click().run()

    assert not app.exception
    assert [metric.label for metric in app.metric] == ["Media", "Varianza"]
    assert len(app.image) == 1

    page_text = " ".join(element.value for element in app.markdown)
    assert "Probabilidad calculada" in page_text
    assert "Desviación estándar" not in page_text
