"""Pruebas de estructura para los gráficos de la aplicación."""

from distribution_engine import DISTRIBUTIONS, calculate_probability, normalize_parameters
from plotting import create_distribution_figure


def test_graphs_are_stacked_vertically_and_use_most_of_the_width():
    spec = DISTRIBUTIONS["Normal"]
    parameters = normalize_parameters(spec, {"mu": 0.0, "sigma": 1.0})
    distribution = spec.build(parameters)
    probability = calculate_probability(
        distribution,
        spec.kind,
        "P(a ≤ X ≤ b)",
        -1.0,
        1.0,
    )

    figure, outside_range = create_distribution_figure(
        spec,
        parameters,
        "P(a ≤ X ≤ b)",
        -1.0,
        1.0,
        probability,
    )

    assert outside_range is False
    assert len(figure.axes) == 2

    upper_position = figure.axes[0].get_position().bounds
    lower_position = figure.axes[1].get_position().bounds
    assert upper_position[1] > lower_position[1]
    assert upper_position[2] > 0.80
    assert lower_position[2] > 0.80
