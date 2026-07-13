import math

import pytest

from distribution_engine import (
    DISTRIBUTIONS,
    calculate_event_from_probability,
    calculate_probability,
    normalize_parameters,
)


@pytest.mark.parametrize("name", list(DISTRIBUTIONS))
def test_default_parameters_build_and_probability(name):
    spec = DISTRIBUTIONS[name]
    raw = {parameter.key: parameter.default for parameter in spec.parameters}
    params = normalize_parameters(spec, raw)
    distribution = spec.build(params)
    probability = calculate_probability(distribution, spec.kind, "P(X ≤ x)", 0.0)
    assert math.isfinite(probability)
    assert 0.0 <= probability <= 1.0


def test_normal_inverse_quantile():
    spec = DISTRIBUTIONS["Normal"]
    params = normalize_parameters(spec, {"mu": 0.0, "sigma": 1.0})
    distribution = spec.build(params)
    event, x, _, achieved = calculate_event_from_probability(distribution, spec.kind, "P(X ≤ x) = p", 0.975)
    assert event == "P(X ≤ x)"
    assert x == pytest.approx(1.9599639845, rel=1e-8)
    assert achieved == pytest.approx(0.975, abs=1e-12)


def test_discrete_non_integer_boundaries():
    spec = DISTRIBUTIONS["Poisson"]
    params = normalize_parameters(spec, {"lambda": 4.0})
    distribution = spec.build(params)
    assert calculate_probability(distribution, spec.kind, "P(X ≤ x)", 3.9) == pytest.approx(distribution.cdf(3))
    assert calculate_probability(distribution, spec.kind, "P(X ≥ x)", 3.1) == pytest.approx(distribution.sf(3))
    assert calculate_probability(distribution, spec.kind, "P(X = x)", 3.2) == 0.0


def test_invalid_uniform_parameters():
    spec = DISTRIBUTIONS["Uniforme"]
    with pytest.raises(ValueError):
        normalize_parameters(spec, {"a": 2.0, "b": 2.0})
