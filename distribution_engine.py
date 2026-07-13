"""Motor estadístico de la calculadora de distribuciones.

Este módulo no depende de Streamlit. Puede probarse y reutilizarse desde otros
entornos sin cargar la interfaz gráfica.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ParameterSpec:
    key: str
    label: str
    default: float
    constraint: str
    kind: str = "float"


@dataclass(frozen=True)
class DistributionSpec:
    name: str
    kind: str
    parameters: tuple[ParameterSpec, ...]
    description: str
    formula: str
    build: Callable[[Mapping[str, float]], object]
    validate: Callable[[Mapping[str, float]], None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_bernoulli(p: Mapping[str, float]) -> None:
    _require(0 <= p["p"] <= 1, "p debe pertenecer al intervalo [0, 1].")


def _validate_binomial(p: Mapping[str, float]) -> None:
    _require(p["n"] >= 1, "n debe ser un entero positivo.")
    _require(0 <= p["p"] <= 1, "p debe pertenecer al intervalo [0, 1].")


def _validate_poisson(p: Mapping[str, float]) -> None:
    _require(p["lambda"] > 0, "λ debe ser mayor que 0.")


def _validate_negative_binomial(p: Mapping[str, float]) -> None:
    _require(p["r"] >= 1, "r debe ser un entero positivo.")
    _require(0 < p["p"] <= 1, "p debe pertenecer al intervalo (0, 1].")


def _validate_geometric(p: Mapping[str, float]) -> None:
    _require(0 < p["p"] <= 1, "p debe pertenecer al intervalo (0, 1].")


def _validate_hypergeometric(p: Mapping[str, float]) -> None:
    population, successes, draws = p["N"], p["K"], p["n"]
    _require(population >= 1, "N debe ser un entero positivo.")
    _require(0 <= successes <= population, "K debe satisfacer 0 ≤ K ≤ N.")
    _require(0 <= draws <= population, "n debe satisfacer 0 ≤ n ≤ N.")


def _validate_normal(p: Mapping[str, float]) -> None:
    _require(p["sigma"] > 0, "σ debe ser mayor que 0.")


def _validate_exponential(p: Mapping[str, float]) -> None:
    _require(p["lambda"] > 0, "λ debe ser mayor que 0.")


def _validate_uniform(p: Mapping[str, float]) -> None:
    _require(p["b"] > p["a"], "b debe ser mayor que a.")


def _validate_gamma(p: Mapping[str, float]) -> None:
    _require(p["alpha"] > 0, "α debe ser mayor que 0.")
    _require(p["theta"] > 0, "θ debe ser mayor que 0.")


def _validate_degrees_of_freedom(p: Mapping[str, float]) -> None:
    _require(p["nu"] > 0, "Los grados de libertad ν deben ser mayores que 0.")


def _validate_f(p: Mapping[str, float]) -> None:
    _require(p["d1"] > 0 and p["d2"] > 0, "d₁ y d₂ deben ser mayores que 0.")


DISTRIBUTIONS: dict[str, DistributionSpec] = {
    "Bernoulli": DistributionSpec(
        "Bernoulli",
        "discreta",
        (ParameterSpec("p", "p (probabilidad de éxito)", 0.5, "0 ≤ p ≤ 1"),),
        "Representa un único ensayo con resultado éxito (1) o fracaso (0).",
        r"P(X=k)=p^k(1-p)^{1-k},\quad k\in\{0,1\}",
        lambda p: stats.bernoulli(p=p["p"]),
        _validate_bernoulli,
    ),
    "Binomial": DistributionSpec(
        "Binomial",
        "discreta",
        (
            ParameterSpec("n", "n (número de ensayos)", 10, "n ∈ ℤ⁺", "int"),
            ParameterSpec("p", "p (probabilidad de éxito)", 0.5, "0 ≤ p ≤ 1"),
        ),
        "Cuenta los éxitos obtenidos en n ensayos independientes con probabilidad p constante.",
        r"P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}",
        lambda p: stats.binom(n=p["n"], p=p["p"]),
        _validate_binomial,
    ),
    "Poisson": DistributionSpec(
        "Poisson",
        "discreta",
        (ParameterSpec("lambda", "λ (tasa media)", 4.0, "λ > 0"),),
        "Cuenta eventos que aparecen en un intervalo fijo con una tasa media λ.",
        r"P(X=k)=\dfrac{e^{-\lambda}\lambda^k}{k!}",
        lambda p: stats.poisson(mu=p["lambda"]),
        _validate_poisson,
    ),
    "Binomial negativa": DistributionSpec(
        "Binomial negativa",
        "discreta",
        (
            ParameterSpec("r", "r (número de éxitos)", 5, "r ∈ ℤ⁺", "int"),
            ParameterSpec("p", "p (probabilidad de éxito)", 0.4, "0 < p ≤ 1"),
        ),
        "X cuenta los fracasos observados antes de alcanzar el r-ésimo éxito.",
        r"P(X=k)=\binom{k+r-1}{k}(1-p)^k p^r,\quad k=0,1,\ldots",
        lambda p: stats.nbinom(n=p["r"], p=p["p"]),
        _validate_negative_binomial,
    ),
    "Geométrica": DistributionSpec(
        "Geométrica",
        "discreta",
        (ParameterSpec("p", "p (probabilidad de éxito)", 0.3, "0 < p ≤ 1"),),
        "X es el número de ensayos necesarios hasta obtener el primer éxito; su soporte comienza en 1.",
        r"P(X=k)=(1-p)^{k-1}p,\quad k=1,2,\ldots",
        lambda p: stats.geom(p=p["p"]),
        _validate_geometric,
    ),
    "Hipergeométrica": DistributionSpec(
        "Hipergeométrica",
        "discreta",
        (
            ParameterSpec("N", "N (tamaño de la población)", 40, "N ∈ ℤ⁺", "int"),
            ParameterSpec("K", "K (éxitos en la población)", 12, "0 ≤ K ≤ N", "int"),
            ParameterSpec("n", "n (extracciones sin reemplazo)", 8, "0 ≤ n ≤ N", "int"),
        ),
        "Cuenta éxitos en n extracciones sin reemplazo de una población de N elementos con K éxitos.",
        r"P(X=k)=\dfrac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}",
        lambda p: stats.hypergeom(M=p["N"], n=p["K"], N=p["n"]),
        _validate_hypergeometric,
    ),
    "Normal": DistributionSpec(
        "Normal",
        "continua",
        (
            ParameterSpec("mu", "μ (media)", 0.0, "μ ∈ ℝ"),
            ParameterSpec("sigma", "σ (desviación estándar)", 1.0, "σ > 0"),
        ),
        "Distribución simétrica cuyos valores se concentran alrededor de la media μ.",
        r"f(x)=\dfrac{1}{\sigma\sqrt{2\pi}}\exp\!\left[-\dfrac{(x-\mu)^2}{2\sigma^2}\right]",
        lambda p: stats.norm(loc=p["mu"], scale=p["sigma"]),
        _validate_normal,
    ),
    "Exponencial": DistributionSpec(
        "Exponencial",
        "continua",
        (ParameterSpec("lambda", "λ (tasa)", 1.0, "λ > 0"),),
        "Modela tiempos de espera entre eventos que ocurren con una tasa promedio λ.",
        r"f(x)=\lambda e^{-\lambda x},\quad x\geq 0",
        lambda p: stats.expon(scale=1.0 / p["lambda"]),
        _validate_exponential,
    ),
    "Uniforme": DistributionSpec(
        "Uniforme",
        "continua",
        (
            ParameterSpec("a", "a (mínimo)", 0.0, "a ∈ ℝ"),
            ParameterSpec("b", "b (máximo)", 10.0, "b > a"),
        ),
        "Todos los valores del intervalo [a, b] poseen la misma densidad.",
        r"f(x)=\dfrac{1}{b-a},\quad a\leq x\leq b",
        lambda p: stats.uniform(loc=p["a"], scale=p["b"] - p["a"]),
        _validate_uniform,
    ),
    "Gamma": DistributionSpec(
        "Gamma",
        "continua",
        (
            ParameterSpec("alpha", "α (forma)", 2.0, "α > 0"),
            ParameterSpec("theta", "θ (escala)", 2.0, "θ > 0"),
        ),
        "Modelo positivo y asimétrico; aquí se parametriza mediante forma α y escala θ.",
        r"f(x)=\dfrac{x^{\alpha-1}e^{-x/\theta}}{\Gamma(\alpha)\theta^\alpha},\quad x>0",
        lambda p: stats.gamma(a=p["alpha"], scale=p["theta"]),
        _validate_gamma,
    ),
    "Chi-cuadrado": DistributionSpec(
        "Chi-cuadrado",
        "continua",
        (ParameterSpec("nu", "ν (grados de libertad)", 5.0, "ν > 0"),),
        "Distribución positiva usada, entre otros casos, en inferencia sobre varianzas y pruebas χ².",
        r"f(x)=\dfrac{x^{\nu/2-1}e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)},\quad x>0",
        lambda p: stats.chi2(df=p["nu"]),
        _validate_degrees_of_freedom,
    ),
    "t de Student": DistributionSpec(
        "t de Student",
        "continua",
        (ParameterSpec("nu", "ν (grados de libertad)", 8.0, "ν > 0"),),
        "Distribución simétrica con colas más pesadas que la Normal; converge a ella cuando ν aumenta.",
        r"f(x)=\dfrac{\Gamma\left((\nu+1)/2\right)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)}\left(1+\dfrac{x^2}{\nu}\right)^{-(\nu+1)/2}",
        lambda p: stats.t(df=p["nu"]),
        _validate_degrees_of_freedom,
    ),
    "F de Fisher": DistributionSpec(
        "F de Fisher",
        "continua",
        (
            ParameterSpec("d1", "d₁ (grados de libertad del numerador)", 5.0, "d₁ > 0"),
            ParameterSpec("d2", "d₂ (grados de libertad del denominador)", 12.0, "d₂ > 0"),
        ),
        "Distribución positiva del cociente de dos varianzas escaladas; se usa en ANOVA y pruebas F.",
        r"f(x)=\dfrac{(d_1/d_2)^{d_1/2}x^{d_1/2-1}}{B(d_1/2,d_2/2)\left(1+d_1x/d_2\right)^{(d_1+d_2)/2}},\quad x>0",
        lambda p: stats.f(dfn=p["d1"], dfd=p["d2"]),
        _validate_f,
    ),
}

DIRECT_EVENTS = ("P(X ≤ x)", "P(X ≥ x)", "P(X = x)", "P(a ≤ X ≤ b)")
INVERSE_EVENTS = (
    "P(X ≤ x) = p",
    "P(X ≥ x) = p",
    "P(a ≤ X ≤ b) = p, con a conocido",
    "P(a ≤ X ≤ b) = p, con b conocido",
)
CALCULATION_MODES = (
    "Probabilidad a partir de un evento",
    "Evento a partir de una probabilidad",
)


def normalize_parameters(spec: DistributionSpec, raw_values: Mapping[str, float]) -> dict[str, float]:
    """Convierte tipos, comprueba finitud y aplica las restricciones del modelo."""
    values: dict[str, float] = {}
    for parameter in spec.parameters:
        raw = raw_values[parameter.key]
        if isinstance(raw, bool):
            raise ValueError(f"{parameter.label} no admite valores lógicos.")
        if parameter.kind == "int":
            numeric = float(raw)
            if not np.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(f"{parameter.label} debe ser un entero.")
            value: float | int = int(numeric)
        else:
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"{parameter.label} debe ser finito.")
        values[parameter.key] = value
    spec.validate(values)
    return values


def calculate_probability(distribution, kind: str, event: str, value_1: float, value_2: float | None = None) -> float:
    """Calcula una probabilidad respetando el soporte discreto o continuo."""
    if not np.isfinite(value_1):
        raise ValueError("El límite del evento debe ser finito.")
    if event == "P(a ≤ X ≤ b)":
        if value_2 is None or not np.isfinite(value_2):
            raise ValueError("Debe indicar dos límites finitos para el intervalo.")
        if value_2 < value_1:
            raise ValueError("El límite superior b debe ser mayor o igual que a.")

    if kind == "discreta":
        if event == "P(X ≤ x)":
            result = distribution.cdf(np.floor(value_1))
        elif event == "P(X ≥ x)":
            result = distribution.sf(np.ceil(value_1) - 1)
        elif event == "P(X = x)":
            result = distribution.pmf(int(round(value_1))) if np.isclose(value_1, np.round(value_1)) else 0.0
        elif event == "P(a ≤ X ≤ b)":
            lower = int(np.ceil(value_1))
            upper = int(np.floor(value_2))
            result = 0.0 if lower > upper else distribution.cdf(upper) - distribution.cdf(lower - 1)
        else:
            raise ValueError("Evento no reconocido.")
    else:
        if event == "P(X ≤ x)":
            result = distribution.cdf(value_1)
        elif event == "P(X ≥ x)":
            result = distribution.sf(value_1)
        elif event == "P(X = x)":
            result = 0.0
        elif event == "P(a ≤ X ≤ b)":
            result = distribution.cdf(value_2) - distribution.cdf(value_1)
        else:
            raise ValueError("Evento no reconocido.")

    result = float(result)
    if not np.isfinite(result):
        raise ValueError("El cálculo produjo una probabilidad no finita.")
    return float(np.clip(result, 0.0, 1.0))


def _discrete_candidates(center: float, support_lower: float, support_upper: float, radius: int = 6) -> list[int]:
    center = int(round(float(center)))
    candidates = set(range(center - radius, center + radius + 1))
    if np.isfinite(support_lower):
        candidates.add(int(np.ceil(support_lower)))
    if np.isfinite(support_upper):
        candidates.add(int(np.floor(support_upper)))
    return sorted(
        candidate
        for candidate in candidates
        if (not np.isfinite(support_lower) or candidate >= np.ceil(support_lower))
        and (not np.isfinite(support_upper) or candidate <= np.floor(support_upper))
    )


def calculate_event_from_probability(
    distribution,
    kind: str,
    inverse_event: str,
    target_probability: float,
    known_limit: float | None = None,
) -> tuple[str, float, float | None, float]:
    """Obtiene un límite de evento a partir de una probabilidad objetivo.

    En distribuciones discretas puede no existir un límite que produzca exactamente
    la probabilidad solicitada; se devuelve el evento alcanzable más cercano.
    """
    p = float(target_probability)
    if not np.isfinite(p) or not 0 < p < 1:
        raise ValueError("La probabilidad p debe estar estrictamente entre 0 y 1.")

    support_lower, support_upper = distribution.support()

    if kind == "continua":
        if inverse_event == "P(X ≤ x) = p":
            x = float(distribution.ppf(p))
            if not np.isfinite(x):
                raise ValueError("No se obtuvo un cuantil finito para esa probabilidad.")
            actual = calculate_probability(distribution, kind, "P(X ≤ x)", x)
            return "P(X ≤ x)", x, None, actual

        if inverse_event == "P(X ≥ x) = p":
            x = float(distribution.isf(p))
            if not np.isfinite(x):
                raise ValueError("No se obtuvo un cuantil finito para esa probabilidad.")
            actual = calculate_probability(distribution, kind, "P(X ≥ x)", x)
            return "P(X ≥ x)", x, None, actual

        if known_limit is None or not np.isfinite(known_limit):
            raise ValueError("Debe indicar un límite conocido finito.")
        limit = float(known_limit)

        if inverse_event == "P(a ≤ X ≤ b) = p, con a conocido":
            a = limit
            available = float(distribution.sf(a))
            if p >= available:
                raise ValueError(
                    f"Con a = {a:.5g}, la probabilidad disponible hasta +∞ es "
                    f"{available:.6g}; ingrese un valor de p menor."
                )
            b = float(distribution.ppf(float(distribution.cdf(a)) + p))
            if not np.isfinite(b):
                raise ValueError("No se obtuvo un límite superior finito.")
            actual = calculate_probability(distribution, kind, "P(a ≤ X ≤ b)", a, b)
            return "P(a ≤ X ≤ b)", a, b, actual

        if inverse_event == "P(a ≤ X ≤ b) = p, con b conocido":
            b = limit
            available = float(distribution.cdf(b))
            if p >= available:
                raise ValueError(
                    f"Con b = {b:.5g}, la probabilidad disponible desde −∞ es "
                    f"{available:.6g}; ingrese un valor de p menor."
                )
            a = float(distribution.ppf(float(distribution.cdf(b)) - p))
            if not np.isfinite(a):
                raise ValueError("No se obtuvo un límite inferior finito.")
            actual = calculate_probability(distribution, kind, "P(a ≤ X ≤ b)", a, b)
            return "P(a ≤ X ≤ b)", a, b, actual

        raise ValueError("Evento inverso no reconocido.")

    if inverse_event == "P(X ≤ x) = p":
        center = distribution.ppf(p)
        candidates = _discrete_candidates(center, support_lower, support_upper)
        if not candidates:
            raise ValueError("No se encontró un valor discreto compatible.")
        x = min(candidates, key=lambda k: (abs(float(distribution.cdf(k)) - p), k))
        actual = calculate_probability(distribution, kind, "P(X ≤ x)", x)
        return "P(X ≤ x)", float(x), None, actual

    if inverse_event == "P(X ≥ x) = p":
        center = distribution.ppf(1 - p)
        if not np.isfinite(center):
            center = support_upper if np.isfinite(support_upper) else distribution.ppf(1 - 1e-12)
        candidates = _discrete_candidates(center, support_lower, support_upper)
        if np.isfinite(support_upper):
            candidates.append(int(np.floor(support_upper)) + 1)
        else:
            base = int(round(float(center)))
            candidates.extend([base + 1, base + 2])
        candidates = sorted(set(candidates))
        if not candidates:
            raise ValueError("No se encontró un valor discreto compatible.")
        x = min(candidates, key=lambda k: (abs(float(distribution.sf(k - 1)) - p), k))
        actual = calculate_probability(distribution, kind, "P(X ≥ x)", x)
        return "P(X ≥ x)", float(x), None, actual

    if known_limit is None or not np.isfinite(known_limit):
        raise ValueError("Debe indicar un límite conocido finito.")

    if inverse_event == "P(a ≤ X ≤ b) = p, con a conocido":
        a = int(np.ceil(float(known_limit)))
        available = float(distribution.sf(a - 1))
        if p > available + 1e-12:
            raise ValueError(f"Con a = {a}, la probabilidad máxima disponible es {available:.6g}.")
        target_cdf = min(1.0, float(distribution.cdf(a - 1)) + p)
        center = float(distribution.ppf(target_cdf))
        if not np.isfinite(center):
            center = float(support_upper) if np.isfinite(support_upper) else float(distribution.ppf(1 - 1e-12))
        candidates = _discrete_candidates(center, max(support_lower, a), support_upper)
        candidates = [b for b in candidates if b >= a]
        if not candidates:
            raise ValueError("No se encontró un límite superior compatible.")
        b = min(
            candidates,
            key=lambda value: (
                abs(calculate_probability(distribution, kind, "P(a ≤ X ≤ b)", a, value) - p),
                value,
            ),
        )
        actual = calculate_probability(distribution, kind, "P(a ≤ X ≤ b)", a, b)
        return "P(a ≤ X ≤ b)", float(a), float(b), actual

    if inverse_event == "P(a ≤ X ≤ b) = p, con b conocido":
        b = int(np.floor(float(known_limit)))
        available = float(distribution.cdf(b))
        if p > available + 1e-12:
            raise ValueError(f"Con b = {b}, la probabilidad máxima disponible es {available:.6g}.")
        target_cdf = max(0.0, float(distribution.cdf(b)) - p)
        previous_center = distribution.ppf(target_cdf)
        center_a = int(round(float(previous_center))) + 1
        candidates = _discrete_candidates(center_a, support_lower, min(support_upper, b))
        candidates = [a for a in candidates if a <= b]
        if not candidates:
            raise ValueError("No se encontró un límite inferior compatible.")
        a = min(
            candidates,
            key=lambda value: (
                abs(calculate_probability(distribution, kind, "P(a ≤ X ≤ b)", value, b) - p),
                -value,
            ),
        )
        actual = calculate_probability(distribution, kind, "P(a ≤ X ≤ b)", a, b)
        return "P(a ≤ X ≤ b)", float(a), float(b), actual

    raise ValueError("Evento inverso no reconocido.")


def event_to_latex(event: str, value_1: float, value_2: float | None = None) -> str:
    if event == "P(X ≤ x)":
        return rf"P(X \leq {value_1:.5g})"
    if event == "P(X ≥ x)":
        return rf"P(X \geq {value_1:.5g})"
    if event == "P(X = x)":
        return rf"P(X = {value_1:.5g})"
    if value_2 is None:
        raise ValueError("El evento de intervalo requiere dos límites.")
    return rf"P({value_1:.5g} \leq X \leq {value_2:.5g})"


def distribution_moments(distribution) -> tuple[float, float, float]:
    mean, variance = distribution.stats(moments="mv")
    mean = float(np.asarray(mean))
    variance = float(np.asarray(variance))
    deviation = np.sqrt(variance) if np.isfinite(variance) and variance >= 0 else variance
    return mean, variance, float(deviation)


def format_moment(value: float) -> str:
    if np.isnan(value):
        return "no definida"
    if np.isposinf(value):
        return "∞"
    if np.isneginf(value):
        return "−∞"
    return f"{value:.4f}"


def _quantile_limit(distribution, probability: float, fallback: float) -> float:
    try:
        value = float(distribution.ppf(probability))
        return value if np.isfinite(value) else fallback
    except Exception:
        return fallback


def build_plot_axis(distribution, kind: str) -> tuple[np.ndarray, bool]:
    """Construye un eje estable, centrado en la masa principal de la distribución."""
    support_lower, support_upper = distribution.support()
    if kind == "discreta":
        lower = int(np.ceil(support_lower)) if np.isfinite(support_lower) else int(np.floor(_quantile_limit(distribution, 0.001, 0)))
        upper = int(np.floor(support_upper)) if np.isfinite(support_upper) else int(np.ceil(_quantile_limit(distribution, 0.999, lower + 30)))
        lower = min(lower, upper)
        width = upper - lower + 1
        if width <= 260:
            return np.arange(lower, upper + 1, dtype=int), False
        points = np.unique(np.rint(np.linspace(lower, upper, 260)).astype(int))
        return points, True

    lower = float(support_lower) if np.isfinite(support_lower) else _quantile_limit(distribution, 0.001, -5.0)
    upper = float(support_upper) if np.isfinite(support_upper) else _quantile_limit(distribution, 0.999, 5.0)
    q25 = _quantile_limit(distribution, 0.25, lower)
    q50 = _quantile_limit(distribution, 0.50, (lower + upper) / 2)
    q75 = _quantile_limit(distribution, 0.75, upper)
    iqr = q75 - q25
    if np.isfinite(iqr) and iqr > 0:
        lower = max(lower, q50 - 25 * iqr)
        upper = min(upper, q50 + 25 * iqr)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        lower, upper = -5.0, 5.0
    return np.linspace(lower, upper, 700), False


def event_mask(x: np.ndarray, kind: str, event: str, value_1: float, value_2: float | None = None) -> np.ndarray:
    if event == "P(X ≤ x)":
        return x <= value_1
    if event == "P(X ≥ x)":
        return x >= value_1
    if event == "P(X = x)":
        if kind == "discreta" and np.isclose(value_1, np.round(value_1)):
            return x == int(round(value_1))
        return np.zeros_like(x, dtype=bool)
    return (x >= value_1) & (x <= value_2)
