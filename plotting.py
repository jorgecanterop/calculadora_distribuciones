"""Gráficos Matplotlib para la calculadora de distribuciones."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from distribution_engine import (
    DistributionSpec,
    build_plot_axis,
    distribution_moments,
    event_mask,
    event_to_latex,
    format_moment,
)

PALETTES = {
    "light": {
        "figure_bg": "#FFFFFF",
        "axes_bg": "#FCFCFD",
        "text": "#1F2937",
        "text_dim": "#586474",
        "grid": "#D9E2EC",
        "highlight": "#D8867F",
        "result": "#3E8B62",
        "panel_bg": "#F7FAFC",
    },
    "dark": {
        "figure_bg": "#0E141D",
        "axes_bg": "#151D28",
        "text": "#E7EEF6",
        "text_dim": "#B7C5D6",
        "grid": "#3B4A5E",
        "highlight": "#F29A90",
        "result": "#67D39A",
        "panel_bg": "#1A2330",
    },
}

DISTRIBUTION_COLORS = {
    "light": {
        "Bernoulli": ("#5DADE2", "#D6EAF8"),
        "Binomial": ("#D988B9", "#F5D5E6"),
        "Poisson": ("#58B58B", "#D7F0E3"),
        "Binomial negativa": ("#8A7FD1", "#E5E1F5"),
        "Geométrica": ("#D99A55", "#F7E3CB"),
        "Hipergeométrica": ("#4FA8A3", "#D4EEEC"),
        "Normal": ("#8B78C6", "#E4DDF2"),
        "Exponencial": ("#D59A42", "#F8E3BF"),
        "Uniforme": ("#5CA9D6", "#DCEFF8"),
        "Gamma": ("#C27850", "#F1DCCE"),
        "Chi-cuadrado": ("#62A875", "#DCEEDD"),
        "t de Student": ("#B76E8B", "#EFD8E2"),
        "F de Fisher": ("#7787C7", "#DEE2F3"),
    },
    "dark": {
        "Bernoulli": ("#78C7F2", "#24465A"),
        "Binomial": ("#E79BCB", "#59344A"),
        "Poisson": ("#72D1A5", "#244E3D"),
        "Binomial negativa": ("#A99BEA", "#403A68"),
        "Geométrica": ("#E6B16E", "#5B4328"),
        "Hipergeométrica": ("#71C9C4", "#264E4C"),
        "Normal": ("#A999E1", "#413A63"),
        "Exponencial": ("#E5B35F", "#5E4724"),
        "Uniforme": ("#78BFE8", "#294B60"),
        "Gamma": ("#DB9470", "#5A3B2F"),
        "Chi-cuadrado": ("#7BC98D", "#2C5136"),
        "t de Student": ("#D38AAA", "#583646"),
        "F de Fisher": ("#97A5E2", "#394263"),
    },
}


def _theme_name(theme: str) -> str:
    normalized = str(theme).lower()
    return normalized if normalized in PALETTES else "light"


def _prepare_axis(axis, palette: dict[str, str]) -> None:
    axis.set_facecolor(palette["axes_bg"])
    axis.grid(True, color=palette["grid"], alpha=0.58, linewidth=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(palette["grid"])
    axis.tick_params(axis="both", colors=palette["text_dim"], labelsize=9)
    axis.xaxis.label.set_color(palette["text_dim"])
    axis.yaxis.label.set_color(palette["text_dim"])


def _draw_discrete_density(
    axis,
    distribution,
    x,
    mask,
    sampled,
    line_color,
    fill_color,
    probability,
    palette: dict[str, str],
) -> None:
    y = np.asarray(distribution.pmf(x), dtype=float)
    if sampled:
        colors = np.where(mask, palette["highlight"], line_color)
        axis.vlines(x, 0, y, colors=colors, linewidth=1.35, alpha=0.9)
        axis.scatter(x, y, c=colors, s=17, zorder=4)
        axis.text(
            0.02,
            0.97,
            "Soporte amplio: se muestran puntos representativos.",
            transform=axis.transAxes,
            va="top",
            color=palette["text_dim"],
            fontsize=8,
        )
    else:
        width = 0.62 if len(x) > 2 else 0.35
        bars = axis.bar(
            x,
            y,
            width=width,
            color=fill_color,
            edgecolor=line_color,
            linewidth=1.0,
            alpha=0.9,
            zorder=3,
        )
        for selected, bar in zip(mask, bars):
            if selected:
                bar.set_color(palette["highlight"])
                bar.set_edgecolor(palette["highlight"])
                bar.set_alpha(0.96)
    axis.set_xlabel("k")
    axis.set_ylabel("f(k) = P(X = k)")
    axis.set_title(
        f"Función de densidad · probabilidad seleccionada = {probability:.8f}",
        color=palette["text_dim"],
        fontsize=11,
        pad=10,
    )


def _draw_continuous_density(
    axis,
    distribution,
    x,
    mask,
    event,
    value_1,
    value_2,
    line_color,
    fill_color,
    probability,
    palette: dict[str, str],
) -> None:
    with np.errstate(all="ignore"):
        y = np.asarray(distribution.pdf(x), dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=np.nan, neginf=0.0)
    if np.any(np.isfinite(y)):
        finite = y[np.isfinite(y)]
        cap = np.quantile(finite, 0.995) * 1.35 if finite.size else 1.0
        if np.isfinite(cap) and cap > 0:
            y = np.nan_to_num(y, nan=cap, posinf=cap, neginf=0.0)
            y = np.clip(y, 0.0, cap)
        else:
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    axis.plot(x, y, color=line_color, linewidth=2.5, zorder=4)
    axis.fill_between(x, y, color=fill_color, alpha=0.30)

    if event == "P(X = x)":
        axis.axvline(value_1, color=palette["highlight"], linewidth=2, linestyle="--")
        axis.text(
            0.03,
            0.93,
            "En una distribución continua, P(X = x) = 0.",
            transform=axis.transAxes,
            va="top",
            color=palette["highlight"],
            fontsize=9,
        )
    else:
        axis.fill_between(
            x,
            y,
            where=mask,
            interpolate=True,
            color=palette["highlight"],
            alpha=0.72,
            zorder=3,
        )

    limits = [value_1]
    if event == "P(a ≤ X ≤ b)" and value_2 is not None:
        limits.append(value_2)
    for limit in limits:
        axis.axvline(limit, color=palette["highlight"], linewidth=1.35, linestyle=":")

    axis.set_xlabel("x")
    axis.set_ylabel("f(x)")
    axis.set_title(
        f"Función de densidad · área seleccionada = {probability:.8f}",
        color=palette["text_dim"],
        fontsize=11,
        pad=10,
    )


def _draw_cdf(
    axis,
    distribution,
    spec,
    x,
    event,
    value_1,
    value_2,
    line_color,
    fill_color,
    palette: dict[str, str],
) -> None:
    cdf = np.asarray(distribution.cdf(x), dtype=float)
    if spec.kind == "discreta":
        axis.step(x, cdf, where="post", color=line_color, linewidth=2.3)
        axis.fill_between(x, cdf, step="post", color=fill_color, alpha=0.25)
    else:
        axis.plot(x, cdf, color=line_color, linewidth=2.5)
        axis.fill_between(x, cdf, color=fill_color, alpha=0.23)

    marks = [value_1]
    if event == "P(a ≤ X ≤ b)" and value_2 is not None:
        marks.append(value_2)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    for index, value in enumerate(marks):
        plotted_value = float(np.floor(value)) if spec.kind == "discreta" else float(value)
        cdf_value = float(distribution.cdf(plotted_value))
        if x_min <= plotted_value <= x_max:
            axis.scatter(
                [plotted_value],
                [cdf_value],
                color=palette["highlight"],
                s=68,
                edgecolors=palette["axes_bg"],
                linewidth=0.8,
                zorder=6,
            )
            axis.axhline(
                cdf_value,
                color=palette["highlight"],
                linewidth=1,
                linestyle="--",
                alpha=0.65,
            )
            axis.axvline(
                plotted_value,
                color=palette["highlight"],
                linewidth=1,
                linestyle="--",
                alpha=0.65,
            )
            offset = -0.13 if index == 0 else 0.07
            axis.annotate(
                f"F({value:.4g}) = {cdf_value:.4f}",
                xy=(plotted_value, cdf_value),
                xytext=(plotted_value, np.clip(cdf_value + offset, 0.04, 1.02)),
                color=palette["highlight"],
                fontsize=8,
            )

    axis.set_ylim(-0.03, 1.08)
    axis.set_xlabel("x" if spec.kind == "continua" else "k")
    axis.set_ylabel("F(x) = P(X ≤ x)")
    axis.set_title("Función de distribución", color=palette["text_dim"], fontsize=11, pad=10)


def create_distribution_figure(
    spec: DistributionSpec,
    parameters: dict[str, float],
    event: str,
    value_1: float,
    value_2: float | None,
    probability: float,
    result_latex: str | None = None,
    theme: str = "light",
) -> tuple[Figure, bool]:
    """Crea la figura y señala si algún límite queda fuera del rango visible."""
    theme_name = _theme_name(theme)
    palette = PALETTES[theme_name]
    distribution = spec.build(parameters)
    x, sampled = build_plot_axis(distribution, spec.kind)
    line_color, fill_color = DISTRIBUTION_COLORS[theme_name][spec.name]
    mean, variance = distribution_moments(distribution)
    mask = event_mask(x, spec.kind, event, value_1, value_2)

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11.8, 10.8),
        facecolor=palette["figure_bg"],
    )
    density_axis, cdf_axis = axes
    _prepare_axis(density_axis, palette)
    _prepare_axis(cdf_axis, palette)

    parameter_text = "  ·  ".join(f"{key}={value:.5g}" for key, value in parameters.items())
    figure.suptitle(
        f"{spec.name}  ({parameter_text})\n"
        f"media = {format_moment(mean)}    ·    varianza = {format_moment(variance)}",
        fontsize=13,
        fontweight="bold",
        color=palette["text"],
        y=0.975,
    )

    if spec.kind == "discreta":
        _draw_discrete_density(
            density_axis,
            distribution,
            x,
            mask,
            sampled,
            line_color,
            fill_color,
            probability,
            palette,
        )
    else:
        _draw_continuous_density(
            density_axis,
            distribution,
            x,
            mask,
            event,
            value_1,
            value_2,
            line_color,
            fill_color,
            probability,
            palette,
        )

    _draw_cdf(
        cdf_axis,
        distribution,
        spec,
        x,
        event,
        value_1,
        value_2,
        line_color,
        fill_color,
        palette,
    )

    result_text = result_latex or rf"{event_to_latex(event, value_1, value_2)}\;=\;{probability:.10f}"
    figure.text(
        0.5,
        0.022,
        rf"$\;{result_text}\;$",
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="bold",
        color=palette["result"],
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": palette["panel_bg"],
            "edgecolor": palette["result"],
            "linewidth": 1.2,
        },
    )
    figure.subplots_adjust(left=0.085, right=0.98, top=0.86, bottom=0.09, hspace=0.42)

    visible_min, visible_max = float(np.min(x)), float(np.max(x))
    limits = [value_1] + ([value_2] if value_2 is not None else [])
    outside_visible_range = any(
        value is not None and not (visible_min <= value <= visible_max) for value in limits
    )
    return figure, outside_visible_range
