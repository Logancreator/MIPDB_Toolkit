"""Plot stage-wise regression results for true and predicted values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error


INPUT_FILE = Path(__file__).with_name("hzau_ev.xlsx")
OUTPUT_FILE = Path(__file__).with_name("regression_by_stage.png")

FIG_SIZE = (7.2, 5.2)
OUTPUT_DPI = 600
RANDOM_SEED = 42

TRUE_COLUMN = "True Values"
PREDICTED_COLUMN = "Predicted Values"
GROUP_COLUMN = "Stage"

COLORS = plt.get_cmap("tab10")
FONT_LABEL = 12
FONT_TITLE = 14
FONT_TICK = 10
FONT_LEGEND = 9
FONT_STATS = 8.5


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate the regression table."""
    data = pd.read_excel(path)
    required_columns = {TRUE_COLUMN, PREDICTED_COLUMN, GROUP_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing}")

    data = data[[TRUE_COLUMN, PREDICTED_COLUMN, GROUP_COLUMN]].dropna()
    if data.empty:
        raise ValueError("The input file contains no complete rows to plot.")
    return data


def confidence_band(
    x: np.ndarray,
    y: np.ndarray,
    x_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return a 95% confidence band for the fitted mean response."""
    if len(x) < 3:
        return None

    slope, intercept, *_ = stats.linregress(x, y)
    degrees_of_freedom = len(x) - 2
    residuals = y - (slope * x + intercept)
    residual_error = np.sqrt(np.sum(residuals**2) / degrees_of_freedom)
    x_mean = np.mean(x)
    sum_squared_deviations = np.sum((x - x_mean) ** 2)
    if sum_squared_deviations == 0:
        return None

    t_value = stats.t.ppf(0.975, degrees_of_freedom)
    margin = t_value * residual_error * np.sqrt(
        1 / len(x) + (x_range - x_mean) ** 2 / sum_squared_deviations
    )
    fitted = slope * x_range + intercept
    return fitted - margin, fitted + margin


def draw_group(
    ax: plt.Axes,
    stage: str,
    data: pd.DataFrame,
    color: tuple[float, ...],
    random_state: np.random.Generator,
    stats_index: int,
) -> None:
    """Draw one stage, its fitted line, confidence band, and statistics."""
    x = data[TRUE_COLUMN].to_numpy(dtype=float)
    y = data[PREDICTED_COLUMN].to_numpy(dtype=float)
    sample_count = len(x)

    jitter_scale = 0.012 * max(np.ptp(x), np.ptp(y), 1.0)
    x_plot = x + random_state.normal(0, jitter_scale, sample_count)
    y_plot = y + random_state.normal(0, jitter_scale, sample_count)

    ax.scatter(
        x_plot,
        y_plot,
        s=42,
        color=color,
        alpha=0.68,
        edgecolors="white",
        linewidths=0.35,
        label=f"{stage} (n={sample_count})",
        zorder=3,
    )

    if sample_count >= 2 and np.ptp(x) > 0:
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        x_range = np.linspace(x.min(), x.max(), 120)
        fitted = slope * x_range + intercept
        ax.plot(
            x_range,
            fitted,
            color=color,
            linewidth=2.0,
            label=f"{stage} fit",
            zorder=4,
        )

        band = confidence_band(x, y, x_range)
        if band is not None:
            ax.fill_between(
                x_range,
                band[0],
                band[1],
                color=color,
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
        r_squared = r_value**2
    else:
        r_squared = float("nan")

    rmse = np.sqrt(mean_squared_error(x, y))
    mae = mean_absolute_error(x, y)
    stats_text = (
        f"{stage}\n"
        f"$R^2$ = {r_squared:.3f}\n"
        f"RMSE = {rmse:.3f}\n"
        f"MAE = {mae:.3f}\n"
        f"n = {sample_count}"
    )
    ax.text(
        0.035,
        0.965 - stats_index * 0.18,
        stats_text,
        transform=ax.transAxes,
        fontsize=FONT_STATS,
        color=color,
        va="top",
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": color,
            "alpha": 0.88,
            "linewidth": 0.8,
        },
        zorder=5,
    )


def create_plot(data: pd.DataFrame) -> plt.Figure:
    """Create the stage-wise true-versus-predicted figure."""
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    random_state = np.random.default_rng(RANDOM_SEED)

    stages = list(data[GROUP_COLUMN].drop_duplicates())
    colors = [COLORS(index % COLORS.N) for index in range(len(stages))]
    for stats_index, (stage, group) in enumerate(
        data.groupby(GROUP_COLUMN, sort=False)
    ):
        draw_group(
            ax,
            str(stage),
            group,
            colors[stats_index],
            random_state,
            stats_index,
        )

    all_values = data[[TRUE_COLUMN, PREDICTED_COLUMN]].to_numpy(dtype=float)
    value_min = all_values.min()
    value_max = all_values.max()
    margin = 0.04 * max(value_max - value_min, 1.0)
    axis_min = value_min - margin
    axis_max = value_max + margin

    ax.plot(
        [axis_min, axis_max],
        [axis_min, axis_max],
        color="#333333",
        linestyle=(0, (4, 3)),
        linewidth=1.3,
        alpha=0.8,
        label="1:1 line",
        zorder=1,
    )
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(TRUE_COLUMN, fontsize=FONT_LABEL, labelpad=8)
    ax.set_ylabel(PREDICTED_COLUMN, fontsize=FONT_LABEL, labelpad=8)
    ax.set_title(
        "True vs. Predicted Values by Growth Stage",
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=14,
    )
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_TICK,
        width=0.9,
        length=4,
        direction="out",
    )
    ax.grid(
        True,
        linestyle=":",
        linewidth=0.8,
        color="#8c8c8c",
        alpha=0.35,
        zorder=0,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#444444")

    ax.legend(
        fontsize=FONT_LEGEND,
        frameon=True,
        framealpha=0.94,
        facecolor="white",
        edgecolor="#d0d0d0",
        loc="lower right",
        ncol=2,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.tight_layout(pad=1.2)
    return fig


def main() -> None:
    """Load the data, create the figure, and save it beside this script."""
    data = load_data(INPUT_FILE)
    figure = create_plot(data)
    figure.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
