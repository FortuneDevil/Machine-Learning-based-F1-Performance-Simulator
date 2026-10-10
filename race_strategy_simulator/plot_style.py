"""Shared chart colours and styling for reports and the app."""

from __future__ import annotations

# Chart chrome (light surface).
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical series colours, assigned in a fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]

# Colour follows the entity: a compound keeps its colour in every chart.
COMPOUND_COLORS = {
    "SOFT": SERIES[0],
    "MEDIUM": SERIES[1],
    "HARD": SERIES[2],
}

ACTUAL_COLOR = SERIES[0]
PREDICTED_COLOR = SERIES[1]
NEUTRAL_COLOR = "#898781"


def apply_matplotlib_style() -> None:
    """Recessive grid and axes, thin marks, one sans typeface."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "text.color": TEXT_PRIMARY,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "xtick.labelcolor": TEXT_SECONDARY,
            "ytick.labelcolor": TEXT_SECONDARY,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def plotly_layout(**overrides) -> dict:
    """Base Plotly layout: recessive grid, legend on top, unified hover."""
    layout = dict(
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(zeroline=False, gridwidth=1),
    )
    layout.update(overrides)
    return layout
