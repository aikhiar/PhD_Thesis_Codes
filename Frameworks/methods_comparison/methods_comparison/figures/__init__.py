from .style import apply_thesis_style, save_figure, get_method_styles, PALETTE
from .plots import (
    plot_comparison_block,
    plot_convergence_curves,
    plot_pareto_aggregated,
    plot_rank_heatmap,
    plot_performance_profiles,
    plot_critical_difference,
    plot_significance_heatmap,
    plot_quality_vs_time,
)

__all__ = [
    "apply_thesis_style", "save_figure", "get_method_styles", "PALETTE",
    "plot_comparison_block", "plot_convergence_curves", "plot_pareto_aggregated",
    "plot_rank_heatmap", "plot_performance_profiles", "plot_critical_difference",
    "plot_significance_heatmap", "plot_quality_vs_time",
]
