"""
Pareto analysis for latency-fidelity trade-off.
Visualizes and analyzes the Pareto frontier of explanation methods.
"""
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    method: str
    latency_ms: float
    fidelity: float
    config: Optional[Dict] = None
    
    @property
    def is_dominated(self) -> bool:
        """Check if this point is dominated (for internal use)."""
        return False


class ParetoAnalyzer:
    """
    Analyze Pareto frontier for latency-fidelity trade-off.
    """
    
    def __init__(self):
        self.points: List[ParetoPoint] = []
        self.pareto_frontier: List[ParetoPoint] = []
        
    def add_point(
        self,
        method: str,
        latency_ms: float,
        fidelity: float,
        config: Optional[Dict] = None
    ):
        """Add a data point."""
        self.points.append(ParetoPoint(method, latency_ms, fidelity, config))
        
    def add_points_from_df(
        self,
        df: pd.DataFrame,
        method_col: str = 'method',
        latency_col: str = 'p95_latency_ms',
        fidelity_col: str = 'pearson_r'
    ):
        """Add points from DataFrame."""
        for _, row in df.iterrows():
            self.add_point(
                method=row[method_col],
                latency_ms=row[latency_col],
                fidelity=row[fidelity_col]
            )
    
    def compute_pareto_frontier(
        self,
        minimize_latency: bool = True,
        maximize_fidelity: bool = True
    ) -> List[ParetoPoint]:
        """
        Compute Pareto frontier.
        
        For our case:
        - Minimize latency (lower is better)
        - Maximize fidelity (higher is better)
        """
        if not self.points:
            return []
        
        # Sort by latency
        sorted_points = sorted(self.points, key=lambda p: p.latency_ms)
        
        pareto = []
        max_fidelity_so_far = -np.inf
        
        for point in sorted_points:
            # A point is on Pareto frontier if its fidelity is higher
            # than all points with lower or equal latency
            if point.fidelity >= max_fidelity_so_far:
                pareto.append(point)
                max_fidelity_so_far = point.fidelity
        
        self.pareto_frontier = pareto
        return pareto
    
    def is_pareto_optimal(self, point: ParetoPoint) -> bool:
        """Check if a point is on the Pareto frontier."""
        if not self.pareto_frontier:
            self.compute_pareto_frontier()
        
        for p in self.pareto_frontier:
            if p.method == point.method and \
               abs(p.latency_ms - point.latency_ms) < 1e-6 and \
               abs(p.fidelity - point.fidelity) < 1e-6:
                return True
        return False
    
    def find_optimal_config(
        self,
        latency_constraint: Optional[float] = None,
        fidelity_constraint: Optional[float] = None
    ) -> Optional[ParetoPoint]:
        """
        Find optimal configuration given constraints.
        
        Args:
            latency_constraint: Maximum acceptable latency
            fidelity_constraint: Minimum acceptable fidelity
            
        Returns:
            Best Pareto point meeting constraints
        """
        if not self.pareto_frontier:
            self.compute_pareto_frontier()
        
        candidates = self.pareto_frontier
        
        if latency_constraint:
            candidates = [p for p in candidates if p.latency_ms <= latency_constraint]
        
        if fidelity_constraint:
            candidates = [p for p in candidates if p.fidelity >= fidelity_constraint]
        
        if not candidates:
            return None
        
        # Return point with best combined score
        # Normalize and combine
        latencies = np.array([p.latency_ms for p in candidates])
        fidelities = np.array([p.fidelity for p in candidates])
        
        norm_latency = (latencies - latencies.min()) / (latencies.max() - latencies.min() + 1e-10)
        norm_fidelity = (fidelities - fidelities.min()) / (fidelities.max() - fidelities.min() + 1e-10)
        
        # Score: high fidelity, low latency
        scores = norm_fidelity - 0.5 * norm_latency
        best_idx = np.argmax(scores)
        
        return candidates[best_idx]
    
    def plot_pareto_frontier(
        self,
        figsize: tuple = (12, 8),
        latency_threshold: float = 100,
        fidelity_threshold: float = 0.9,
        save_path: Optional[str] = None,
        title: str = "Latency-Fidelity Pareto Frontier"
    ):
        """
        Plot Pareto frontier with all points.
        
        Args:
            figsize: Figure size
            latency_threshold: Draw threshold line at this latency
            fidelity_threshold: Draw threshold line at this fidelity
            save_path: Path to save figure
            title: Plot title
        """
        if not self.pareto_frontier:
            self.compute_pareto_frontier()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all points
        for point in self.points:
            color = 'green' if self.is_pareto_optimal(point) else 'lightgray'
            size = 150 if self.is_pareto_optimal(point) else 50
            alpha = 1.0 if self.is_pareto_optimal(point) else 0.5
            
            ax.scatter(
                point.latency_ms,
                point.fidelity,
                c=color,
                s=size,
                alpha=alpha,
                edgecolors='black' if self.is_pareto_optimal(point) else 'gray',
                linewidths=2 if self.is_pareto_optimal(point) else 1,
                zorder=5 if self.is_pareto_optimal(point) else 1
            )
            
            # Label Pareto points
            if self.is_pareto_optimal(point):
                ax.annotate(
                    point.method,
                    (point.latency_ms, point.fidelity),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=10,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
                )
        
        # Draw Pareto frontier line
        if len(self.pareto_frontier) > 1:
            frontier_x = [p.latency_ms for p in self.pareto_frontier]
            frontier_y = [p.fidelity for p in self.pareto_frontier]
            ax.plot(frontier_x, frontier_y, 'g--', linewidth=2, alpha=0.5, label='Pareto Frontier')
        
        # Draw threshold lines
        ax.axvline(x=latency_threshold, color='r', linestyle='--', alpha=0.5, label=f'{latency_threshold}ms threshold')
        ax.axhline(y=fidelity_threshold, color='r', linestyle='--', alpha=0.5, label=f'{fidelity_threshold} fidelity threshold')
        
        # Shade acceptable region
        ax.fill_between(
            [0, latency_threshold],
            fidelity_threshold,
            1.0,
            alpha=0.1,
            color='green',
            label='Acceptable Region'
        )
        
        ax.set_xlabel('Latency (ms)', fontsize=12)
        ax.set_ylabel('Fidelity (Pearson r)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
        
        # Set limits
        ax.set_xlim(0, max(p.latency_ms for p in self.points) * 1.1)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved Pareto plot to {save_path}")
        
        return fig
    
    def plot_latency_fidelity_curve(
        self,
        figsize: tuple = (12, 5),
        save_path: Optional[str] = None
    ):
        """Plot latency-fidelity trade-off curves for different methods."""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Group by method
        methods = set(p.method for p in self.points)
        
        for method in methods:
            method_points = [p for p in self.points if p.method == method]
            latencies = [p.latency_ms for p in method_points]
            fidelities = [p.fidelity for p in method_points]
            
            axes[0].plot(latencies, fidelities, 'o-', label=method, linewidth=2, markersize=8)
        
        axes[0].set_xlabel('Latency (ms)')
        axes[0].set_ylabel('Fidelity')
        axes[0].set_title('Latency-Fidelity Trade-off by Method')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(x=50, color='r', linestyle='--', alpha=0.5)
        axes[0].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
        
        # Efficiency frontier
        if self.pareto_frontier:
            frontier_df = pd.DataFrame([
                {'latency': p.latency_ms, 'fidelity': p.fidelity}
                for p in self.pareto_frontier
            ])
            axes[1].plot(frontier_df['latency'], frontier_df['fidelity'], 'go-', linewidth=3, markersize=10)
            axes[1].fill_between(frontier_df['latency'], 0, frontier_df['fidelity'], alpha=0.2, color='green')
            axes[1].set_xlabel('Latency (ms)')
            axes[1].set_ylabel('Fidelity')
            axes[1].set_title('Pareto Efficiency Frontier')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved trade-off curve to {save_path}")
        
        return fig
    
    def generate_report(self) -> str:
        """Generate text report of Pareto analysis."""
        if not self.pareto_frontier:
            self.compute_pareto_frontier()
        
        lines = [
            "=" * 80,
            "PARETO FRONTIER ANALYSIS REPORT",
            "=" * 80,
            "",
            f"Total configurations evaluated: {len(self.points)}",
            f"Pareto-optimal configurations: {len(self.pareto_frontier)}",
            "",
            "Pareto Frontier (sorted by latency):",
            "-" * 60,
            f"{'Method':<20} {'Latency (ms)':<15} {'Fidelity':<15}",
            "-" * 60
        ]
        
        for point in self.pareto_frontier:
            lines.append(f"{point.method:<20} {point.latency_ms:<15.2f} {point.fidelity:<15.4f}")
        
        # Find optimal under constraints
        optimal = self.find_optimal_config(latency_constraint=50, fidelity_constraint=0.9)
        lines.extend([
            "",
            "Optimal Configuration (latency<50ms, fidelity>0.9):",
            "-" * 60
        ])
        
        if optimal:
            lines.append(f"Method: {optimal.method}")
            lines.append(f"Latency: {optimal.latency_ms:.2f} ms")
            lines.append(f"Fidelity: {optimal.fidelity:.4f}")
        else:
            lines.append("No configuration meets both constraints!")
        
        lines.append("")
        
        return "\n".join(lines)
    
    def get_pareto_df(self) -> pd.DataFrame:
        """Get Pareto frontier as DataFrame."""
        if not self.pareto_frontier:
            self.compute_pareto_frontier()
        
        return pd.DataFrame([
            {
                'method': p.method,
                'latency_ms': p.latency_ms,
                'fidelity': p.fidelity,
                **(p.config or {})
            }
            for p in self.pareto_frontier
        ])


def compute_hypervolume(
    points: List[ParetoPoint],
    reference_point: Tuple[float, float] = (200, 0.0)
) -> float:
    """
    Compute hypervolume indicator for Pareto frontier quality.
    
    Args:
        points: Pareto frontier points
        reference_point: Reference point (latency, fidelity) for hypervolume
        
    Returns:
        Hypervolume value
    """
    if not points:
        return 0.0
    
    # Sort by latency
    sorted_points = sorted(points, key=lambda p: p.latency_ms)
    
    hypervolume = 0.0
    prev_latency = 0
    
    for point in sorted_points:
        # Rectangle width: difference in latency
        width = point.latency_ms - prev_latency
        # Rectangle height: fidelity - reference_fidelity
        height = point.fidelity - reference_point[1]
        
        if height > 0:
            hypervolume += width * height
        
        prev_latency = point.latency_ms
    
    return hypervolume


def compare_pareto_frontiers(
    frontier_a: List[ParetoPoint],
    frontier_b: List[ParetoPoint],
    reference_point: Tuple[float, float] = (200, 0.0)
) -> Dict:
    """
    Compare two Pareto frontiers using hypervolume indicator.
    
    Returns:
        Dict with comparison metrics
    """
    hv_a = compute_hypervolume(frontier_a, reference_point)
    hv_b = compute_hypervolume(frontier_b, reference_point)
    
    return {
        'hypervolume_a': hv_a,
        'hypervolume_b': hv_b,
        'ratio': hv_a / hv_b if hv_b > 0 else float('inf'),
        'better_frontier': 'A' if hv_a > hv_b else 'B' if hv_b > hv_a else 'Equal'
    }
