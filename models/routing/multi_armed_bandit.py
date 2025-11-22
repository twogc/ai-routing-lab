"""Multi-Armed Bandit - Online route selection with exploration/exploitation."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class MABRouteSelection:
    """MAB route selection"""

    selected_route: int
    ucb_scores: Dict[int, float]
    arm_statistics: Dict[int, Dict[str, float]]


class MultiArmedBanditRouter:
    """UCB-based multi-armed bandit for online route selection."""

    def __init__(
        self, n_routes: int = 5, epsilon: float = 0.1, logger: Optional[logging.Logger] = None
    ):
        self.n_routes = n_routes
        self.epsilon = epsilon
        self.logger = logger or logging.getLogger(__name__)

        self.rewards = [0.0] * n_routes
        self.counts = [0] * n_routes
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiArmedBanditRouter":
        """Initialize rewards from training data"""
        # Initialize with empirical rewards
        unique_routes = np.unique(y)
        for route in unique_routes:
            mask = y == route
            self.rewards[int(route)] = np.mean(X[mask].sum(axis=1)) if np.any(mask) else 0.0
            self.counts[int(route)] = np.sum(mask)

        self.fitted = True
        self.logger.info(f"MAB Router initialized with {self.n_routes} arms")
        return self

    def select_route(self, state: Optional[np.ndarray] = None) -> Tuple[int, Dict[int, float]]:
        """Select route using UCB"""
        if not self.fitted:
            # Random selection before fit
            return np.random.randint(0, self.n_routes), {}

        ucb_scores = self._calculate_ucb()

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            selected = np.random.randint(0, self.n_routes)
        else:
            selected = np.argmax(ucb_scores)

        return selected, ucb_scores

    def update_reward(self, route: int, reward: float):
        """Update reward for selected route"""
        self.counts[route] += 1
        self.rewards[route] += (reward - self.rewards[route]) / self.counts[route]

    def predict_sample(self, x: np.ndarray) -> MABRouteSelection:
        """Predict with MAB"""
        selected, ucb_scores = self.select_route(x)

        arm_stats = {
            i: {
                "mean_reward": self.rewards[i],
                "count": self.counts[i],
                "ucb_score": ucb_scores.get(i, 0.0),
            }
            for i in range(self.n_routes)
        }

        return MABRouteSelection(
            selected_route=selected, ucb_scores=ucb_scores, arm_statistics=arm_stats
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "n_routes": self.n_routes,
            "total_pulls": sum(self.counts),
            "mean_rewards": self.rewards.copy(),
            "arm_counts": self.counts.copy(),
        }

    def _calculate_ucb(self) -> np.ndarray:
        """Calculate UCB scores"""
        ucb = np.zeros(self.n_routes)
        total = sum(self.counts)

        for i in range(self.n_routes):
            if self.counts[i] == 0:
                ucb[i] = float("inf")
            else:
                exploitation = self.rewards[i]
                exploration = np.sqrt(2 * np.log(total + 1) / self.counts[i])
                ucb[i] = exploitation + exploration

        return ucb
