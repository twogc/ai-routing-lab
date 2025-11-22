"""Q-Learning Route Optimizer - Reinforcement learning based routing."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class QLearningDecision:
    """Q-Learning decision"""

    selected_route: int
    q_values: Dict[int, float]
    expected_reward: float


class QLearningRouter:
    """Q-Learning for adaptive route optimization."""

    def __init__(
        self,
        n_routes: int = 5,
        n_states: int = 10,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        logger: Optional[logging.Logger] = None,
    ):
        self.n_routes = n_routes
        self.n_states = n_states
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.logger = logger or logging.getLogger(__name__)

        # Q-table: states x routes
        self.q_table = np.zeros((n_states, n_routes))
        self.fitted = False
        self.total_updates = 0

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> "QLearningRouter":
        """Train Q-Learning on historical data"""
        X = X.astype(np.float32)
        y = y.astype(int)

        for epoch in range(epochs):
            total_reward = 0

            for i in range(len(X)):
                state = self._discretize_state(X[i])
                action = y[i]
                reward = self._get_reward(X[i], action)

                next_state = self._discretize_state(X[i] if i + 1 >= len(X) else X[i + 1])
                max_next_q = np.max(self.q_table[next_state])

                old_value = self.q_table[state, action]
                new_value = old_value + self.learning_rate * (
                    reward + self.discount_factor * max_next_q - old_value
                )
                self.q_table[state, action] = new_value
                self.total_updates += 1
                total_reward += reward

            if epoch % (epochs // 2) == 0:
                self.logger.debug(f"Epoch {epoch}, Avg Reward: {total_reward / len(X):.4f}")

        self.fitted = True
        self.logger.info(f"Q-Learning trained with {self.total_updates} updates")
        return self

    def select_route(self, x: np.ndarray, explore: bool = False) -> QLearningDecision:
        """Select route using Q-values"""
        if not self.fitted:
            return QLearningDecision(
                selected_route=0,
                q_values=dict.fromkeys(range(self.n_routes), 0.0),
                expected_reward=0.0,
            )

        state = self._discretize_state(x)
        q_values = self.q_table[state]

        if explore and np.random.random() < 0.1:
            selected_route = np.random.randint(0, self.n_routes)
        else:
            selected_route = np.argmax(q_values)

        return QLearningDecision(
            selected_route=selected_route,
            q_values={i: float(q_values[i]) for i in range(self.n_routes)},
            expected_reward=float(np.max(q_values)),
        )

    def update(self, x: np.ndarray, action: int, reward: float):
        """Update Q-table with experience"""
        state = self._discretize_state(x)
        old_value = self.q_table[state, action]
        new_value = old_value + self.learning_rate * (
            reward + self.discount_factor * np.max(self.q_table[state]) - old_value
        )
        self.q_table[state, action] = new_value
        self.total_updates += 1

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "n_routes": self.n_routes,
            "n_states": self.n_states,
            "total_updates": self.total_updates,
            "learning_rate": self.learning_rate,
            "mean_q_value": float(np.mean(self.q_table)),
        }

    def _discretize_state(self, x: np.ndarray) -> int:
        """Convert continuous state to discrete"""
        # Simple discretization by norm
        norm = np.linalg.norm(x)
        state_idx = int((norm % 1.0) * self.n_states)
        return min(state_idx, self.n_states - 1)

    @staticmethod
    def _get_reward(x: np.ndarray, action: int) -> float:
        """Calculate reward for state-action pair"""
        # Reward based on feature values (simplified)
        return float(np.sum(x) / (len(x) + 1))
