import numpy as np
from typing import List, Dict, Any
from Internal.SimulationEngine import SimulationResult
from Internal.Config import SimulationConfig


class EvaluationEngine:
    """
    Ranks configurations based on user effort targets, equilibrium points,
    and physical opening constraints. Weights are normalized automatically.
    """

    def __init__(self,
                 target_close_force_n: float = 20.0,
                 target_open_force_n: float = 20.0,
                 target_equilibrium_deg: float = 40.0,
                 target_max_angle_deg: float = 100.0,
                 target_max_hinge_force_n: float = 1000.0):
        """
        :param target_close_force_n: Desired force to pull the door shut.
        :param target_open_force_n: Desired force to lift the door.
        :param target_equilibrium_deg: Angle where the door becomes self-supporting.
        :param target_max_angle_deg: The ideal full-open angle for the hatchback.
        :param target_max_hinge_force_n: The maximum allowable stress on the hinge hardware.
        """
        self.target_close_force = target_close_force_n
        self.target_open_force = target_open_force_n
        self.target_equilibrium_deg = target_equilibrium_deg
        self.target_max_angle = target_max_angle_deg
        self.target_max_hinge_force = target_max_hinge_force_n

        # Define raw weights (can be any positive numbers)
        self.raw_weights = {
            "hinge_stress": 1,
            "close_accuracy": 9,
            "open_accuracy": 9,
            "equilibrium_accuracy": 0,
            "angle_accuracy": 6
        }

        self.weights = self._normalize_weights(self.raw_weights)

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Ensures the sum of weights equals 1.0 for score consistency."""
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0 / len(weights) for k in weights.keys()}
        return {k: v / total for k, v in weights.items()}

    def _get_user_force_at_handle(self, net_torque: float, door_length: float) -> float:
        return net_torque / door_length

    def calculate_metrics(self, result: SimulationResult, cfg: SimulationConfig) -> Dict[str, float]:
        # Forces at the handle (Closed vs Fully Opened)
        actual_close_force = abs(self._get_user_force_at_handle(result.net_torques[0], cfg.door_length))
        actual_open_force = abs(self._get_user_force_at_handle(result.net_torques[-1], cfg.door_length))

        max_hinge_force = max(result.hinge_forces) if result.hinge_forces else 0.0
        eq_angle = result.equilibrium_angle if result.equilibrium_angle is not None else 0.0
        max_angle = result.max_physical_angle

        return {
            "actual_close_force": actual_close_force,
            "actual_open_force": actual_open_force,
            "max_hinge_force": max_hinge_force,
            "equilibrium_angle": eq_angle,
            "max_physical_angle": max_angle
        }

    def _calculate_pct_score(self, actual: float, target: float) -> float:
        """Calculates a score from 0.0 to 1.0 based on percentage deviation."""
        if target == 0:
            return 1.0 if actual == 0 else 0.0

        # Calculate percentage error (e.g., 0.1 for 10% off)
        error = abs(actual - target) / target
        # Score is 1 minus the error, clipped at 0
        return max(0.0, 1.0 - error)

    def score_solution(self, metrics: Dict[str, float]) -> float:
        # Score components based on percentage distance from target
        s_close = self._calculate_pct_score(metrics["actual_close_force"], self.target_close_force)
        s_open = self._calculate_pct_score(metrics["actual_open_force"], self.target_open_force)
        s_eq = self._calculate_pct_score(metrics["equilibrium_angle"], self.target_equilibrium_deg)
        s_angle = self._calculate_pct_score(metrics["max_physical_angle"], self.target_max_angle)

        # For hinge stress, we penalize only if it EXCEEDS the target
        if metrics["max_hinge_force"] <= self.target_max_hinge_force:
            s_hinge = 1.0
        else:
            s_hinge = self._calculate_pct_score(metrics["max_hinge_force"], self.target_max_hinge_force)

        # Final weighted sum
        return (s_close * self.weights["close_accuracy"] +
                s_open * self.weights["open_accuracy"] +
                s_hinge * self.weights["hinge_stress"] +
                s_eq * self.weights["equilibrium_accuracy"] +
                s_angle * self.weights["angle_accuracy"])

    def evaluate_all(self,
                     solutions: List[Dict[str, Any]],
                     base_cfg: SimulationConfig) -> List[Dict[str, Any]]:
        for sol in solutions:
            metrics = self.calculate_metrics(sol['result'], base_cfg)
            sol['score'] = self.score_solution(metrics)
            sol['metrics'] = metrics

            sol['summary'] = (f"Score: {sol['score']:.2f} | "
                              f"Max: {metrics['max_physical_angle']:.1f}° | "
                              f"Eq: {metrics['equilibrium_angle']:.1f}° | "
                              f"F_cls: {metrics['actual_close_force']:.1f}N | "
                              f"F_opn: {metrics['actual_open_force']:.1f}N")

        return sorted(solutions, key=lambda x: x['score'], reverse=True)