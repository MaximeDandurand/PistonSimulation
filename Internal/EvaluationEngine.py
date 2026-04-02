import numpy as np
from typing import List, Dict, Any
from Internal.SimulationEngine import SimulationResult
from Internal.Config import SimulationConfig


class EvaluationEngine:
    def __init__(self,
                 target_close_force_n: float = 20.0,
                 target_open_force_n: float = 20.0,
                 target_equilibrium_deg: float = 40.0,
                 target_max_hinge_force_n: float = 1000.0,
                 person_height: float = 1.75):

        self.target_close_force = target_close_force_n
        self.target_open_force = target_open_force_n
        self.target_equilibrium_deg = target_equilibrium_deg
        self.target_max_hinge_force = target_max_hinge_force_n
        self.person_height = person_height

        self.max_reach_height = person_height + 0.4

        # Weights categorized by importance
        self.raw_weights = {
            "hinge_stress": 1,
            "close_accuracy": 7,
            "open_accuracy": 7,
            "equilibrium_accuracy": 0,
            "tip_clearance": 10,
            "mid_clearance": 15
        }

        self.weights = self._normalize_weights(self.raw_weights)

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0 / len(weights) for k in weights.keys()}
        return {k: v / total for k, v in weights.items()}

    def _get_user_force_at_handle(self, net_torque: float, door_length: float) -> float:
        return net_torque / door_length

    def calculate_metrics(self, result: SimulationResult, cfg: SimulationConfig) -> Dict[str, float]:
        actual_close_force = abs(self._get_user_force_at_handle(result.net_torques[0], cfg.door_length))
        actual_open_force = abs(self._get_user_force_at_handle(result.net_torques[-1], cfg.door_length))

        closed_rad = np.radians(cfg.door_close_angle_deg)
        hinge_height_above_ground = cfg.ground_height_offset + (cfg.door_length * np.cos(closed_rad))

        max_rad = np.radians(result.max_physical_angle)

        tip_height = hinge_height_above_ground - (cfg.door_length * np.cos(max_rad))

        mid_y_linear = -(cfg.door_length / 2.0) * np.cos(max_rad)
        mid_y_curve = -(cfg.middle_door_curve_offset * np.sin(max_rad))

        mid_height = hinge_height_above_ground + mid_y_linear + mid_y_curve

        return {
            "actual_close_force": actual_close_force,
            "actual_open_force": actual_open_force,
            "max_hinge_force": max(result.hinge_forces) if result.hinge_forces else 0.0,
            "equilibrium_angle": result.equilibrium_angle if result.equilibrium_angle is not None else 0.0,
            "tip_height": tip_height,
            "mid_height": mid_height,
            "max_angle": result.max_physical_angle
        }

    def score_solution(self, metrics: Dict[str, float]) -> float:
        #Force to close or open
        s_close = self._calculate_pct_score(metrics["actual_close_force"], self.target_close_force)
        s_open = self._calculate_pct_score(metrics["actual_open_force"], self.target_open_force)
        s_eq = self._calculate_pct_score(metrics["equilibrium_angle"], self.target_equilibrium_deg)

        # --- Middle Section Clearance (Standing Room) ---
        mid_penalty = 0.0
        if metrics["mid_height"] < self.person_height:
            mid_penalty = (self.person_height - metrics["mid_height"]) * 6.0
        s_mid = max(0.0, 1.0 - mid_penalty)

        # --- Tip Clearance and Reachability ---
        tip_penalty = 0.0
        # Penalty for being too low
        if metrics["tip_height"] < self.person_height:
            tip_penalty += (self.person_height - metrics["tip_height"]) * 2.0
        # Penalty for being too high to reach
        if metrics["tip_height"] > self.max_reach_height:
            tip_penalty += (metrics["tip_height"] - self.max_reach_height) * 3.0
        s_tip = max(0.0, 1.0 - tip_penalty)

        # Hinge Stress
        if metrics["max_hinge_force"] <= self.target_max_hinge_force:
            s_hinge = 1.0
        else:
            s_hinge = self._calculate_pct_score(metrics["max_hinge_force"], self.target_max_hinge_force)

        # Final Weighted Calculation
        final_score = (s_close * self.weights["close_accuracy"] +
                       s_open * self.weights["open_accuracy"] +
                       s_hinge * self.weights["hinge_stress"] +
                       s_eq * self.weights["equilibrium_accuracy"] +
                       s_mid * self.weights["mid_clearance"] +
                       s_tip * self.weights["tip_clearance"])

        return final_score

    def _calculate_pct_score(self, actual: float, target: float) -> float:
        if target == 0: return 1.0 if actual == 0 else 0.0
        error = abs(actual - target) / target
        return max(0.0, 1.0 - error)

    def evaluate_all(self, solutions: List[Dict[str, Any]], base_cfg: SimulationConfig) -> List[Dict[str, Any]]:
        for sol in solutions:
            metrics = self.calculate_metrics(sol['result'], base_cfg)
            sol['score'] = self.score_solution(metrics)
            sol['metrics'] = metrics
            sol['summary'] = (f"Score: {sol['score']:.2f} | "
                              f"Tip H: {metrics['tip_height']:.2f}m | "
                              f"Mid H: {metrics['mid_height']:.2f}m | "
                              f"F_cls: {metrics['actual_close_force']:.1f}N")
        return sorted(solutions, key=lambda x: x['score'], reverse=True)