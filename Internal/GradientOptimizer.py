import numpy as np
from typing import List, Dict, Any, Tuple
from matplotlib.path import Path
from Internal.SimulationEngine import HatchbackPhysicsEngine, SimulationResult
from Internal.Config import SimulationConfig
from Internal.EvaluationEngine import EvaluationEngine


class PistonSpec:
    def __init__(self, name: str, max_length: float, stroke: float, f_ext: float, f_comp: float):
        self.name = name
        self.max_length = max_length
        self.stroke = stroke
        self.f_ext = f_ext
        self.f_comp = f_comp


class MountingArea:
    def __init__(self, vertices: List[Tuple[float, float]]):
        self.path = Path(vertices)
        v_np = np.array(vertices)
        self.x_min, self.y_min = v_np.min(axis=0)
        self.x_max, self.y_max = v_np.max(axis=0)
        self.vertices = v_np

    def contains(self, point: np.ndarray) -> bool:
        return self.path.contains_point(point)


class DiscreteGradientOptimizer:
    def __init__(self, base_cfg: SimulationConfig, evaluator: EvaluationEngine):
        self.base_cfg = base_cfg
        self.evaluator = evaluator
        self.engine = HatchbackPhysicsEngine()
        self.weights = {
            "net_torques": 10.,
            "max_physical_angle": 15.,
            "invalid_count": 5.
        }
    def get_score(self, params: np.ndarray, piston: Any) -> float:
        cx, cy, dx, dy = params
        self.base_cfg.chassis_piston_anchor = np.array([cx, cy])
        self.base_cfg.piston_mount_on_door = np.array([dx, dy])
        self.base_cfg.strut_max_length = piston.max_length
        self.base_cfg.strut_min_length = piston.max_length - piston.stroke
        self.base_cfg.f_ext = piston.f_ext
        self.base_cfg.f_comp = piston.f_comp

        result = self.engine.run(self.base_cfg)

        # 1. Start with a neutral penalty
        penalty = 0.0
        is_invalid = False

        # 2. Piston Stroke Penalty: How many simulation steps were physically impossible?
        # result.is_valid_mask is a list of Booleans.
        invalid_count = result.is_valid_mask.count(False)
        if invalid_count > 0:
            penalty -= (invalid_count * self.weights["invalid_count"])  # Heavy penalty per invalid step
            is_invalid = True

        # 3. Closing Torque Penalty: Door must want to stay closed at 0 degrees
        # (Net torque at index 0 should be negative)
        if result.net_torques[0] > 0:
            penalty -= (result.net_torques[0] * self.weights["net_torques"])
            is_invalid = True

        # 4. Opening Torque Penalty: Piston must hold door up at the top
        # (Net torque at the last index should be positive)
        if result.net_torques[-1] < 0:
            penalty -= (abs(result.net_torques[-1]) * self.weights["net_torques"])
            is_invalid = True

        # 5. Angle Range Penalty: Door must open between 50 and 140 degrees
        if result.max_physical_angle < 50.0:
            penalty -= (50.0 - result.max_physical_angle) * self.weights["max_physical_angle"]
            is_invalid = True
        elif result.max_physical_angle > 140.0:
            penalty -= (result.max_physical_angle - 140.0) * self.weights["max_physical_angle"]
            is_invalid = True

        # If any of the above failed, return only the penalty (negative score)
        if is_invalid:
            # We return penalty - 1.0 to ensure it's always below any valid score
            return penalty - 1.0

        # If it passed all physical checks, return the actual performance score (positive)
        metrics = self.evaluator.calculate_metrics(result, self.base_cfg)
        return self.evaluator.score_solution(metrics)

    def run_discrete_search(
            self,
            piston: Any,
            chassis_poly: MountingArea,
            door_poly: MountingArea,
            start_pos: np.ndarray,
            resolution: float = 0.001,
            max_steps: int = 100
    ) -> Dict[str, Any]:
        current_params = np.round(start_pos / resolution) * resolution
        current_score = self.get_score(current_params, piston)

        # Track the path for visualization
        history = [current_params.copy()]

        for _ in range(max_steps):
            best_neighbor = current_params.copy()
            best_neighbor_score = current_score

            for i in range(4):
                for direction in [-1, 1]:
                    neighbor = current_params.copy()
                    neighbor[i] += direction * resolution

                    if not chassis_poly.path.contains_point(neighbor[0:2]) or \
                            not door_poly.path.contains_point(neighbor[2:4]):
                        continue

                    score = self.get_score(neighbor, piston)
                    if score > best_neighbor_score:
                        best_neighbor_score = score
                        best_neighbor = neighbor

            if np.array_equal(best_neighbor, current_params):
                break

            current_params = best_neighbor
            current_score = best_neighbor_score
            history.append(current_params.copy())

        return {
            'params': current_params,
            'score': current_score,
            'history': np.array(history)
        }