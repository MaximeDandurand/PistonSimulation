import numpy as np
from typing import List, Dict, Any
from matplotlib.path import Path
from SimulationEngine import HatchbackPhysicsEngine, SimulationResult
from Config import SimulationConfig


class PistonSpec:
    def __init__(self, name: str, max_length: float, stroke: float, f_ext: float, f_comp: float):
        self.name = name
        self.max_length = max_length
        self.stroke = stroke
        self.f_ext = f_ext
        self.f_comp = f_comp


class MountingArea:
    def __init__(self, vertices: List[np.ndarray]):
        self.path = Path(vertices)
        v_np = np.array(vertices)
        self.x_min, self.y_min = v_np.min(axis=0)
        self.x_max, self.y_max = v_np.max(axis=0)

    def get_valid_points(self, resolution: float) -> List[np.ndarray]:
        x_range = np.arange(self.x_min, self.x_max, resolution)
        y_range = np.arange(self.y_min, self.y_max, resolution)

        valid_points: List[np.ndarray] = []
        for x in x_range:
            for y in y_range:
                point = np.array([x, y])
                if self.path.contains_point(point):
                    valid_points.append(point)
        return valid_points


def run_grid_search(
        base_cfg: SimulationConfig,
        chassis_poly: MountingArea,
        door_poly: MountingArea,
        pistons: List[PistonSpec],
        resolution: float = 0.02,
        min_angle_threshold: float = 60.0,
        max_angle_threshold: float = 140.0,
) -> List[Dict[str, Any]]:
    """
    Performs an optimized grid search for valid hatchback gas spring mounting points.
    """
    engine = HatchbackPhysicsEngine()
    valid_solutions: List[Dict[str, Any]] = []

    # Calculate valid position
    chassis_candidates = chassis_poly.get_valid_points(resolution)
    door_candidates = door_poly.get_valid_points(resolution)
    rad_closed = np.radians(base_cfg.door_close_angle_deg)

    # Search
    for piston in pistons:
        for c_anchor in chassis_candidates:
            cx, cy = c_anchor

            for d_mount_local in door_candidates:
                d_mount_closed = engine.get_rotated_point(d_mount_local, rad_closed)

                # Making sure that the piston push the door to close at the min angle
                if (cx * d_mount_closed[1] - cy * d_mount_closed[0]) >= 0:
                    continue
                base_cfg.chassis_piston_anchor = c_anchor
                base_cfg.piston_mount_on_door = d_mount_local
                base_cfg.strut_max_length = piston.max_length
                base_cfg.strut_min_length = piston.max_length - piston.stroke
                base_cfg.f_ext = piston.f_ext
                base_cfg.f_comp = piston.f_comp

                result: SimulationResult = engine.run(base_cfg)

                # 3. Hard Constraint Validation
                if(result.is_valid_mask.__contains__(False)):
                    continue
                if result.max_physical_angle < min_angle_threshold:
                    continue

                if result.max_physical_angle > max_angle_threshold:
                    continue

                if result.net_torques[0] > 0:
                    continue

                if result.net_torques[-1] < 0:
                    continue
                print(f"Piston={piston.name}, "
                      f"Chassis={c_anchor}, Door={d_mount_local}, "
                      f"Initial Torque={result.net_torques[0]:.2f} Nm")
                valid_solutions.append({
                    'piston': piston,
                    'chassis_mount': c_anchor,
                    'door_mount': d_mount_local,
                    'result': result
                })

    return valid_solutions