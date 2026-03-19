import time

import numpy as np
from typing import List, Dict, Any
from Internal.SimulationEngine import HatchbackPhysicsEngine, SimulationResult
from Internal.Config import SimulationConfig, SimulationConstraint, MountingArea, PistonSpec




def run_grid_search(
        base_cfg: SimulationConfig,
        chassis_poly: MountingArea,
        door_poly: MountingArea,
        pistons: List[PistonSpec],
        resolution: float = 0.02,
        simulation_constraints: SimulationConstraint=None,
        show_metrics: bool = False
) -> List[Dict[str, Any]]:
    """
    Performs an optimized grid search for valid hatchback gas spring mounting points.
    """
    start_time = time.perf_counter()
    engine = HatchbackPhysicsEngine()
    valid_solutions: List[Dict[str, Any]] = []

    # Calculate valid position
    cell_start_time = time.perf_counter()
    chassis_candidates = chassis_poly.get_valid_points(resolution)
    door_candidates = door_poly.get_valid_points(resolution)
    rad_closed = np.radians(base_cfg.door_close_angle_deg)
    cell_end_time = time.perf_counter()
    cell_duration = cell_end_time - cell_start_time

    # Search
    for piston in pistons:
        for c_anchor in chassis_candidates:
            cx, cy = c_anchor

            for d_mount_local in door_candidates:
                d_mount_closed = engine.get_rotated_point(d_mount_local, rad_closed)

                # Making sure that the piston push the door to close at the min angle
                # if (cx * d_mount_closed[1] - cy * d_mount_closed[0]) >= 0:
                #     continue
                if c_anchor[1] > d_mount_closed[1]:
                    continue
                base_cfg.chassis_piston_anchor_meter = c_anchor
                base_cfg.piston_mount_on_door_meter = d_mount_local
                base_cfg.strut_max_length = piston.max_length
                base_cfg.strut_min_length = piston.max_length - piston.stroke
                base_cfg.f_ext = piston.f_ext
                base_cfg.f_comp = piston.f_comp

                result: SimulationResult = engine.run(cfg=base_cfg, constraints=simulation_constraints, steps=10)

                if not result.simulation_finished:
                    continue

                # print(f"Piston={piston.name}, "
                #       f"Chassis={c_anchor}, Door={d_mount_local}, "
                #       f"Initial Torque={result.net_torques[0]:.2f} Nm")
                valid_solutions.append({
                    'piston': piston,
                    'chassis_mount': c_anchor,
                    'door_mount': d_mount_local,
                    'result': result
                })
    if show_metrics:
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"\n--- Grid Search Metrics ---")
        print(f"Total Execution Time: {duration:.4f} seconds")
        print(f"    Find Cells Time: {cell_duration} seconds")
        print(f"    Simulation Time: {duration-cell_duration} seconds")
        print(f"Total Cell: {len(chassis_candidates)*len(door_candidates)}")
        print(f"Solutions Found: {len(valid_solutions)}")
        print(f"---------------------------\n")
    return valid_solutions
