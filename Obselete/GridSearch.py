import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from Internal.Config import SimulationConfig, SimulationConstraint
from Internal.Optimizer import PistonSpec, MountingArea, run_grid_search
from Internal.EvaluationEngine import EvaluationEngine
from Internal.ExcelCalculator import get_top_n_pistons
@dataclass
class OptimizationConfig:
    """Encapsulates all parameters for the hybrid optimization process."""
    grid_resolution: float = 0.04
    top_n_grid: int = 5
    piston_catalog_size: int = 10
    max_gradient_steps: int = 100
    # Mounting zones defined as lists of (x, y) tuples
    chassis_zone_coords: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-0.05, -0.05), (-0.05, -0.6), (0.05, -0.6), (0.05, -0.05)
    ])
    door_zone_coords: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.0, 0.1), (0.5, 0.1), (0.5, -0.1), (0.0, -0.1)
    ])
    max_iteration = 3

def run_grid_simulation_workflow(opt_cfg: OptimizationConfig) -> None:

    base_cfg = SimulationConfig()

    # --- 1. Geometry Setup ---
    # Rotate chassis coordinates based on the door's closed angle
    angle_rad = math.radians(base_cfg.door_close_angle_deg)
    rotated_chassis = []
    for x, y in opt_cfg.chassis_zone_coords:
        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        rotated_chassis.append((new_x, new_y))

    chassis_zone = MountingArea(rotated_chassis)
    door_zone = MountingArea(opt_cfg.door_zone_coords)

    # piston_catalog = [
    #     PistonSpec("HeavyDuty-500N", max_length=0.6680, stroke=0.2791, f_ext=355, f_comp=462)
    # ]
    piston_catalog = get_top_n_pistons(base_cfg, 1)

    for piston in piston_catalog:
        piston.print()
    # Run grid search
    print(f"Searching for valid mounting configurations...")
    valid_solutions = run_grid_search(
        base_cfg=base_cfg,
        chassis_poly=chassis_zone,
        door_poly=door_zone,
        pistons=piston_catalog,
        resolution=0.01,
        simulation_constraints= SimulationConstraint(),
        show_metrics=True,
    )

    if not valid_solutions:
        print("No valid configurations found.")
        return

    # Evaluate
    evaluator = EvaluationEngine(target_close_force_n=20.0, target_open_force_n=20.0)
    ranked_results = evaluator.evaluate_all(valid_solutions, base_cfg)

    # Output the Results
    print(f"\nFound {len(ranked_results)} valid solutions.")

    print("\nTop Candidate overall:")
    best = ranked_results[0]
    best_piston = best['piston']
    c_mount = best['chassis_mount']
    d_mount = best['door_mount']

    print(f"Piston: {best_piston.name}")
    print(f"Chassis Mount: {c_mount}")
    print(f"Door Mount:    {d_mount}")
    print(f"Score: {best['score']:.2f}")
    print(best['summary'])

    # --- Print Full Config (VisualizeOptimizer Style) ---
    print("\n" + "-" * 40)
    print("COPY THIS TO SimulationGraph.py:")
    print(
        f"SimulationConfig("
        f"chassis_piston_anchor=np.array([{c_mount[0]:.6f}, {c_mount[1]:.6f}]), "
        f"piston_mount_on_door=np.array([{d_mount[0]:.6f}, {d_mount[1]:.6f}]), "
        f"center_of_mass_on_door=np.array([{base_cfg.center_of_mass_on_door[0]:.6f}, {base_cfg.center_of_mass_on_door[1]:.6f}]), "
        f"door_length={base_cfg.door_length}, "
        f"door_mass_kg={base_cfg.door_mass_kg}, "
        f"strut_max_length={best_piston.max_length}, "
        f"strut_stroke={best_piston.stroke}, "
        f"extension_force_n={best_piston.f_ext}, "
        f"compression_force_n={best_piston.f_comp}, "
        f"door_close_angle_deg={base_cfg.door_close_angle_deg})"
    )
    print("-" * 40)


if __name__ == "__main__":
     run_grid_simulation_workflow(OptimizationConfig())
