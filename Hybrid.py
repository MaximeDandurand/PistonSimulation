import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple
from Internal.Config import SimulationConfig, SimulationConstraint
from Internal.Optimizer import MountingArea, run_grid_search
from Internal.GradientOptimizer import DiscreteGradientOptimizer
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
        (0.5109729125976563, -1.0940095209813385),
        (-0.12770208740234376, -1.0940095209813385),
        (0.03530183681109429, -0.08038170002264337),
        (0.19289287463076574, -0.136317052224804),
        (0.32657148998097174, -0.22276893314767585),
        (0.42751932675598014, -0.3329369159373982),
        (0.48992729274340413, -0.45872352571806),
        (0.5109729125976562, -0.5920095209813384)
    ])
    door_zone_coords: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0.40878223478677705, -0.20650860812068528),
        (0.01470208740234375, -0.20650860812068528),
        (0.01470208740234375, -0.0107567505372308),
        (0.15068555513542542, -0.05248641835595799),
        (0.2911477514508052, -0.1222805342890506),
        (0.40878223478677705, -0.21650860812068526)
    ])
    max_iteration = 3

def run_hybrid_optimization(opt_cfg: OptimizationConfig, iteration=1):
    base_cfg = SimulationConfig()
    evaluator = EvaluationEngine(target_close_force_n=20.0, target_open_force_n=20.0)
    optimizer = DiscreteGradientOptimizer(base_cfg, evaluator)

    # --- 1. Geometry Setup ---
    # Rotate chassis coordinates based on the door's closed angle

    chassis_zone = MountingArea(opt_cfg.chassis_zone_coords)
    door_zone = MountingArea(transform_door_zone(opt_cfg.door_zone_coords, base_cfg.door_close_angle_deg))
    print(door_zone)
    # --- 2. Phase 1: Grid Search ---
    print(f"PHASE 1: Running Wide Grid Search (Resolution: {opt_cfg.grid_resolution})...")
    piston_catalog = get_top_n_pistons(base_cfg, opt_cfg.piston_catalog_size)

    all_valid_grid_solutions = run_grid_search(
        base_cfg=base_cfg,
        chassis_poly=chassis_zone,
        door_poly=door_zone,
        pistons=piston_catalog,
        resolution=opt_cfg.grid_resolution,
        simulation_constraints=SimulationConstraint(),
        show_metrics=True
    )

    if not all_valid_grid_solutions:
        if iteration>opt_cfg.max_iteration:
            print(f"No valid configurations found in Phase 1 after {iteration} iterations.")
            return
        print("No valid configurations found in Phase 1.")
        print("Running a more precise grid search...")
        opt_cfg.grid_resolution = opt_cfg.grid_resolution / 1.5
        run_hybrid_optimization(opt_cfg, iteration+1)
        return

    # Rank grid results to find the best starting points
    ranked_grid_results = evaluator.evaluate_all(all_valid_grid_solutions, base_cfg)
    top_seeds = ranked_grid_results[:opt_cfg.top_n_grid]

    print(f"Found {len(ranked_grid_results)} grid solutions. Refining top {len(top_seeds)} seeds...")

    # --- 3. Phase 2: Gradient Refinement ---
    print("\nPHASE 2: Running Gradient Descent Refinement...")
    best_overall_score = -float('inf')
    best_final_result = None

    start_time = time.time()

    for idx, seed in enumerate(top_seeds):
        piston = seed['piston']
        c_mount = seed['chassis_mount']
        d_mount = seed['door_mount']

        # Format starting position for the gradient optimizer
        start_pos = np.array([c_mount[0], c_mount[1], d_mount[0], d_mount[1]])

        # Run discrete search with a finer resolution than the initial grid
        gradient_resolution = opt_cfg.grid_resolution / 90
        res = optimizer.run_discrete_search(
            piston,
            chassis_zone,
            door_zone,
            start_pos,
            max_steps=opt_cfg.max_gradient_steps,
            resolution=gradient_resolution
        )

        if res['score'] > best_overall_score:
            best_overall_score = res['score']
            best_final_result = {
                'params': res['params'],
                'piston': piston,
                'score': res['score']
            }

        print(f"  Seed {idx + 1}/{len(top_seeds)} refined: {seed['score']:.2f} -> {res['score']:.2f}")

    elapsed = time.time() - start_time
    print(f"\nRefinement completed in {elapsed:.2f} seconds.")
    # --- 4. Final Output ---
    if best_final_result:
        params = best_final_result['params']
        p = best_final_result['piston']

        # Update config for final report
        base_cfg.chassis_piston_anchor_meter = params[0:2]
        base_cfg.piston_mount_on_door_meter = params[2:4]
        base_cfg.strut_max_length = p.max_length
        base_cfg.strut_min_length = p.max_length - p.stroke
        base_cfg.f_ext = p.f_ext
        base_cfg.f_comp = p.f_comp

        best_res = optimizer.engine.run(base_cfg)
        metrics = evaluator.calculate_metrics(best_res, base_cfg)
        print("-" * 40)
        print("--- HYBRID OPTIMIZATION BEST RESULT ---")
        print(f"Piston:         {p.name}")
        print(f"Chassis Anchor: {base_cfg.chassis_piston_anchor_meter}")
        print(f"Door Anchor:    {base_cfg.piston_mount_on_door_meter}")
        print(f"Final Score:    {best_final_result['score']:.4f}")
        print("\n--- PERFORMANCE ---")
        print(f"User Force Close: {metrics['actual_close_force']:.1f} N")
        print(f"User Force Open:  {metrics['actual_open_force']:.1f} N")
        print(f"Hinge Stress:     {metrics['max_hinge_force']:.1f} N")

        print("\nCOPY TO SimulationGraph.py:")
        print(
            f"SimulationConfig("
            f"chassis_piston_anchor=np.array([{params[0]:.6f}, {params[1]:.6f}]), "
            f"piston_mount_on_door=np.array([{params[2]:.6f}, {params[3]:.6f}]), "
            f"center_of_mass_on_door=np.array([{base_cfg.center_of_mass_on_door[0]:.6f}, {base_cfg.center_of_mass_on_door[1]:.6f}]), "
            f"door_length={base_cfg.door_length}, "
            f"door_mass_kg={base_cfg.door_mass_kg}, "
            f"strut_max_length={p.max_length}, "
            f"strut_stroke={p.stroke}, "
            f"extension_force_n={p.f_ext}, "
            f"compression_force_n={p.f_comp}, "
            f"door_close_angle_deg={base_cfg.door_close_angle_deg})"
        )
        print("-" * 40)
def transform_to_door_coordinate(point: Tuple[float, float], angle_deg: float) -> Tuple[float, float]:
    """
    Transforms a point from global coordinates to a local system where 0 degrees
    aligns the local x-axis with the global negative y-axis.
    """
    radians = math.radians(angle_deg)
    cos_val = math.cos(radians)
    sin_val = math.sin(radians)

    # Global point coordinates
    gx, gy = point


    local_x = gx * sin_val - gy * cos_val
    local_y = gx * cos_val + gy * sin_val

    return (local_x, local_y)

# Example of transforming your entire list:
def transform_door_zone(coords: List[Tuple[float, float]], angle_deg: float) -> List[Tuple[float, float]]:
    return [transform_to_door_coordinate(p, angle_deg) for p in coords]
if __name__ == "__main__":
    # Example of how to use the new config class
    custom_config = OptimizationConfig(
        grid_resolution=0.03,
        top_n_grid=5,
        piston_catalog_size=50
    )
    run_hybrid_optimization(custom_config)