import numpy as np
from typing import List, Dict, Any
from Config import SimulationConfig
from Optimizer import PistonSpec, MountingArea, run_grid_search
from EvaluationEngine import EvaluationEngine


def run_simulation_workflow() -> None:
    base_cfg = SimulationConfig()


    chassis_zone = MountingArea([
        (0.0, -0.3),
        (0.1, -0.3),
        (0.1, -0.5),
        (0.0, -0.5)
    ])

    door_zone = MountingArea([
        (0.1, 0.0),
        (0.3, 0.0),
        (0.2, -0.1)
    ])

    #Piston that we know is going to work
    piston_catalog = [
        PistonSpec(
            name="Reference-Piston",
            max_length=base_cfg.strut_max_length,
            stroke=base_cfg.strut_max_length - base_cfg.strut_min_length,
            f_ext=base_cfg.f_ext,
            f_comp=base_cfg.f_comp
        ),
        PistonSpec("HeavyDuty-800N", max_length=0.6, stroke=0.25, f_ext=800, f_comp=1000)
    ]

    #Run grid search
    print(f"Searching for valid mounting configurations...")
    valid_solutions = run_grid_search(
        base_cfg=base_cfg,
        chassis_poly=chassis_zone,
        door_poly=door_zone,
        pistons=piston_catalog,
        resolution=0.01,  # Finer resolution to hit the reference points
        min_angle_threshold=70.0
    )

    if not valid_solutions:
        print("No valid configurations found.")
        return

    #Evaluate
    evaluator = EvaluationEngine(target_close_force_n=20.0, target_open_force_n=10.0) #Value to determine 10N aprox. 1kg to lift
    ranked_results = evaluator.evaluate_all(valid_solutions, base_cfg)

    # 6. Output the Results
    print(f"\nFound {len(ranked_results)} valid solutions.")

    # Check if our Reference setup made it into the valid list
    found_ref = False
    for sol in ranked_results:
        if sol['piston'].name == "Reference-Piston":
            # Check if coordinates are near the Config.py defaults
            c_dist = np.linalg.norm(sol['chassis_mount'] - np.array([0.05, -0.4]))
            d_dist = np.linalg.norm(sol['door_mount'] - np.array([0.2, -0.035]))

            if c_dist < 0.015 and d_dist < 0.015:
                print(f"\n--- Verified Reference Configuration Found ---")
                print(f"Score: {sol['score']:.2f}")
                print(f"Summary: {sol['summary']}")
                found_ref = True
                break

    if not found_ref:
        print("\nNote: The exact reference point was missed by the grid resolution.")

    print("\nTop Candidate overall:")
    best = ranked_results[0]
    print(f"Piston: {best['piston'].name}")
    print(f"Chassis Mount: {best['chassis_mount']}")
    print(f"Door Mount:    {best['door_mount']}")
    print(f"Score: {best['score']:.2f}")
    print(f"Detail: {best['summary']}")


if __name__ == "__main__":
    run_simulation_workflow()