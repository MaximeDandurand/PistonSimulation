import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from Internal.Config import SimulationConfig
from Internal.ExcelCalculator import get_optimal_start
from Internal.GradientOptimizer import PistonSpec, MountingArea, DiscreteGradientOptimizer
from Internal.EvaluationEngine import EvaluationEngine


def visualize_optimization(num_seeds=100):
    base_cfg = SimulationConfig()
    evaluator = EvaluationEngine()
    optimizer = DiscreteGradientOptimizer(base_cfg, evaluator)

    # Your base coordinates
    base_coords = [(-0.05, -0.05), (-0.05, -0.9), (0.05, -0.9), (0.05, -0.05)]

    # Example: 10 degrees clockwise from the negative y-axis
    angle_rad = math.radians(base_cfg.door_close_angle_deg)

    rotated_coords = []

    for x, y in base_coords:
        new_x = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        new_y = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        rotated_coords.append((new_x, new_y))

    chassis_zone = MountingArea(rotated_coords)
    door_zone = MountingArea([(0., 0.1), (0.5, 0.1), (0.5, -0.1), (0., -0.1)])

    # 2. Get Geometric Seeds based on the established 0° downward logic
    # returns: best_piston, door_anchor_local, valid_frame_arc
    piston, ideal_door_anchor, valid_frame_arc = get_optimal_start(
        base_cfg, opened_angle=100, closed_angle=base_cfg.door_close_angle_deg, area=chassis_zone
    )

    # Pre-generate valid door points to ensure we have a pool to sample from
    door_candidates = door_zone.get_valid_points(resolution=0.04)

    all_histories = []
    path_lengths = []

    # Trackers for the final report
    count_length_1_valid = 0
    count_length_1_invalid = 0
    count_moving_valid = 0
    count_moving_invalid = 0

    best_overall_score = -float('inf')
    best_params = None

    print(f"Running {num_seeds} seeds with deterministic sampling...")

    start_time = time.time()  # Start timer

    for i in range(num_seeds):
        # 3. Sample Chassis Point
        if valid_frame_arc:
            c_start = random.choice(valid_frame_arc)
        else:
            chassis_candidates = chassis_zone.get_valid_points(resolution=0.02)
            c_start = random.choice(chassis_candidates)

        # 4. Sample Door Point
        if door_candidates:
            distances = [np.linalg.norm(p - ideal_door_anchor) for p in door_candidates]
            closest_indices = np.argsort(distances)[:10]
            d_start = door_candidates[random.choice(closest_indices)]
        else:
            d_start = ideal_door_anchor

        start_pos = np.concatenate([c_start, d_start])
        res = optimizer.run_discrete_search(piston, chassis_zone, door_zone, start_pos, max_steps=200)

        if res['score'] > best_overall_score:
            best_overall_score = res['score']
            best_params = res['params']

        history = res['history']
        all_histories.append(history)
        path_lengths.append(len(history))

        is_valid = res['score'] > 0
        has_moved = len(history) > 1

        if has_moved:
            if is_valid:
                count_moving_valid += 1
            else:
                count_moving_invalid += 1
        else:
            if is_valid:
                count_length_1_valid += 1
            else:
                count_length_1_invalid += 1

    end_time = time.time()  # End timer
    elapsed = end_time - start_time

    # --- Diagnostic Summary ---
    print("-" * 40)
    print(f"TOTAL SEEDS PROCESSED: {num_seeds}")
    print(f"TOTAL DURATION:        {elapsed:.2f} seconds")
    print(f"AVG TIME PER SEED:     {elapsed / num_seeds:.4f} seconds")
    print(f"\nSTATIONARY PATHS (Length = 1):")
    print(f"  - Valid/Good Starts:       {count_length_1_valid}")
    print(f"  - Invalid/Stuck Starts:    {count_length_1_invalid}")
    print(f"\nOPTIMIZED PATHS (Length > 1):")
    print(f"  - Successfully made Valid: {count_moving_valid}")
    print(f"  - Stayed Invalid:          {count_moving_invalid}")
    print("-" * 40)

    # --- Best Solution Printout ---
    if best_params is not None:
        print(f"BEST SOLUTION FOUND (Score: {best_overall_score:.4f}):")

        base_cfg.chassis_piston_anchor_meter = best_params[0:2]
        base_cfg.piston_mount_on_door_meter = best_params[2:4]

        base_cfg.strut_max_length = piston.max_length
        base_cfg.strut_min_length = piston.max_length - piston.stroke
        base_cfg.f_ext = piston.f_ext
        base_cfg.f_comp = piston.f_comp

        best_res = optimizer.engine.run(base_cfg)
        best_metrics = evaluator.calculate_metrics(best_res, base_cfg)

        print("-" * 40)
        print("--- BEST CONFIGURATION PARAMETERS ---")
        print(
            f"  Chassis Anchor: np.array([{base_cfg.chassis_piston_anchor_meter[0]:.4f}, {base_cfg.chassis_piston_anchor_meter[1]:.4f}])")
        print(
            f"  Door Anchor:    np.array([{base_cfg.piston_mount_on_door_meter[0]:.4f}, {base_cfg.piston_mount_on_door_meter[1]:.4f}])")
        print(f"  Max Length:     {base_cfg.strut_max_length:.4f}")
        print(f"  Min Length:     {base_cfg.strut_min_length:.4f}")
        print(f"  Force Ext/Comp: {base_cfg.f_ext}N / {base_cfg.f_comp}N")

        print("\n--- PERFORMANCE METRICS ---")
        print(f"  Max Opening Angle: {best_res.max_physical_angle:.2f}°")
        print(f"  User Force Close:  {best_metrics['actual_close_force']:.1f} N")
        print(f"  User Force Open:   {best_metrics['actual_open_force']:.1f} N")
        print(f"  Max Hinge Stress:  {best_metrics['max_hinge_force']:.1f} N")
        print("-" * 40)

        print("\nCOPY THIS TO SimulationGraph.py:")
        print(
            f"SimulationConfig("
            f"chassis_piston_anchor=np.array([{base_cfg.chassis_piston_anchor_meter[0]:.6f}, {base_cfg.chassis_piston_anchor_meter[1]:.6f}]), "
            f"piston_mount_on_door=np.array([{base_cfg.piston_mount_on_door_meter[0]:.6f}, {base_cfg.piston_mount_on_door_meter[1]:.6f}]), "
            f"center_of_mass_on_door=np.array([{base_cfg.center_of_mass_on_door[0]:.6f}, {base_cfg.center_of_mass_on_door[1]:.6f}]), "
            f"door_length={base_cfg.door_length}, "
            f"door_mass_kg={base_cfg.door_mass_kg}, "
            f"strut_max_length={base_cfg.strut_max_length}, "
            f"strut_stroke={piston.stroke}, "
            f"extension_force_n={base_cfg.f_ext}, "
            f"compression_force_n={base_cfg.f_comp}, "
            f"door_close_angle_deg={base_cfg.door_close_angle_deg})"
        )
        print("-" * 40)
    print("-" * 40)

    avg_len = np.mean(path_lengths)
    fig, ax = plt.subplots(figsize=(10, 8))

    for zone, label in zip([chassis_zone, door_zone], ["Chassis", "Door"]):
        v = np.array(zone.path.vertices)
        v = np.vstack([v, v[0]])
        ax.plot(v[:, 0], v[:, 1], 'k--', alpha=0.3, label=label)

    ax.set_title(f"Optimization Paths (Avg: {avg_len:.1f} steps)")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    lines_c = [ax.plot([], [], '-', alpha=0.3, color='blue')[0] for _ in all_histories]
    lines_d = [ax.plot([], [], '-', alpha=0.3, color='orange')[0] for _ in all_histories]
    heads_c = [ax.plot([], [], 'o', markersize=5, markeredgecolor='black')[0] for _ in all_histories]
    heads_d = [ax.plot([], [], 'o', markersize=5, markeredgecolor='black')[0] for _ in all_histories]

    max_steps = max(path_lengths)

    def init():
        for item in lines_c + heads_c + lines_d + heads_d:
            item.set_data([], [])
        return lines_c + heads_c + lines_d + heads_d

    def update(frame):
        for i, hist in enumerate(all_histories):
            step = min(frame, len(hist) - 1)
            current_pos = hist[step]
            score = optimizer.get_score(current_pos, piston)
            dot_color = 'green' if score > 0 else 'red'

            lines_c[i].set_data(hist[:step + 1, 0], hist[:step + 1, 1])
            heads_c[i].set_data([current_pos[0]], [current_pos[1]])
            heads_c[i].set_markerfacecolor(dot_color)

            lines_d[i].set_data(hist[:step + 1, 2], hist[:step + 1, 3])
            heads_d[i].set_data([current_pos[2]], [current_pos[3]])
            heads_d[i].set_markerfacecolor(dot_color)

        return lines_c + heads_c + lines_d + heads_d

    global ani
    ani = animation.FuncAnimation(fig, update, frames=max_steps, init_func=init,
                                  blit=True, interval=50, repeat=True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_optimization(num_seeds=20)