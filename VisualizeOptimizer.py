import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from Internal.Config import SimulationConfig
from Internal.GradientOptimizer import PistonSpec, MountingArea, DiscreteGradientOptimizer
from Internal.EvaluationEngine import EvaluationEngine


def visualize_optimization(num_seeds=100):
    base_cfg = SimulationConfig()
    evaluator = EvaluationEngine()
    optimizer = DiscreteGradientOptimizer(base_cfg, evaluator)

    piston = PistonSpec("HeavyDuty-800N", 0.6, 0.25, 800, 1000)

    chassis_zone = MountingArea([(0.0, -0.4), (0.1, -0.4), (0.1, -0.6), (0.0, -0.6)])
    door_zone = MountingArea([(0., 0.0), (0.2, 0.0), (0.1, -0.1)])

    all_histories = []
    path_lengths = []

    # Trackers for the final report
    count_length_1_valid = 0
    count_length_1_invalid = 0
    count_moving_valid = 0
    count_moving_invalid = 0

    # Track the absolute best solution
    best_overall_score = -float('inf')
    best_params = None

    print(f"Running {num_seeds} seeds... Please wait.")

    for i in range(num_seeds):
        # Generate points strictly inside polygons
        while True:
            c_start = np.array([
                random.uniform(chassis_zone.x_min, chassis_zone.x_max),
                random.uniform(chassis_zone.y_min, chassis_zone.y_max)
            ])
            if chassis_zone.contains(c_start): break

        while True:
            d_start = np.array([
                random.uniform(door_zone.x_min, door_zone.x_max),
                random.uniform(door_zone.y_min, door_zone.y_max)
            ])
            if door_zone.contains(d_start): break

        start_pos = np.concatenate([c_start, d_start])
        res = optimizer.run_discrete_search(piston, chassis_zone, door_zone, start_pos)

        # Update Best Global Solution
        if res['score'] > best_overall_score:
            best_overall_score = res['score']
            best_params = res['params']

        history = res['history']
        all_histories.append(history)
        path_lengths.append(len(history))

        # Categorize for summary
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

    # --- Diagnostic Summary ---
    print("-" * 40)
    print(f"TOTAL SEEDS PROCESSED: {num_seeds}")
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

        # Apply the best parameters to the config
        base_cfg.chassis_piston_anchor = best_params[0:2]
        base_cfg.piston_mount_on_door = best_params[2:4]

        # Ensure the config matches the optimizer's piston specs for accuracy
        base_cfg.strut_max_length = piston.max_length
        base_cfg.strut_min_length = piston.max_length - piston.stroke
        base_cfg.f_ext = piston.f_ext
        base_cfg.f_comp = piston.f_comp

        # Run final simulation for metrics
        best_res = optimizer.engine.run(base_cfg)
        best_metrics = evaluator.calculate_metrics(best_res, base_cfg)

        print("-" * 40)
        print("--- BEST CONFIGURATION PARAMETERS ---")
        print(
            f"  Chassis Anchor: np.array([{base_cfg.chassis_piston_anchor[0]:.4f}, {base_cfg.chassis_piston_anchor[1]:.4f}])")
        print(
            f"  Door Anchor:    np.array([{base_cfg.piston_mount_on_door[0]:.4f}, {base_cfg.piston_mount_on_door[1]:.4f}])")
        print(f"  Max Length:     {base_cfg.strut_max_length:.4f}")
        print(f"  Min Length:     {base_cfg.strut_min_length:.4f}")
        print(f"  Force Ext/Comp: {base_cfg.f_ext}N / {base_cfg.f_comp}N")

        print("\n--- PERFORMANCE METRICS ---")
        print(f"  Max Opening Angle: {best_res.max_physical_angle:.2f}°")
        print(f"  User Force Close:  {best_metrics['actual_close_force']:.1f} N")
        print(f"  User Force Open:   {best_metrics['actual_open_force']:.1f} N")
        print(f"  Max Hinge Stress:  {best_metrics['max_hinge_force']:.1f} N")
        print("-" * 40)

        # Copy-paste friendly line for SimulationGraph.py
        print("\nCOPY THIS TO SimulationGraph.py:")
        print(
            f"SimulationConfig(chassis_piston_anchor=np.array([{base_cfg.chassis_piston_anchor[0]:.6f}, {base_cfg.chassis_piston_anchor[1]:.6f}]), "
            f"piston_mount_on_door=np.array([{base_cfg.piston_mount_on_door[0]:.6f}, {base_cfg.piston_mount_on_door[1]:.6f}]), "
            f"strut_max_length={base_cfg.strut_max_length}, strut_stroke={piston.stroke})")
        print("-" * 40)
    print("-" * 40)

    avg_len = np.mean(path_lengths)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, zone in zip([ax1, ax2], [chassis_zone, door_zone]):
        v = np.array(zone.path.vertices)
        v = np.vstack([v, v[0]])
        ax.plot(v[:, 0], v[:, 1], 'k--', alpha=0.3)

    ax1.set_title(f"Chassis (Avg: {avg_len:.1f} steps)")
    ax2.set_title("Door")

    lines_c = [ax1.plot([], [], '-', alpha=0.4)[0] for _ in all_histories]
    heads_c = [ax1.plot([], [], 'o', markersize=6, markeredgecolor='black')[0] for _ in all_histories]
    lines_d = [ax2.plot([], [], '-', alpha=0.4)[0] for _ in all_histories]
    heads_d = [ax2.plot([], [], 'o', markersize=6, markeredgecolor='black')[0] for _ in all_histories]

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