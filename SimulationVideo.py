import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Config import SimulationConfig


def get_rotated_point(local_coords, theta):
    along, depth = local_coords
    x = along * np.sin(theta) + depth * np.cos(theta)
    y = -along * np.cos(theta) + depth * np.sin(theta)
    return np.array([x, y])


def animate_hatchback(cfg):
    # --- 1. DATA PREPARATION ---
    angles_deg = np.linspace(0, 95, 200)
    simulation_results = []

    for deg in angles_deg:
        rad = np.radians(deg)
        p_door_xy = get_rotated_point(cfg.piston_mount_on_door, rad)
        door_end_xy = np.array([cfg.door_length * np.sin(rad), -cfg.door_length * np.cos(rad)])

        # Piston Physics
        piston_vec = p_door_xy - cfg.chassis_piston_anchor
        current_L = np.linalg.norm(piston_vec)
        piston_unit = piston_vec / current_L

        # Force Interpolation
        if current_L <= cfg.strut_min_length:
            current_force = cfg.f2
        elif current_L >= cfg.strut_max_length:
            current_force = cfg.f1
        else:
            compression_pct = (cfg.strut_max_length - current_L) / (cfg.strut_max_length - cfg.strut_min_length)
            current_force = cfg.f1 + (cfg.f2 - cfg.f1) * compression_pct

        # Torque
        t_piston = np.cross(p_door_xy, piston_unit * current_force)
        t_gravity = np.cross(get_rotated_point(cfg.center_of_mass_on_door, rad),
                             np.array([0, -cfg.door_mass_kg * cfg.gravity_constant]))

        simulation_results.append({
            'deg': deg, 'rad': rad, 'p_door_xy': p_door_xy, 'door_end_xy': door_end_xy,
            'net_torque': t_piston + t_gravity, 'current_force': current_force
        })

    # --- 2. PLOT SETUP ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Geometry Plot Elements
    door_line, = ax1.plot([], [], 'k-', lw=3)
    strut_line, = ax1.plot([], [], 'g--', alpha=0.8)
    perp_arrow = ax1.quiver(0, 0, 0, 0, color='green', scale=1, scale_units='xy')
    ax1.scatter(*cfg.chassis_piston_anchor, color='black', zorder=5)
    ax1.set_xlim(-0.3, 1.2);
    ax1.set_ylim(-1.2, 0.3);
    ax1.set_aspect('equal')

    # Torque Graph Elements
    degs_all = [r['deg'] for r in simulation_results]
    nets_all = [r['net_torque'] for r in simulation_results]
    ax2.plot(degs_all, nets_all, 'k-', alpha=0.3)  # Static background curve
    indicator_dot, = ax2.plot([], [], 'ro')  # Moving dot
    ax2.set_xlim(0, 95);
    ax2.set_ylim(min(nets_all) - 10, max(nets_all) + 10)
    ax2.axhline(0, color='black', lw=1)

    # --- 3. ANIMATION UPDATE FUNCTION ---
    def update(frame):
        res = simulation_results[frame]

        # Update Door
        door_line.set_data([0, res['door_end_xy'][0]], [0, res['door_end_xy'][1]])

        # Update Strut
        strut_line.set_data([cfg.chassis_piston_anchor[0], res['p_door_xy'][0]],
                            [cfg.chassis_piston_anchor[1], res['p_door_xy'][1]])

        # Update Perpendicular Arrow
        perp_dir = np.array([np.cos(res['rad']), np.sin(res['rad'])])
        mag = res['net_torque'] / 300.0
        perp_arrow.set_offsets(res['door_end_xy'])
        perp_arrow.set_UVC(perp_dir[0] * mag, perp_dir[1] * mag)
        perp_arrow.set_color('green' if res['net_torque'] > 0 else 'red')

        # Update Torque Dot
        indicator_dot.set_data([res['deg']], [res['net_torque']])

        return door_line, strut_line, perp_arrow, indicator_dot

    ani = FuncAnimation(fig, update, frames=len(simulation_results), interval=30, blit=True)
    plt.show()


if __name__ == "__main__":
    animate_hatchback(SimulationConfig())