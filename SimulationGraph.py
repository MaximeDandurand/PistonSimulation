import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Internal.Config import SimulationConfig
from Internal.SimulationEngine import HatchbackPhysicsEngine


def draw_frame(ax, cfg, state_data, alpha=1.0, title=""):
    """
    Renders a door frame using pre-calculated state data.
    """
    p_door_xy = state_data['mount']
    p_com_xy = state_data['com']
    p_end_xy = state_data['end']
    h_vec = state_data['hinge_v']
    color = state_data['color']
    deg = state_data['deg']
    rad = np.radians(deg)

    # 1. Door Rendering
    t_arc = np.linspace(0, np.pi, 20)
    arc_points = np.array([HatchbackPhysicsEngine.get_rotated_point([x, 0.15 * np.sin(t)], rad)
                           for x, t in zip(np.linspace(0, cfg.door_length, 20), t_arc)])
    ax.plot(arc_points[:, 0], arc_points[:, 1], 'k-', alpha=alpha, lw=2.5)

    # 2. Strut and Anchor
    ax.plot([cfg.chassis_piston_anchor_meter[0], p_door_xy[0]],
            [cfg.chassis_piston_anchor_meter[1], p_door_xy[1]], '--', color=color, alpha=alpha * 0.6)
    ax.scatter(*cfg.chassis_piston_anchor_meter, color='black', s=60, zorder=10, alpha=alpha)

    # 3. Mount and CoM Dots
    ax.scatter(*p_door_xy, color='darkorange', s=35, zorder=5, alpha=alpha)
    ax.scatter(*p_com_xy, color='blue', s=35, zorder=5, alpha=alpha)

    # 4. Force Arrows
    perp_dir = np.array([np.cos(rad), np.sin(rad)])
    net_mag = state_data['net']
    dx, dy = perp_dir * (net_mag / 400.0)
    ax.arrow(p_end_xy[0], p_end_xy[1], dx, dy, head_width=0.04, color=color, alpha=alpha * 0.8)
    ax.arrow(0, 0, h_vec[0], h_vec[1], head_width=0.03, color='purple', alpha=alpha * 0.5)

    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-2.0, 1.5)
    ax.set_title(title)


def run_interactive_simulation(cfg):
    engine = HatchbackPhysicsEngine()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 9))
    plt.subplots_adjust(bottom=0.4, wspace=0.3)
    ax2b = ax2.twinx()

    # --- SLIDERS ---
    sw, sh = 0.12, 0.02
    c1, c2, c3, c4 = 0.05, 0.28, 0.52, 0.76
    ax_cx = plt.axes([c1, 0.25, sw, sh]);
    s_cx = Slider(ax_cx, 'Chassis X', -2.0, 2.0, valinit=cfg.chassis_piston_anchor_meter[0])
    ax_cy = plt.axes([c1, 0.21, sw, sh]);
    s_cy = Slider(ax_cy, 'Chassis Y', -2.0, 2.0, valinit=cfg.chassis_piston_anchor_meter[1])
    ax_mx = plt.axes([c2, 0.25, sw, sh]);
    s_mx = Slider(ax_mx, 'Mount X', 0.0, 2.5, valinit=cfg.piston_mount_on_door_meter[0])
    ax_my = plt.axes([c2, 0.21, sw, sh]);
    s_my = Slider(ax_my, 'Mount Y', -1.0, 1.0, valinit=cfg.piston_mount_on_door_meter[1])
    ax_length = plt.axes([c2, 0.17, sw, sh]);
    s_length = Slider(ax_length, 'Door Len', 0.1, 4.0, valinit=cfg.door_length)
    ax_mass = plt.axes([c3, 0.25, sw, sh]);
    s_mass = Slider(ax_mass, 'Mass (kg)', 1.0, 150.0, valinit=cfg.door_mass_kg)
    ax_comx = plt.axes([c3, 0.21, sw, sh]);
    s_comx = Slider(ax_comx, 'CoM X', 0.0, 2.5, valinit=cfg.center_of_mass_on_door[0])
    ax_comy = plt.axes([c3, 0.17, sw, sh]);
    s_comy = Slider(ax_comy, 'CoM Y', -1.0, 1.0, valinit=cfg.center_of_mass_on_door[1])
    ax_maxl = plt.axes([c4, 0.25, sw, sh]);
    s_maxl = Slider(ax_maxl, 'Max L', 0.1, 5.0, valinit=cfg.strut_max_length)
    ax_stroke = plt.axes([c4, 0.21, sw, sh]);
    s_stroke = Slider(ax_stroke, 'Stroke', 0.01, 4.0, valinit=cfg.strut_max_length - cfg.strut_min_length)
    ax_fext = plt.axes([c4, 0.17, sw, sh]);
    s_fext = Slider(ax_fext, 'f_ext (P1)', 0, 3000, valinit=cfg.f_ext)
    ax_fcomp = plt.axes([c4, 0.13, sw, sh]);
    s_fcomp = Slider(ax_fcomp, 'f_comp (P2)', 0, 4000, valinit=cfg.f_comp)

    def update(val):
        cfg.chassis_piston_anchor_meter = np.array([s_cx.val, s_cy.val])
        cfg.piston_mount_on_door_meter = np.array([s_mx.val, s_my.val])
        cfg.center_of_mass_on_door = np.array([s_comx.val, s_comy.val])
        cfg.door_length = s_length.val
        cfg.door_mass_kg = s_mass.val
        cfg.strut_max_length = s_maxl.val
        cfg.strut_min_length = s_maxl.val - s_stroke.val
        cfg.f_ext = s_fext.val
        cfg.f_comp = s_fcomp.val

        res = engine.run(cfg, steps=100)
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax2b.clear()

        # Render Path Snapshots
        num_res = len(res.angles_deg)
        if num_res > 0:
            snapshot_indices = [0, num_res // 2, num_res - 1]
            snapshot_indices = sorted(list(set(snapshot_indices)))

            for idx in snapshot_indices:
                state = {
                    'mount': res.mount_coords[idx], 'com': res.com_coords[idx],
                    'end': res.door_end_coords[idx], 'hinge_v': res.hinge_vectors[idx],
                    'color': res.strut_colors[idx], 'deg': res.angles_deg[idx], 'net': res.net_torques[idx]
                }
                alpha = 0.6 if res.is_valid_mask[idx] else 0.15
                draw_frame(ax1, cfg, state, alpha=alpha, title=f"Motion Path (Max: {res.max_physical_angle:.1f}°)")

        # Render Equilibrium
        if res.equilibrium_angle:
            eq_state = engine.get_state_at_angle(cfg, res.equilibrium_angle)
            draw_frame(ax3, cfg, eq_state, title=f"Equilibrium: {res.equilibrium_angle:.1f}°")
        else:
            ax3.set_title("No Balance Point")

        # --- Plots (ax2) ---
        # 1. Plot the main torque curve
        ax2.plot(res.angles_deg, res.net_torques, 'k-', zorder=3)
        ax2.axhline(0, color='black', lw=1, zorder=2)

        # 2. Fill functional regions (Green for lifting assistance, Red for closing weight)
        ax2.fill_between(res.angles_deg, 0, res.net_torques, where=np.array(res.net_torques) > 0,
                         color='green', alpha=0.3)
        ax2.fill_between(res.angles_deg, 0, res.net_torques, where=np.array(res.net_torques) < 0,
                         color='red', alpha=0.3)

        # 3. Highlight INVALID regions (Physical constraints violated)
        # This creates a hatched grey background where the mask is False
        invalid_mask = ~np.array(res.is_valid_mask)
        ax2.fill_between(res.angles_deg, -1e6, 1e6, where=invalid_mask,
                         color='grey', alpha=0.2, hatch='//', label='Invalid Geometry', zorder=1)

        # Secondary Axis for Hinge Forces
        ax2b.plot(res.angles_deg, res.hinge_forces, 'p--', alpha=0.3)

        # Clean up y-limits so the 'invalid' fill doesn't zoom out the graph to 1e6
        if len(res.net_torques) > 0:
            y_min, y_max = min(res.net_torques), max(res.net_torques)
            padding = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            ax2.set_ylim(y_min - padding, y_max + padding)

        fig.canvas.draw_idle()

    for s in [s_cx, s_cy, s_mx, s_my, s_length, s_mass, s_comx, s_comy, s_maxl, s_stroke, s_fext, s_fcomp]:
        s.on_changed(update)

    update(None)
    plt.show()


if __name__ == "__main__":
    run_interactive_simulation(
        SimulationConfig(chassis_piston_anchor=np.array([0.080667, -0.530000]),
                         piston_mount_on_door=np.array([0.130000, -0.040000]),
                         center_of_mass_on_door=np.array([0.500000, 0.000000]), door_length=1, door_mass_kg=25.0,
                         strut_max_length=0.66802, strut_stroke=0.278892, extension_force_n=444.822,
                         compression_force_n=578.2686, door_close_angle_deg=10.0)

    )