"""
Interactive hatchback simulation.
RESTORED: Torque arrows, P/G labels, and CoM/Anchor markers.
ADDED: Third graph for Equilibrium Point (Net Torque = 0).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from Config import SimulationConfig

def get_rotated_point(local_coords, theta):
    """Transforms local door coordinates to global [x, y] based on rotation theta."""
    along, depth = local_coords
    x = along * np.sin(theta) + depth * np.cos(theta)
    y = -along * np.cos(theta) + depth * np.sin(theta)
    return np.array([x, y])

def draw_complete_geometry(ax, cfg, rad, alpha=1.0, title=""):
    """
    Renders the door with all technical indicators:
    Arrows, Torque Labels, Mounting Points, and CoM.
    """
    p_door_xy = get_rotated_point(cfg.piston_mount_on_door, rad)
    p_com_xy = get_rotated_point(cfg.center_of_mass_on_door, rad)
    door_end_xy = np.array([cfg.door_length * np.sin(rad), -cfg.door_length * np.cos(rad)])

    # Physics Calculation for indicators
    p_vec = p_door_xy - cfg.chassis_piston_anchor
    L = np.linalg.norm(p_vec)
    p_unit = p_vec / L

    if L >= cfg.strut_max_length: f_piston = cfg.f_ext
    elif L <= cfg.strut_min_length: f_piston = cfg.f_comp
    else:
        pct = (cfg.strut_max_length - L) / (cfg.strut_max_length - cfg.strut_min_length)
        f_piston = cfg.f_ext + (cfg.f_comp - cfg.f_ext) * pct

    tp = np.cross(p_door_xy, p_unit * f_piston)
    tg = np.cross(p_com_xy, np.array([0, -cfg.door_mass_kg * cfg.gravity_constant]))
    net = tp + tg
    color = 'green' if net > 0 else 'red'

    # 1. Door Rendering
    t_arc = np.linspace(0, np.pi, 20)
    arc_points = np.array([get_rotated_point([x, 0.15 * np.sin(t)], rad)
                           for x, t in zip(np.linspace(0, cfg.door_length, 20), t_arc)])
    ax.plot(arc_points[:, 0], arc_points[:, 1], 'k-', alpha=alpha, lw=2.5)

    # 2. Strut and Anchor
    ax.plot([cfg.chassis_piston_anchor[0], p_door_xy[0]],
             [cfg.chassis_piston_anchor[1], p_door_xy[1]], '--', color=color, alpha=alpha*0.6)
    ax.scatter(*cfg.chassis_piston_anchor, color='black', s=60, zorder=10, alpha=alpha)

    # 3. Mount and CoM Dots
    ax.scatter(*p_door_xy, color='darkorange', s=35, zorder=5, alpha=alpha)
    ax.scatter(*p_com_xy, color='blue', s=35, zorder=5, alpha=alpha)

    # 4. Torque Labels
    ax.text(p_door_xy[0], p_door_xy[1]+0.04, f"P:{tp:.0f}", fontsize=8, color='darkorange', alpha=alpha)
    ax.text(p_com_xy[0], p_com_xy[1]-0.08, f"G:{tg:.0f}", fontsize=8, color='blue', alpha=alpha)

    # 5. Torque Arrow
    perp_dir = np.array([np.cos(rad), np.sin(rad)])
    dx, dy = perp_dir * (net / 400.0)
    ax.arrow(door_end_xy[0], door_end_xy[1], dx, dy, head_width=0.04, color=color, alpha=alpha*0.8)

    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-2.0, 1.5)
    ax.set_title(title)

def run_interactive_simulation(cfg):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 9))
    plt.subplots_adjust(bottom=0.4, wspace=0.3)

    # --- SLIDERS ---
    sw, sh = 0.12, 0.02
    c1, c2, c3, c4 = 0.05, 0.28, 0.52, 0.76

    ax_cx = plt.axes([c1, 0.25, sw, sh]); s_cx = Slider(ax_cx, 'Chassis X', -1.5, 0.5, valinit=cfg.chassis_piston_anchor[0])
    ax_cy = plt.axes([c1, 0.21, sw, sh]); s_cy = Slider(ax_cy, 'Chassis Y', -1.5, 0.5, valinit=cfg.chassis_piston_anchor[1])

    ax_mx = plt.axes([c2, 0.25, sw, sh]); s_mx = Slider(ax_mx, 'Mount X', 0.1, 1.5, valinit=cfg.piston_mount_on_door[0])
    ax_my = plt.axes([c2, 0.21, sw, sh]); s_my = Slider(ax_my, 'Mount Y', -0.3, 0.3, valinit=cfg.piston_mount_on_door[1])
    ax_length = plt.axes([c2, 0.17, sw, sh]); s_length = Slider(ax_length, 'Door Len', 0.5, 2.0, valinit=cfg.door_length)

    ax_mass = plt.axes([c3, 0.25, sw, sh]); s_mass = Slider(ax_mass, 'Mass (kg)', 5.0, 80.0, valinit=cfg.door_mass_kg)
    ax_comx = plt.axes([c3, 0.21, sw, sh]); s_comx = Slider(ax_comx, 'CoM X', 0.1, 1.5, valinit=cfg.center_of_mass_on_door[0])
    ax_comy = plt.axes([c3, 0.17, sw, sh]); s_comy = Slider(ax_comy, 'CoM Y', -0.3, 0.3, valinit=cfg.center_of_mass_on_door[1])

    ax_maxl = plt.axes([c4, 0.25, sw, sh]); s_maxl = Slider(ax_maxl, 'Max L', 0.1, 4.0, valinit=cfg.strut_max_length)
    ax_stroke = plt.axes([c4, 0.21, sw, sh]); s_stroke = Slider(ax_stroke, 'Stroke', 0.05, 3.0, valinit=cfg.strut_max_length - cfg.strut_min_length)
    ax_fext = plt.axes([c4, 0.17, sw, sh]); s_fext = Slider(ax_fext, 'f_ext (P1)', 100, 1500, valinit=cfg.f_ext)
    ax_fcomp = plt.axes([c4, 0.13, sw, sh]); s_fcomp = Slider(ax_fcomp, 'f_comp (P2)', 100, 2000, valinit=cfg.f_comp)

    def update(val):
        cfg.chassis_piston_anchor = np.array([s_cx.val, s_cy.val])
        cfg.piston_mount_on_door = np.array([s_mx.val, s_my.val])
        cfg.center_of_mass_on_door = np.array([s_comx.val, s_comy.val])
        cfg.door_length = s_length.val
        cfg.door_mass_kg = s_mass.val
        cfg.strut_max_length = s_maxl.val
        cfg.strut_min_length = s_maxl.val - s_stroke.val
        cfg.f_ext = s_fext.val
        cfg.f_comp = s_fcomp.val

        ax1.clear(); ax2.clear(); ax3.clear()

        angles_deg = np.linspace(30, 120, 100)
        results = []

        # Motion Path Calculations
        for i, deg in enumerate(angles_deg):
            rad = np.radians(deg)
            p_door_xy = get_rotated_point(cfg.piston_mount_on_door, rad)
            p_vec = p_door_xy - cfg.chassis_piston_anchor
            L = np.linalg.norm(p_vec)

            # Use separate function for torque consistency
            from math import copysign # to help find zero crossing later

            p_unit = p_vec / L
            if L >= cfg.strut_max_length: f_piston = cfg.f_ext
            elif L <= cfg.strut_min_length: f_piston = cfg.f_comp
            else:
                pct = (cfg.strut_max_length - L) / (cfg.strut_max_length - cfg.strut_min_length)
                f_piston = cfg.f_ext + (cfg.f_comp - cfg.f_ext) * pct

            tp = np.cross(p_door_xy, p_unit * f_piston)
            p_com_xy = get_rotated_point(cfg.center_of_mass_on_door, rad)
            tg = np.cross(p_com_xy, np.array([0, -cfg.door_mass_kg * cfg.gravity_constant]))
            net = tp + tg

            res = {'deg': deg, 'net': net, 'valid': cfg.strut_min_length <= L <= cfg.strut_max_length}
            results.append(res)

            if i in [0, 50, 99]:
                draw_complete_geometry(ax1, cfg, rad, alpha=0.5 if res['valid'] else 0.1, title="Motion Path")

        # Equilibrium Search
        nets = [r['net'] for r in results]
        zero_angle = None
        for i in range(len(nets)-1):
            if np.sign(nets[i]) != np.sign(nets[i+1]):
                zero_angle = angles_deg[i] - nets[i] * (angles_deg[i+1] - angles_deg[i]) / (nets[i+1] - nets[i])
                break

        if zero_angle:
            draw_complete_geometry(ax3, cfg, np.radians(zero_angle), title=f"Equilibrium: {zero_angle:.1f}°")
        else:
            ax3.text(0.5, 0.5, "No Balance Point Found", ha='center', transform=ax3.transAxes)
            ax3.set_title("Equilibrium State")

        # Torque Plot
        ax2.plot(angles_deg, nets, 'k-')
        ax2.axhline(0, color='black', lw=1)
        ax2.fill_between(angles_deg, 0, nets, where=np.array(nets)>0, color='green', alpha=0.3)
        ax2.fill_between(angles_deg, 0, nets, where=np.array(nets)<0, color='red', alpha=0.3)
        ax2.set_title("Torque Chart")

        fig.canvas.draw_idle()

    all_sliders = [s_cx, s_cy, s_mx, s_my, s_length, s_mass, s_comx, s_comy, s_maxl, s_stroke, s_fext, s_fcomp]
    for s in all_sliders:
        s.on_changed(update)

    update(None)
    plt.show()

if __name__ == "__main__":
    run_interactive_simulation(SimulationConfig())