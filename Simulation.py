import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from Config import SimulationConfig


def get_rotated_point(local_coords, theta):
    """Transforms local door coordinates to global [x, y] based on rotation theta."""
    along, depth = local_coords
    x = along * np.sin(theta) + depth * np.cos(theta)
    y = -along * np.cos(theta) + depth * np.sin(theta)
    return np.array([x, y])


def draw_complete_geometry(ax, cfg, rad, alpha=1.0, title="", show_labels=True):
    """
    Renders the door with all technical indicators:
    Arrows, Torque Labels, Mounting Points, CoM, and Hinge Forces.
    Uses the provided alpha to indicate validity (full for valid, faded for invalid).
    """
    p_door_xy = get_rotated_point(cfg.piston_mount_on_door, rad)
    p_com_xy = get_rotated_point(cfg.center_of_mass_on_door, rad)
    door_end_xy = np.array([cfg.door_length * np.sin(rad), -cfg.door_length * np.cos(rad)])

    # Physics Calculation for indicators
    p_vec = p_door_xy - cfg.chassis_piston_anchor
    L = np.linalg.norm(p_vec)
    p_unit = p_vec / L

    # Determine force and validity for indicators
    if L >= cfg.strut_max_length:
        f_piston = cfg.f_ext
    elif L <= cfg.strut_min_length:
        f_piston = cfg.f_comp
    else:
        pct = (cfg.strut_max_length - L) / (cfg.strut_max_length - cfg.strut_min_length)
        f_piston = cfg.f_ext + (cfg.f_comp - cfg.f_ext) * pct

    # Forces for Hinge Calculation
    f_strut_vec = p_unit * f_piston
    f_grav_vec = np.array([0, -cfg.door_mass_kg * cfg.gravity_constant])

    # Hinge acting force
    f_hinge_vec = (f_strut_vec + f_grav_vec)
    f_hinge_mag = np.linalg.norm(f_hinge_vec)

    tp = np.cross(p_door_xy, f_strut_vec)
    tg = np.cross(p_com_xy, f_grav_vec)
    net = tp + tg
    color = 'green' if net > 0 else 'red'

    # 1. Door Rendering
    t_arc = np.linspace(0, np.pi, 20)
    arc_points = np.array([get_rotated_point([x, 0.15 * np.sin(t)], rad)
                           for x, t in zip(np.linspace(0, cfg.door_length, 20), t_arc)])
    ax.plot(arc_points[:, 0], arc_points[:, 1], 'k-', alpha=alpha, lw=2.5)

    # 2. Strut and Anchor
    ax.plot([cfg.chassis_piston_anchor[0], p_door_xy[0]],
            [cfg.chassis_piston_anchor[1], p_door_xy[1]], '--', color=color, alpha=alpha * 0.6)
    ax.scatter(*cfg.chassis_piston_anchor, color='black', s=60, zorder=10, alpha=alpha)

    # 3. Mount and CoM Dots
    ax.scatter(*p_door_xy, color='darkorange', s=35, zorder=5, alpha=alpha)
    ax.scatter(*p_com_xy, color='blue', s=35, zorder=5, alpha=alpha)

    # 4. Torque and Force Labels (Toggleable)
    if show_labels:
        ax.text(p_door_xy[0], p_door_xy[1] + 0.04, f"P:{tp:.0f}", fontsize=8, color='darkorange', alpha=alpha)
        ax.text(p_com_xy[0], p_com_xy[1] - 0.08, f"G:{tg:.0f}", fontsize=8, color='blue', alpha=alpha)

    # 5. Torque Arrow (at door end)
    perp_dir = np.array([np.cos(rad), np.sin(rad)])
    dx, dy = perp_dir * (net / 400.0)
    ax.arrow(door_end_xy[0], door_end_xy[1], dx, dy, head_width=0.04, color=color, alpha=alpha * 0.8)

    # 6. Hinge Force Arrow (at pivot [0,0])
    h_dx, h_dy = f_hinge_vec / 1000.0  # Scaled for visibility
    ax.arrow(0, 0, h_dx, h_dy, head_width=0.03, color='purple', alpha=alpha * 0.5, label='Hinge Force')

    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 2.0)
    ax.set_ylim(-2.0, 1.5)
    ax.set_title(title)


def run_interactive_simulation(cfg):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 9))
    plt.subplots_adjust(bottom=0.4, wspace=0.3)

    ax2b = ax2.twinx()

    # State variable for toggles
    state = {'show_labels': True}

    # --- SLIDERS ---
    sw, sh = 0.12, 0.02
    c1, c2, c3, c4 = 0.05, 0.28, 0.52, 0.76

    ax_cx = plt.axes([c1, 0.25, sw, sh])
    s_cx = Slider(ax_cx, 'Chassis X', -1.5, 0.5, valinit=cfg.chassis_piston_anchor[0])
    ax_cy = plt.axes([c1, 0.21, sw, sh])
    s_cy = Slider(ax_cy, 'Chassis Y', -1.5, 0.5, valinit=cfg.chassis_piston_anchor[1])

    ax_mx = plt.axes([c2, 0.25, sw, sh])
    s_mx = Slider(ax_mx, 'Mount X', 0.1, 1.5, valinit=cfg.piston_mount_on_door[0])
    ax_my = plt.axes([c2, 0.21, sw, sh])
    s_my = Slider(ax_my, 'Mount Y', -0.3, 0.3, valinit=cfg.piston_mount_on_door[1])
    ax_length = plt.axes([c2, 0.17, sw, sh])
    s_length = Slider(ax_length, 'Door Len', 0.5, 2.0, valinit=cfg.door_length)

    ax_mass = plt.axes([c3, 0.25, sw, sh])
    s_mass = Slider(ax_mass, 'Mass (kg)', 5.0, 80.0, valinit=cfg.door_mass_kg)
    ax_comx = plt.axes([c3, 0.21, sw, sh])
    s_comx = Slider(ax_comx, 'CoM X', 0.1, 1.5, valinit=cfg.center_of_mass_on_door[0])
    ax_comy = plt.axes([c3, 0.17, sw, sh])
    s_comy = Slider(ax_comy, 'CoM Y', -0.3, 0.3, valinit=cfg.center_of_mass_on_door[1])

    ax_maxl = plt.axes([c4, 0.25, sw, sh])
    s_maxl = Slider(ax_maxl, 'Max L', 0.1, 4.0, valinit=cfg.strut_max_length)
    ax_stroke = plt.axes([c4, 0.21, sw, sh])
    s_stroke = Slider(ax_stroke, 'Stroke', 0.05, 3.0, valinit=cfg.strut_max_length - cfg.strut_min_length)
    ax_fext = plt.axes([c4, 0.17, sw, sh])
    s_fext = Slider(ax_fext, 'f_ext (P1)', 100, 1500, valinit=cfg.f_ext)
    ax_fcomp = plt.axes([c4, 0.13, sw, sh])
    s_fcomp = Slider(ax_fcomp, 'f_comp (P2)', 100, 2000, valinit=cfg.f_comp)

    # --- BUTTONS ---
    ax_btn_print = plt.axes([0.38, 0.05, 0.1, 0.04])
    btn_print = Button(ax_btn_print, 'Print Stats', color='lightgray', hovercolor='0.95')

    ax_btn_toggle = plt.axes([0.52, 0.05, 0.1, 0.04])
    btn_toggle = Button(ax_btn_toggle, 'Toggle Labels', color='lightgray', hovercolor='0.95')

    def calculate_max_angle(cfg):
        """Calculates the physical limit of the door based on piston max length."""
        a = np.linalg.norm(cfg.chassis_piston_anchor)
        b = np.linalg.norm(cfg.piston_mount_on_door)
        c = cfg.strut_max_length

        if c >= a + b or c <= abs(a - b):
            return 180.0

        cos_gamma = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
        gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
        anchor_angle = np.arctan2(cfg.chassis_piston_anchor[0], -cfg.chassis_piston_anchor[1])
        mount_local_angle = np.arctan2(cfg.piston_mount_on_door[0], -cfg.piston_mount_on_door[1])
        max_rad = anchor_angle + gamma - mount_local_angle
        max_deg = np.degrees(max_rad)

        return np.clip(max_deg, 5.0, 180.0)

    def print_snapshot_info(event):
        max_angle = calculate_max_angle(cfg)
        angles_deg = np.linspace(5.0, max_angle, 100)
        print("\n--- Current Mechanical Snapshot ---")
        for i in [0, 50, 99]:
            deg = angles_deg[i]
            p_door_xy = get_rotated_point(cfg.piston_mount_on_door, np.radians(deg))
            L = np.linalg.norm(p_door_xy - cfg.chassis_piston_anchor)
            valid = "VALID" if cfg.strut_min_length <= L <= cfg.strut_max_length else "INVALID"
            print(f"Angle: {deg:5.1f}° | Status: {valid} | L: {L:.3f} (Limits: {cfg.strut_min_length:.3f}-{cfg.strut_max_length:.3f})")

    def toggle_labels(event):
        state['show_labels'] = not state['show_labels']
        update(None)

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

        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax2b.clear()

        max_angle = calculate_max_angle(cfg)
        angles_deg = np.linspace(cfg.door_close_angle_deg, max_angle, 100)
        results = []

        for i, deg in enumerate(angles_deg):
            rad = np.radians(deg)
            p_door_xy = get_rotated_point(cfg.piston_mount_on_door, rad)
            p_vec = p_door_xy - cfg.chassis_piston_anchor
            L = np.linalg.norm(p_vec)
            is_valid = cfg.strut_min_length <= L <= cfg.strut_max_length

            p_unit = p_vec / L
            if L >= cfg.strut_max_length:
                f_piston = cfg.f_ext
            elif L <= cfg.strut_min_length:
                f_piston = cfg.f_comp
            else:
                pct = (cfg.strut_max_length - L) / (cfg.strut_max_length - cfg.strut_min_length)
                f_piston = cfg.f_ext + (cfg.f_comp - cfg.f_ext) * pct

            f_strut_vec = p_unit * f_piston
            f_grav_vec = np.array([0, -cfg.door_mass_kg * cfg.gravity_constant])
            f_hinge_mag = np.linalg.norm(f_strut_vec + f_grav_vec)

            tp = np.cross(p_door_xy, f_strut_vec)
            p_com_xy = get_rotated_point(cfg.center_of_mass_on_door, rad)
            tg = np.cross(p_com_xy, f_grav_vec)
            net = tp + tg

            res = {'deg': deg, 'net': net, 'h_force': f_hinge_mag, 'valid': is_valid}
            results.append(res)

            if i in [0, 50, 99]:
                snap_alpha = 0.6 if is_valid else 0.15
                draw_complete_geometry(ax1, cfg, rad, alpha=snap_alpha, title=f"Motion Path (Max: {max_angle:.1f}°)", show_labels=state['show_labels'])

        nets = [r['net'] for r in results]
        zero_angle = None
        for i in range(len(nets) - 1):
            if np.sign(nets[i]) != np.sign(nets[i + 1]):
                zero_angle = angles_deg[i] - nets[i] * (angles_deg[i + 1] - angles_deg[i]) / (nets[i + 1] - nets[i])
                break

        if zero_angle:
            rad_eq = np.radians(zero_angle)
            p_door_eq = get_rotated_point(cfg.piston_mount_on_door, rad_eq)
            L_eq = np.linalg.norm(p_door_eq - cfg.chassis_piston_anchor)
            valid_eq = cfg.strut_min_length <= L_eq <= cfg.strut_max_length
            eq_alpha = 1.0 if valid_eq else 0.2
            draw_complete_geometry(ax3, cfg, rad_eq, alpha=eq_alpha, title=f"Equilibrium: {zero_angle:.1f}°", show_labels=state['show_labels'])
        else:
            ax3.text(0.5, 0.5, "No Balance Point Found", ha='center', transform=ax3.transAxes)
            ax3.set_title("Equilibrium State")

        ax2.plot(angles_deg, nets, 'k-', label='Net Torque')
        ax2.axhline(0, color='black', lw=1)
        ax2.fill_between(angles_deg, 0, nets, where=np.array(nets) > 0, color='green', alpha=0.3)
        ax2.fill_between(angles_deg, 0, nets, where=np.array(nets) < 0, color='red', alpha=0.3)

        valid_mask = np.array([r['valid'] for r in results])
        ax2.fill_between(angles_deg, 0, 1, where=~valid_mask, color='red', alpha=0.15, transform=ax2.get_xaxis_transform())

        ax2b.plot(angles_deg, [r['h_force'] for r in results], 'p--', alpha=0.3)
        ax2b.set_ylabel("Hinge Force (N)", color='purple', fontsize=8)
        ax2b.tick_params(axis='y', labelcolor='purple')
        ax2.set_title("Torque & Hinge Stress")

        fig.canvas.draw_idle()

    all_sliders = [s_cx, s_cy, s_mx, s_my, s_length, s_mass, s_comx, s_comy, s_maxl, s_stroke, s_fext, s_fcomp]
    for s in all_sliders:
        s.on_changed(update)

    btn_print.on_clicked(print_snapshot_info)
    btn_toggle.on_clicked(toggle_labels)

    update(None)
    plt.show()


if __name__ == "__main__":
    run_interactive_simulation(SimulationConfig())