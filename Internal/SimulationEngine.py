import numpy as np
from Internal.Config import SimulationConfig


class SimulationResult:
    def __init__(self):
        self.angles_deg = []
        self.net_torques = []
        self.hinge_forces = []  # Standardized name for EvaluationEngine compatibility
        self.is_valid_mask = []
        self.equilibrium_angle = None
        self.max_physical_angle = 0.0

        # Pre-calculated geometric data for SimulationGraph
        self.mount_coords = []
        self.com_coords = []
        self.door_end_coords = []
        self.hinge_vectors = []
        self.strut_colors = []


class HatchbackPhysicsEngine:
    @staticmethod
    def get_rotated_point(local_coords, theta_rad):
        along, depth = local_coords
        x = along * np.sin(theta_rad) + depth * np.cos(theta_rad)
        y = -along * np.cos(theta_rad) + depth * np.sin(theta_rad)
        return np.array([x, y])

    def calculate_max_angle(self, cfg):
        a = np.linalg.norm(cfg.chassis_piston_anchor)
        b = np.linalg.norm(cfg.piston_mount_on_door)
        c = cfg.strut_max_length
        if c >= a + b or c <= abs(a - b): return 180.0
        cos_gamma = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
        gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))
        anchor_angle = np.arctan2(cfg.chassis_piston_anchor[0], -cfg.chassis_piston_anchor[1])
        mount_local_angle = np.arctan2(cfg.piston_mount_on_door[0], -cfg.piston_mount_on_door[1])
        return np.clip(np.degrees(anchor_angle + gamma - mount_local_angle), 5.0, 180.0)

    def get_state_at_angle(self, cfg, deg):
        """Returns all geometric and physics data for a single frame."""
        rad = np.radians(deg)
        p_door_xy = self.get_rotated_point(cfg.piston_mount_on_door, rad)
        p_com_xy = self.get_rotated_point(cfg.center_of_mass_on_door, rad)
        p_end_xy = np.array([cfg.door_length * np.sin(rad), -cfg.door_length * np.cos(rad)])

        p_vec = p_door_xy - cfg.chassis_piston_anchor
        L = np.linalg.norm(p_vec)
        p_unit = p_vec / L

        if L >= cfg.strut_max_length:
            f_piston = cfg.f_ext
        elif L <= cfg.strut_min_length:
            f_piston = cfg.f_comp
        else:
            pct = (cfg.strut_max_length - L) / (cfg.strut_max_length - cfg.strut_min_length)
            f_piston = cfg.f_ext + (cfg.f_comp - cfg.f_ext) * pct

        f_strut_v = p_unit * f_piston
        f_grav_v = np.array([0, -cfg.door_mass_kg * cfg.gravity_constant])
        net_t = np.cross(p_door_xy, f_strut_v) + np.cross(p_com_xy, f_grav_v)

        return {
            'deg': deg, 'mount': p_door_xy, 'com': p_com_xy, 'end': p_end_xy,
            'hinge_v': (f_strut_v + f_grav_v) / 1000.0, 'net': net_t,
            'h_force_mag': np.linalg.norm(f_strut_v + f_grav_v),
            'color': 'green' if net_t > 0 else 'red',
            'valid': cfg.strut_min_length <= L <= cfg.strut_max_length
        }

    def run(self, cfg, steps=100):
        res = SimulationResult()
        res.max_physical_angle = self.calculate_max_angle(cfg)
        angles = np.linspace(cfg.door_close_angle_deg, res.max_physical_angle, steps)

        for deg in angles:
            s = self.get_state_at_angle(cfg, deg)
            res.angles_deg.append(deg)
            res.net_torques.append(s['net'])
            res.hinge_forces.append(s['h_force_mag'])
            res.is_valid_mask.append(s['valid'])
            res.mount_coords.append(s['mount'])
            res.com_coords.append(s['com'])
            res.door_end_coords.append(s['end'])
            res.hinge_vectors.append(s['hinge_v'])
            res.strut_colors.append(s['color'])

        for i in range(len(res.net_torques) - 1):
            if np.sign(res.net_torques[i]) != np.sign(res.net_torques[i + 1]):
                t1, t2, a1, a2 = res.net_torques[i], res.net_torques[i + 1], res.angles_deg[i], res.angles_deg[i + 1]
                res.equilibrium_angle = a1 - t1 * (a2 - a1) / (t2 - t1)
                break
        return res