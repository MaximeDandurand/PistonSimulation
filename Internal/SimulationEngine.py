import numpy as np
from Internal.Config import SimulationConfig, SimulationConstraint


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

        self.simulation_finished = True


class HatchbackPhysicsEngine:
    @staticmethod
    def get_rotated_point(local_coords, theta_rad):
        along, depth = local_coords
        x = along * np.sin(theta_rad) + depth * np.cos(theta_rad)
        y = -along * np.cos(theta_rad) + depth * np.sin(theta_rad)
        return np.array([x, y])

    def calculate_max_angle(self, cfg):
        a_vec = cfg.chassis_piston_anchor_meter
        b_vec = cfg.piston_mount_on_door_meter

        a = np.linalg.norm(a_vec)
        b = np.linalg.norm(b_vec)
        c = cfg.strut_max_length

        # Check for invalid geometry
        if c >= (a + b) or c <= abs(a - b):
            # Log this or handle appropriately, do not return 180
            return 180

        cos_gamma = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
        gamma = np.arccos(np.clip(cos_gamma, -1.0, 1.0))

        # Calculate angular positions relative to the coordinate system
        anchor_angle = np.arctan2(a_vec[0], -a_vec[1])
        mount_local_angle = np.arctan2(b_vec[0], -b_vec[1])

        # Calculate the resulting angle
        # Note: Depending on mounting, you may need (anchor_angle - gamma - mount_local_angle)
        final_angle = np.degrees(anchor_angle + gamma - mount_local_angle)

        return np.clip(final_angle, cfg.door_close_angle_deg, 180.0)

    def get_state_at_angle(self, cfg, deg):
        """Returns all geometric and physics data for a single frame."""
        rad = np.radians(deg)
        p_door_xy = self.get_rotated_point(cfg.piston_mount_on_door_meter, rad)
        p_com_xy = self.get_rotated_point(cfg.center_of_mass_on_door, rad)
        p_end_xy = np.array([cfg.door_length * np.sin(rad), -cfg.door_length * np.cos(rad)])

        p_vec = p_door_xy - cfg.chassis_piston_anchor_meter
        L = np.linalg.norm(p_vec)
        p_unit = p_vec / L

        if L >= cfg.strut_max_length:
            f_piston = cfg.f_ext
        elif L <= cfg.strut_min_length:
            f_piston = cfg.f_comp
        else:
            pct = (cfg.strut_max_length - L) / (cfg.strut_max_length - cfg.strut_min_length)
            f_piston = cfg.f_ext + (cfg.f_comp - cfg.f_ext) * pct

        f_strut_v = p_unit * f_piston * cfg.number_piston
        f_grav_v = np.array([0, -cfg.door_mass_kg * cfg.gravity_constant])
        net_t = np.cross(p_door_xy, f_strut_v) + np.cross(p_com_xy, f_grav_v)

        return {
            'deg': deg, 'mount': p_door_xy, 'com': p_com_xy, 'end': p_end_xy,
            'hinge_v': (f_strut_v + f_grav_v) / 1000.0, 'net': net_t,
            'h_force_mag': np.linalg.norm(f_strut_v + f_grav_v),
            'color': 'green' if net_t > 0 else 'red',
            'valid': cfg.strut_min_length <= L <= cfg.strut_max_length
        }

    def run(self, cfg:SimulationConfig, steps=100, constraints:SimulationConstraint=None):
        res = SimulationResult()
        res.max_physical_angle = self.calculate_max_angle(cfg)

        if constraints is not None:
            valid = True
            if res.max_physical_angle == 180: valid = False
            if res.max_physical_angle > constraints.open_max_angle_deg: valid = False
            if res.max_physical_angle < constraints.open_min_angle_deg: valid = False
            if not valid:
                res.simulation_finished = False
                return res

        angles = np.linspace(cfg.door_close_angle_deg, res.max_physical_angle, steps)

        #Run first simulation at closing angle
        closed_angle_simulation = self.get_state_at_angle(cfg, cfg.door_close_angle_deg)

        res.angles_deg.append(cfg.door_close_angle_deg)
        res.net_torques.append(closed_angle_simulation['net'])
        res.hinge_forces.append(closed_angle_simulation['h_force_mag'])
        res.is_valid_mask.append(closed_angle_simulation['valid'])
        res.mount_coords.append(closed_angle_simulation['mount'])
        res.com_coords.append(closed_angle_simulation['com'])
        res.door_end_coords.append(closed_angle_simulation['end'])
        res.hinge_vectors.append(closed_angle_simulation['hinge_v'])
        res.strut_colors.append(closed_angle_simulation['color'])
        if constraints is not None:
            valid = True
            if abs(closed_angle_simulation['h_force_mag']) > constraints.max_hinge_torque: valid = False
            if abs(closed_angle_simulation['net']) > constraints.max_opening_torque: valid = False
            if abs(closed_angle_simulation['net']) < constraints.min_opening_torque: valid = False
            if constraints.need_negative_closed_torque and closed_angle_simulation['net'] >= 0: valid = False
            if not closed_angle_simulation['valid']: valid = False
            if not valid:
                res.simulation_finished = False
                return res

        #Run simulation at opened angle
        opened_angle_simulation = self.get_state_at_angle(cfg, res.max_physical_angle)
        if constraints is not None:
            valid = True
            if abs(opened_angle_simulation['h_force_mag']) > constraints.max_hinge_torque: valid = False
            if abs(opened_angle_simulation['net']) > constraints.max_closing_torque: valid = False
            if abs(opened_angle_simulation['net']) < constraints.min_closing_torque: valid = False
            if constraints.need_positive_opened_torque and opened_angle_simulation['net'] <= 0: valid = False
            if not opened_angle_simulation['valid']: valid = False
            if not valid:
                res.simulation_finished = False
                return res

        for deg in angles:
            if deg == cfg.door_close_angle_deg or deg == res.max_physical_angle: continue
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

            if constraints is not None:
                valid = True
                if s['h_force_mag'] > constraints.max_hinge_torque: valid = False
                if not s['valid']: valid = False
                if not valid:
                    res.simulation_finished = False
                    return res

        #Add the simulation of the fully opened door
        res.angles_deg.append(res.max_physical_angle)
        res.net_torques.append(opened_angle_simulation['net'])
        res.hinge_forces.append(opened_angle_simulation['h_force_mag'])
        res.is_valid_mask.append(opened_angle_simulation['valid'])
        res.mount_coords.append(opened_angle_simulation['mount'])
        res.com_coords.append(opened_angle_simulation['com'])
        res.door_end_coords.append(opened_angle_simulation['end'])
        res.hinge_vectors.append(opened_angle_simulation['hinge_v'])
        res.strut_colors.append(opened_angle_simulation['color'])

        for i in range(len(res.net_torques) - 1):
            if np.sign(res.net_torques[i]) != np.sign(res.net_torques[i + 1]):
                t1, t2, a1, a2 = res.net_torques[i], res.net_torques[i + 1], res.angles_deg[i], res.angles_deg[i + 1]
                res.equilibrium_angle = a1 - t1 * (a2 - a1) / (t2 - t1)
                break
        return res