import numpy as np
from Config import SimulationConfig


class SimulationResult:
    """
    Holds the computed physical state of the hatchback system across its range of motion.
    """

    def __init__(self):
        self.angles_deg = []
        self.net_torques = []
        self.hinge_forces = []
        self.is_valid_mask = []  # True if strut is within length limits
        self.equilibrium_angle = None
        self.max_physical_angle = 0.0

    def __repr__(self):
        eq_str = f"{self.equilibrium_angle:.2f}°" if self.equilibrium_angle else "None"
        return (f"<SimulationResult: Max Angle={self.max_physical_angle:.1f}°, "
                f"Equilibrium={eq_str}>")


class HatchbackPhysicsEngine:
    """
    Computes the mechanical forces and torques for a given SimulationConfig
    without graphical overhead.
    """

    @staticmethod
    def get_rotated_point(local_coords, theta_rad):
        """Transforms local door coordinates to global [x, y] based on rotation theta."""
        along, depth = local_coords
        x = along * np.sin(theta_rad) + depth * np.cos(theta_rad)
        y = -along * np.cos(theta_rad) + depth * np.sin(theta_rad)
        return np.array([x, y])

    def calculate_max_angle(self, cfg):
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

        return np.clip(np.degrees(max_rad), 5.0, 180.0)

    def run(self, cfg, steps=100):
        """
        Executes the simulation and returns a SimulationResult object.
        """
        res = SimulationResult()
        res.max_physical_angle = self.calculate_max_angle(cfg)

        angles = np.linspace(cfg.door_close_angle_deg, res.max_physical_angle, steps)

        for deg in angles:
            rad = np.radians(deg)

            # 1. Geometry & Strut Length
            p_door_xy = self.get_rotated_point(cfg.piston_mount_on_door, rad)
            p_vec = p_door_xy - cfg.chassis_piston_anchor
            L = np.linalg.norm(p_vec)

            is_valid = cfg.strut_min_length <= L <= cfg.strut_max_length

            # 2. Piston Force Calculation
            p_unit = p_vec / L
            if L >= cfg.strut_max_length:
                f_piston_mag = cfg.f_ext
            elif L <= cfg.strut_min_length:
                f_piston_mag = cfg.f_comp
            else:
                pct = (cfg.strut_max_length - L) / (cfg.strut_max_length - cfg.strut_min_length)
                f_piston_mag = cfg.f_ext + (cfg.f_comp - cfg.f_ext) * pct

            # 3. Torque Calculation
            f_strut_vec = p_unit * f_piston_mag
            f_grav_vec = np.array([0, -cfg.door_mass_kg * cfg.gravity_constant])

            tp = np.cross(p_door_xy, f_strut_vec)
            p_com_xy = self.get_rotated_point(cfg.center_of_mass_on_door, rad)
            tg = np.cross(p_com_xy, f_grav_vec)
            net_torque = tp + tg

            # 4. Hinge Stress
            f_hinge_mag = np.linalg.norm(f_strut_vec + f_grav_vec)

            # Store Data
            res.angles_deg.append(deg)
            res.net_torques.append(net_torque)
            res.hinge_forces.append(f_hinge_mag)
            res.is_valid_mask.append(is_valid)

        # Find Equilibrium (Zero-crossing of net torque)
        for i in range(len(res.net_torques) - 1):
            if np.sign(res.net_torques[i]) != np.sign(res.net_torques[i + 1]):
                # Linear interpolation for more precise angle
                t1, t2 = res.net_torques[i], res.net_torques[i + 1]
                a1, a2 = res.angles_deg[i], res.angles_deg[i + 1]
                res.equilibrium_angle = a1 - t1 * (a2 - a1) / (t2 - t1)
                break

        return res


if __name__ == "__main__":
    # Example Usage:
    config = SimulationConfig()
    engine = HatchbackPhysicsEngine()
    result = engine.run(config)

    print(f"Simulation Complete.")
    print(f"Max Opening Angle: {result.max_physical_angle:.2f}°")
    if result.equilibrium_angle:
        print(f"Door stays open at: {result.equilibrium_angle:.2f}°")
    else:
        print("No equilibrium point found; door will likely fall or stay pressurized.")