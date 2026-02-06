import numpy as np

class SimulationConfig:
    """
    Holds all physical and geometric parameters for the hatchback simulation.
    Now using simplified Extension and Compression force parameters.
    """

    def __init__(self,
                 chassis_piston_anchor=np.array([-0.3, -0.2]),
                 piston_mount_on_door=np.array([0.4, 0.0]),
                 center_of_mass_on_door=np.array([0.5, 0.0]),
                 door_length=1.0,
                 door_mass_kg=20.0,
                 strut_max_length=0.6,
                 strut_stroke=0.2,
                 extension_force_n=500.0,   # Force at Max Length
                 compression_force_n=650.0  # Force at Min Length
                 ):
        self.chassis_piston_anchor = chassis_piston_anchor
        self.piston_mount_on_door = piston_mount_on_door
        self.center_of_mass_on_door = center_of_mass_on_door
        self.door_length = door_length
        self.door_mass_kg = door_mass_kg
        self.strut_max_length = strut_max_length
        self.strut_min_length = strut_max_length - strut_stroke
        self.f_ext = extension_force_n   # P1 equivalent
        self.f_comp = compression_force_n # P2 equivalent
        self.gravity_constant = 9.81