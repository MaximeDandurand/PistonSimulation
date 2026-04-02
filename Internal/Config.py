import numpy as np
import numpy as np
from matplotlib.path import Path
from typing import List, Tuple, Union

class SimulationConfig:
    """
    Holds all physical and geometric parameters for the hatchback simulation.
    Now using simplified Extension and Compression force parameters.
    """

    def __init__(self,
                 chassis_piston_anchor=np.array([0.05, -0.4]),
                 piston_mount_on_door=np.array([0.2, -0.035]),
                 center_of_mass_on_door=np.array([0.77, 0.25]),
                 door_length=1.4260107632981374,
                 door_mass_kg=14.886813957918724,
                 strut_max_length=0.6,
                 strut_stroke=0.25,
                 extension_force_n=400.0,   # Force at Max Length
                 compression_force_n=500.0,  # Force at Min Length
                 door_close_angle_deg=8.318045652861889,# Starting position when closed
                 number_piston=2,
                 ground_height_offset=0.406,
                 middle_door_curve_offset=0.3666492136331058
                 ):
        self.chassis_piston_anchor_meter = chassis_piston_anchor #0,0 is the hinge,
        self.piston_mount_on_door_meter = piston_mount_on_door #Relative to the door orientation. 0,0 is still the hinge.
        self.center_of_mass_on_door = center_of_mass_on_door #Relative to the door orientation. 0,0 is still the hinge.
        self.door_length = door_length
        self.door_mass_kg = door_mass_kg
        self.strut_max_length = strut_max_length
        self.strut_min_length = strut_max_length - strut_stroke
        self.f_ext = extension_force_n   # P1 equivalent
        self.f_comp = compression_force_n # P2 equivalent
        self.gravity_constant = 9.81
        self.door_close_angle_deg = door_close_angle_deg
        self.number_piston = number_piston
        self.ground_height_offset = ground_height_offset
        self.middle_door_curve_offset = middle_door_curve_offset

class SimulationConstraint:
    def __init__(self,
                 open_max_angle_deg=145.0,
                 open_min_angle_deg=70.0,

                 max_opening_torque=100.0,
                 min_opening_torque=5.0,
                 max_closing_torque=100.0,
                 min_closing_torque=5.0,
                 max_hinge_torque=4000.0,
                 need_positive_opened_torque=True,
                 need_negative_closed_torque=True,

                 ):
        self.open_max_angle_deg = open_max_angle_deg
        self.open_min_angle_deg = open_min_angle_deg
        self.max_opening_torque = max_opening_torque
        self.min_opening_torque = min_opening_torque
        self.max_closing_torque = max_closing_torque
        self.min_closing_torque = min_closing_torque
        self.max_hinge_torque = max_hinge_torque
        self.need_positive_opened_torque = need_positive_opened_torque
        self.need_negative_closed_torque = need_negative_closed_torque





class MountingArea:
    def __init__(self, vertices: Union[List[np.ndarray], List[Tuple[float, float]]]):
        """
        Initializes the MountingArea with a set of vertices.
        Works with both lists of numpy arrays and lists of tuples.
        """
        # Convert to numpy array once to handle different input types and find bounds
        v_np = np.array(vertices)
        self.vertices = v_np
        self.path = Path(v_np)

        # Calculate bounding box for grid generation
        self.x_min, self.y_min = v_np.min(axis=0)
        self.x_max, self.y_max = v_np.max(axis=0)

    def contains(self, point: np.ndarray) -> bool:
        """Check if a single point is inside the mounting area."""
        return self.path.contains_point(point)

    def get_valid_points(self, resolution: float) -> List[np.ndarray]:
        """Generate a grid of points within the bounding box that fall inside the path."""
        x_range = np.arange(self.x_min, self.x_max, resolution)
        y_range = np.arange(self.y_min, self.y_max, resolution)

        valid_points: List[np.ndarray] = []
        for x in x_range:
            for y in y_range:
                point = np.array([x, y])
                if self.contains(point):  # Reuse the contains method
                    valid_points.append(point)
        return valid_points

    def __str__(self) -> str:
        """
        Returns a string representation equivalent to the C# ToString override.
        """
        coords = ", ".join([f"({v[0]:.2f}, {v[1]:.2f})" for v in self.vertices])
        return (f"MountingArea | Vertices: [{coords}] | "
                f"Bounds: [X: {self.x_min:.2f} to {self.x_max:.2f}, Y: {self.y_min:.2f} to {self.y_max:.2f}]")
class PistonSpec:
    def __init__(self, name: str, max_length: float, stroke: float, f_ext: float, f_comp: float):
        self.name = name
        self.max_length = max_length
        self.stroke = stroke
        self.f_ext = f_ext
        self.f_comp = f_comp
        self.min_length = max_length- stroke
    def print(self):
        print(f"{self.name}: length = {self.max_length}, stroke = {self.stroke}, f_ext = {self.f_ext}, f_comp = {self.f_comp}")
