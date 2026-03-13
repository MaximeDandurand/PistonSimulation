import math
from typing import Tuple, List, Union
import numpy as np
from Internal.Config import SimulationConfig, MountingArea, PistonSpec
from Internal.piston import ALL_MCMASTER_PISTONS

IDEAL_PISTON_LENGTH_RATIO = 0.6
ESTIMATED_STROKE_RATIO = 0.4
DISTANCE_HINGE_METER_RATIO = 0.85
SAFETY_FACTOR = 1
NEWTON_TO_POUNDS = 0.2248
def get_optimal_start(cfg: SimulationConfig, opened_angle: float, closed_angle: float, area: MountingArea):
    # Base geometry calculations
    ideal_piston_length_meter = cfg.door_length * IDEAL_PISTON_LENGTH_RATIO
    estimated_stroke_meter = ideal_piston_length_meter * ESTIMATED_STROKE_RATIO

    # distance_hinge is already relative to the hinge (the edge)
    distance_hinge_meter = estimated_stroke_meter * DISTANCE_HINGE_METER_RATIO

    dead_weight_newton = calculate_lifting_force_at_middle(cfg, 45)
    force_require_newton = (dead_weight_newton * get_com_distance_from_hinge(cfg)) / (distance_hinge_meter * 2)
    force_safety_factor_newton = force_require_newton * SAFETY_FACTOR
    force_require_total_newton = force_require_newton + force_safety_factor_newton
    force_pound = force_require_total_newton * NEWTON_TO_POUNDS
    # Find the closest Force Category
    available_forces = list(ALL_MCMASTER_PISTONS.keys())
    best_match_force = min(available_forces, key=lambda x: abs(x - force_pound))
    candidate_pistons = ALL_MCMASTER_PISTONS[best_match_force]

    best_piston = get_top_n_pistons(cfg=cfg, n=1)[0]

    # Door anchor point relative to hinge at OPENED angle
    rad_open = math.radians(opened_angle)
    dx = distance_hinge_meter * math.sin(rad_open)
    dy = -distance_hinge_meter * math.cos(rad_open)
    door_anchor_at_angle = [dx, dy]

    # Door anchor point relative to hinge at CLOSED angle
    rad_closed = math.radians(closed_angle)
    closed_dx = distance_hinge_meter * math.sin(rad_closed)
    closed_dy = -distance_hinge_meter * math.cos(rad_closed)

    possible_frame_points = []
    radius = best_piston.max_length

    # Iterate through the circle of possible points
    for degree in range(360):
        phi = math.radians(degree)
        fx = dx + radius * math.cos(phi)
        fy = dy + radius * math.sin(phi)
        f_point = np.array([fx, fy])

        if not area.contains(f_point):
            continue

        possible_frame_points.append(f_point)

    # Since coordinates are now relative to the hinge,
    # the local anchor point is simply the distance along the door.
    door_anchor_local = [distance_hinge_meter, 0]
    return best_piston, door_anchor_local, possible_frame_points


def get_top_n_pistons(cfg: SimulationConfig, n: int = 5):
    """
    Evaluates all available pistons and returns the top N candidates
    """
    # 1. Calculate ideal targets (consistent with original logic)
    ideal_piston_length_meter = cfg.door_length * IDEAL_PISTON_LENGTH_RATIO
    estimated_stroke_meter = ideal_piston_length_meter * ESTIMATED_STROKE_RATIO
    distance_hinge_meter = estimated_stroke_meter * DISTANCE_HINGE_METER_RATIO

    dead_weight_newton = calculate_lifting_force_at_middle(cfg, 45)
    force_require_newton = (dead_weight_newton * get_com_distance_from_hinge(cfg)) / (distance_hinge_meter * 2)
    force_safety_factor_newton = force_require_newton * SAFETY_FACTOR
    force_require_total_newton = force_require_newton + force_safety_factor_newton

    target_force_pound = force_require_total_newton * NEWTON_TO_POUNDS

    scored_pistons = []

    # 2. Iterate through all force categories and all pistons
    for force_cat, pistons in ALL_MCMASTER_PISTONS.items():
        # Penalty for force deviation
        force_error = abs(force_cat - target_force_pound)

        for p in pistons:
            # Geometric errors
            length_error = abs(p.max_length * 0.0254 - ideal_piston_length_meter)
            stroke_error = abs(p.stroke * 0.0254 - estimated_stroke_meter)

            # Heavy penalty if the stroke is physically too short
            if p.stroke * 0.0254 < estimated_stroke_meter:
                stroke_error *= 5

                # Composite score (Lower is better)
            # We weight force error heavily to ensure the piston can actually lift the door
            total_score = (force_error * 1.5) + length_error + stroke_error

            # Convert to metric for the return list
            metric_piston = PistonSpec(
                p.name,
                p.max_length * 0.0254,
                p.stroke * 0.0254,
                p.f_ext * 4.44822,
                p.f_comp * 4.44822
            )

            scored_pistons.append((total_score, metric_piston))

    # 3. Sort by score and return the top N
    scored_pistons.sort(key=lambda x: x[0])

    return [p for score, p in scored_pistons[:n]]

def get_com_distance_from_hinge(config: SimulationConfig):
    """
    Calculates the absolute distance from the hinge (0,0) to the Center of Mass.
    Now assumes center_of_mass_on_door is already relative to the hinge (the edge).
    """
    # We no longer need the mid_point_offset shift.
    x_from_hinge = config.center_of_mass_on_door[0]
    y_from_hinge = config.center_of_mass_on_door[1]

    return np.sqrt(x_from_hinge ** 2 + y_from_hinge ** 2)


def calculate_lifting_force_at_middle(config: SimulationConfig, current_angle_deg: float):
    """
    Calculates the perpendicular force required at the middle of the door
    to hold it at a specific angle.
    """
    theta = np.radians(current_angle_deg)

    # Rotation Matrix
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([
        [c, -s],
        [s, c]
    ])

    # Transform CoM (already relative to hinge) to current orientation
    com_rotated = rotation_matrix.dot(config.center_of_mass_on_door)

    # Horizontal distance (Lever Arm)
    lever_arm_gravity = abs(com_rotated[0])

    # Gravity Torque
    weight = config.door_mass_kg * config.gravity_constant
    torque_gravity = weight * lever_arm_gravity

    # Lifting Force at middle (L/2)
    # Note: Even if coordinates are at the edge, the 'middle' is still L/2 distance away.
    lever_arm_user = config.door_length / 2.0

    return torque_gravity / lever_arm_user