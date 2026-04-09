import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


def get_coords(a, b, c):
    # a = distance from hinge to piston wall anchor
    # b = length of door
    # c = length of piston
    val = np.clip((a ** 2 + b ** 2 - c ** 2) / (2 * a * b), -1, 1)
    theta = np.arccos(val)
    # Rotating to vertical wall:
    # Door (hinge at 0,0) extends along negative y-axis
    # x = b * sin(theta)
    # y = -b * cos(theta)
    return theta, b * np.sin(theta), -b * np.cos(theta)


# Set figure size
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.35)

# Set axis limits to handle the range of sliders (1.0 to 20.0)
# We use a buffer to ensure everything stays in view
limit = 25
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.6)

# Graphics
# Wall is now vertical along the y-axis
line_wall, = ax.plot([0, 0], [-limit, limit], 'k-', lw=4, label='Wall')
door_closed, = ax.plot([], [], 'b-', lw=6, label='Closed Door')
piston_closed, = ax.plot([], [], 'r-', lw=3, label='Piston (Closed)')
door_open, = ax.plot([], [], 'b--', lw=4, alpha=0.5, label='Open Door')
piston_open, = ax.plot([], [], 'r--', lw=2, alpha=0.5, label='Piston (Open)')
ax.legend(loc='upper right')


def update(val):
    a, b = slider_a.val, slider_b.val
    l_min, l_max = slider_lmin.val, slider_lmax.val

    theta_c, dx_c, dy_c = get_coords(a, b, l_min)
    theta_o, dx_o, dy_o = get_coords(a, b, l_max)

    # Door from (0,0) to (dx, dy)
    door_closed.set_data([0, dx_c], [0, dy_c])
    # Piston from (0, a) to (dx, dy)
    piston_closed.set_data([0, dx_c], [a, dy_c])

    door_open.set_data([0, dx_o], [0, dy_o])
    piston_open.set_data([0, dx_o], [a, dy_o])

    ax.set_title(f"Swing Range: {abs(np.degrees(theta_o - theta_c)):.1f}°")
    fig.canvas.draw_idle()


# Sliders
ax_a = plt.axes([0.2, 0.20, 0.65, 0.03])
ax_b = plt.axes([0.2, 0.15, 0.65, 0.03])
ax_lmin = plt.axes([0.2, 0.10, 0.65, 0.03])
ax_lmax = plt.axes([0.2, 0.05, 0.65, 0.03])

slider_a = Slider(ax_a, 'Hinge-Wall ($a$)', 1.0, 20.0, valinit=10)
slider_b = Slider(ax_b, 'Hinge-Door ($b$)', 1.0, 20.0, valinit=5)
slider_lmin = Slider(ax_lmin, 'Piston Min ($L_{min}$)', 1.0, 20.0, valinit=6)
slider_lmax = Slider(ax_lmax, 'Piston Max ($L_{max}$)', 1.0, 20.0, valinit=12)

for s in [slider_a, slider_b, slider_lmin, slider_lmax]:
    s.on_changed(update)

update(None)
plt.show()