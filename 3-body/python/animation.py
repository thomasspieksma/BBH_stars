#!/usr/bin/env python3
"""
Animate points from a file with lines like:
0 (-0.0454545, -0, -0) (0.454545, 0, 0) (10, 10, 0) (0, 0, 0)
Supports 2D (x vs y) or 3D animation if use_z=True.
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===== CONFIG =====
filename = "trajectory.txt"
interval = 10          # ms per frame
sample_one_frame_every = 10
trail_length = 500
use_z = True           # set True for 3D animation

# ===== PARSING HELPERS =====
time_re = re.compile(r'^\s*([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)')
paren_re = re.compile(r'\(([^)]*)\)')

times = []
positions = []

with open(filename, 'r') as f:
    for lineno, raw in enumerate(f, start=1):
        line = raw.strip()
        if not line:
            continue

        m = time_re.match(line)
        if not m:
            print(f"[line {lineno}] can't find time at start: {line!r}")
            continue
        t = float(m.group(1))

        vec_texts = paren_re.findall(line)
        if not vec_texts:
            print(f"[line {lineno}] no vectors found: {line!r}")
            continue

        vecs = []
        bad = False
        for vt in vec_texts:
            parts = [p.strip() for p in vt.split(',')]
            if len(parts) != 3:
                print(f"[line {lineno}] vector doesn't have 3 components: ({vt})")
                bad = True
                break
            try:
                comps = [float(p) for p in parts]
            except ValueError as e:
                print(f"[line {lineno}] can't convert vector components to float: ({vt}) -> {e}")
                bad = True
                break
            vecs.append(comps)
        if bad:
            continue

        times.append(t)
        positions.append(np.array(vecs, dtype=float))

# ===== CONVERT AND INTERPOLATE =====
if not times:
    raise SystemExit("No valid data parsed from file.")

times = np.array(times)
n_points = positions[0].shape[0]
for i, arr in enumerate(positions):
    if arr.shape[0] != n_points:
        raise ValueError(f"Inconsistent number of vectors at line {i+1}")

positions = np.stack(positions, axis=0)

# Uniform time grid
n_frames = int(len(times) / sample_one_frame_every)
uniform_times = np.linspace(times.min(), times.max(), n_frames)

positions_uniform = np.empty((n_frames, n_points, 3))
for i in range(n_points):
    for j in range(3):
        positions_uniform[:, i, j] = np.interp(uniform_times, times, positions[:, i, j])

times = uniform_times
positions = positions_uniform
n_times = n_frames

# ===== PLOT SETUP =====
if use_z:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
else:
    fig, ax = plt.subplots()

colors = plt.cm.tab10(np.linspace(0, 1, n_points))

if use_z:
    points_plots = [ax.plot([], [], [], 'o', markersize=6, color=colors[i])[0] for i in range(n_points)]
    trail_plots  = [ax.plot([], [], [], '-', linewidth=1, alpha=0.6, color=colors[i])[0] for i in range(n_points)]
else:
    points_plots = [ax.plot([], [], 'o', markersize=6, color=colors[i])[0] for i in range(n_points)]
    trail_plots  = [ax.plot([], [], '-', linewidth=1, alpha=0.6, color=colors[i])[0] for i in range(n_points)]

# ===== FIXED AXIS LIMITS =====
xs = positions[..., 0].ravel()
ys = positions[..., 1].ravel()
ax.set_xlim(xs.min() - 0.1*(xs.max()-xs.min()), xs.max() + 0.1*(xs.max()-xs.min()))
ax.set_ylim(ys.min() - 0.1*(ys.max()-ys.min()), ys.max() + 0.1*(ys.max()-ys.min()))

if use_z:
    zs = positions[..., 2].ravel()
    ax.set_zlim(zs.min() - 0.1*(zs.max()-zs.min()), zs.max() + 0.1*(zs.max()-zs.min()))
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
else:
    ax.set_xlabel('x'); ax.set_ylabel('y')

# ax.set_xlim(-2,2)
# ax.set_ylim(-2,2)
# if use_z:
#     ax.set_zlim(-2,2)

ax.set_aspect('auto' if use_z else 'equal')

# ===== INFO BOX / TIME DISPLAY =====
if use_z:
    # 3D: we’ll just print time to console
    def update_time_text(frame):
        print(f"t = {times[frame]:.3f}  (frame {frame+1}/{n_times})")
else:
    # 2D: use text2D
    info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes)
    def update_time_text(frame):
        info_text.set_text(f"t = {times[frame]:.3f}  (frame {frame+1}/{n_times})")

# ===== ROTATION CONFIG =====
if use_z:
    rotation_speed = 0.1  # degrees per frame
    elev = 20             # fixed elevation angle

# ===== ANIMATION FUNCTION =====
def update(frame):
    for i in range(n_points):
        x = positions[frame, i, 0]
        y = positions[frame, i, 1]
        z = positions[frame, i, 2]

        if use_z:
            points_plots[i].set_data_3d([x], [y], [z])
            start = max(0, frame - trail_length)
            trail_plots[i].set_data_3d(positions[start:frame+1, i, 0],
                                       positions[start:frame+1, i, 1],
                                       positions[start:frame+1, i, 2])
        else:
            points_plots[i].set_data([x], [y])
            start = max(0, frame - trail_length)
            trail_plots[i].set_data(positions[start:frame+1, i, 0],
                                    positions[start:frame+1, i, 1])

    update_time_text(frame)

    # Rotate view slowly if 3D
    if use_z:
        azim = (rotation_speed * frame) % 360
        ax.view_init(elev=elev, azim=azim)

    return points_plots + trail_plots  # remove info_text from return

ani = FuncAnimation(fig, update, frames=n_times, interval=interval)  # no blit=True
plt.show()