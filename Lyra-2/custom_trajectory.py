import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
out = BASE_DIR / "assets/custom_trajectory_examples/example_2"
out.mkdir(parents=True, exist_ok=True)

W, H = 1280, 720
N = 321
fps = 16

# Three motion phases, matching captions.json:
#   0-48:   ~3s forward and slightly downward.
#   48-64:  ~1s settle from a slight downward angle into a level view.
#   64-320: slow right turn toward the hidden wall and door.
forward_end = 48
level_end = 64

# Intrinsics similar to existing examples: ~77 degree horizontal FOV
fx = fy = 805.0
cx, cy = W / 2, H / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
intrinsics = np.repeat(K[None], N, axis=0)

def ease(t):
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3 - 2 * t)

def rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c],
    ], dtype=np.float32)

def rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c],
    ], dtype=np.float32)

def c2w_from_pose(C, yaw, pitch=0.0):
    # OpenCV-like camera axes: x right, y down, z forward.
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rot_y(yaw) @ rot_x(pitch)
    c2w[:3, 3] = C
    return c2w

w2c = np.zeros((N, 4, 4), dtype=np.float32)

# Tune these two if motion feels too strong/weak.
forward_dist = 1.2
down_dist = 0.35
yaw_right_deg = 90.0
max_down_pitch_deg = -6.0

for i in range(N):
    if i <= forward_end:
        u = ease(i / forward_end)
        C = np.array([0.0, down_dist * u, forward_dist * u], dtype=np.float32)
        pitch = np.deg2rad(max_down_pitch_deg) * u
        yaw = 0.0
    elif i <= level_end:
        u = ease((i - forward_end) / (level_end - forward_end))
        C = np.array([0.0, down_dist, forward_dist], dtype=np.float32)
        pitch = np.deg2rad(max_down_pitch_deg) * (1.0 - u)
        yaw = 0.0
    else:
        u = ease((i - level_end) / (N - level_end - 1))
        C = np.array([0.0, down_dist, forward_dist], dtype=np.float32)
        pitch = 0.0
        yaw = np.deg2rad(yaw_right_deg) * u

    c2w = c2w_from_pose(C, yaw, pitch)
    w2c[i] = np.linalg.inv(c2w).astype(np.float32)

np.savez(
    out / "trajectory.npz",
    w2c=w2c,
    intrinsics=intrinsics,
    image_height=np.array(H),
    image_width=np.array(W),
)

print(f"Wrote {out / 'trajectory.npz'}")
print(f"N={N}, fps={fps}, forward_end={forward_end}, level_end={level_end}")
print(f"Use captions at frames: 0, {forward_end}, {level_end}")
