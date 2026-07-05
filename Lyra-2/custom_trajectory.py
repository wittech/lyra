import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
out = BASE_DIR / "assets/custom_trajectory_examples/example_2"
out.mkdir(parents=True, exist_ok=True)

W, H = 1280, 720
N = 481
fps = 16

# Two motion phases, matching captions.json:
#   0-80:   ~5s rotate right while slowly pulling backward.
#   80-480: continue rotating right until facing the hidden wall and door.
turn_pull_end = 80

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

# Tune these if motion feels too strong/weak.
pull_back_dist = 0.9
first_yaw_right_deg = 35.0
final_yaw_right_deg = 92.0
right_drift = 0.15

for i in range(N):
    if i <= turn_pull_end:
        u = ease(i / turn_pull_end)
        C = np.array([right_drift * 0.35 * u, 0.0, -pull_back_dist * u], dtype=np.float32)
        pitch = 0.0
        yaw = np.deg2rad(first_yaw_right_deg) * u
    else:
        u = ease((i - turn_pull_end) / (N - turn_pull_end - 1))
        C = np.array([
            right_drift * (0.35 + 0.65 * u),
            0.0,
            -pull_back_dist,
        ], dtype=np.float32)
        pitch = 0.0
        yaw = np.deg2rad(first_yaw_right_deg + (final_yaw_right_deg - first_yaw_right_deg) * u)

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
print(f"N={N}, fps={fps}, turn_pull_end={turn_pull_end}")
print(f"Use captions at frames: 0, {turn_pull_end}")
