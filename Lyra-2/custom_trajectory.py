import json
import numpy as np
from pathlib import Path

out = Path("assets/custom_trajectory_examples/example_2")
out.mkdir(parents=True, exist_ok=True)

W, H = 1280, 720
N = 481
fps = 24
phase1 = 121  # about 5s

# Intrinsics similar to existing examples: ~77 degree horizontal FOV
fx = fy = 805.0
cx, cy = W / 2, H / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
intrinsics = np.repeat(K[None], N, axis=0)

def ease(t):
    return t * t * (3 - 2 * t)

def c2w_from_pose(C, yaw):
    # OpenCV-like camera axes: x right, y down, z forward.
    c, s = np.cos(yaw), np.sin(yaw)
    R_c2w = np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c],
    ], dtype=np.float32)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = C
    return c2w

w2c = np.zeros((N, 4, 4), dtype=np.float32)

# Tune these two if motion feels too strong/weak.
forward_dist = 1.2
down_dist = 0.35
yaw_right_deg = 90.0

for i in range(N):
    if i < phase1:
        u = ease(i / (phase1 - 1))
        C = np.array([0.0, down_dist * u, forward_dist * u], dtype=np.float32)
        yaw = 0.0
    else:
        u = ease((i - phase1) / (N - phase1 - 1))
        C = np.array([0.0, down_dist, forward_dist], dtype=np.float32)
        yaw = np.deg2rad(yaw_right_deg) * u

    c2w = c2w_from_pose(C, yaw)
    w2c[i] = np.linalg.inv(c2w).astype(np.float32)

np.savez(
    out / "trajectory.npz",
    w2c=w2c,
    intrinsics=intrinsics,
    image_height=np.array(H),
    image_width=np.array(W),
)

# captions = {
#     "0": "describe your starting scene here",
#     "121": "describe the same scene as the camera turns right",
# }
# (out / "captions.json").write_text(json.dumps(captions, indent=2), encoding="utf-8")