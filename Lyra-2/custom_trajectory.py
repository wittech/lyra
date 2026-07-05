import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
out = BASE_DIR / "assets/custom_trajectory_examples/example_2"
out.mkdir(parents=True, exist_ok=True)

W, H = 1280, 720
N = 481
fps = 16

# AR-aligned caption phases, matching captions.json:
#   0:   start from the original view.
#   81:  after ~5s, finish the initial right turn and backward pull.
#   161: face the hidden wall with the old wooden door near the desk.
#   241: continue across the opposite side of the room.
#   321: continue the 360-degree scan back toward the original side.
#   401: final AR block, returning close to the starting view by frame 480.
caption_frames = [0, 81, 161, 241, 321, 401]

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

# Tune these keyframes if motion feels too strong/weak. Frame 480 is the last
# real frame for N=481; key 401 is the start of the last AR block.
pose_keyframes = [
    (0, 0.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
    (81, 35.0, np.array([0.05, 0.0, -0.90], dtype=np.float32)),
    (161, 105.0, np.array([0.09, 0.0, -0.90], dtype=np.float32)),
    (241, 170.0, np.array([0.12, 0.0, -0.80], dtype=np.float32)),
    (321, 240.0, np.array([0.10, 0.0, -0.70], dtype=np.float32)),
    (401, 310.0, np.array([0.04, 0.0, -0.35], dtype=np.float32)),
    (480, 360.0, np.array([0.0, 0.0, 0.0], dtype=np.float32)),
]

for i in range(N):
    for (f0, yaw0, c0), (f1, yaw1, c1) in zip(pose_keyframes[:-1], pose_keyframes[1:]):
        if f0 <= i <= f1:
            u = ease((i - f0) / (f1 - f0))
            C = (1.0 - u) * c0 + u * c1
            yaw = np.deg2rad(yaw0 + (yaw1 - yaw0) * u)
            break
    pitch = 0.0

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
print(f"N={N}, fps={fps}, caption_frames={caption_frames}")
print("Pose keyframes:", [frame for frame, _, _ in pose_keyframes])
