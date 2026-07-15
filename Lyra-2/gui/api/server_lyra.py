# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from api_types import (
	CompressedInferenceResult,
	InferenceRequest,
	InferenceResult,
	SeedingRequest,
	SeedingResult,
)
from encoding import CompressionFormat
from server_base import InferenceModel, ROOT_DIR


DEFAULT_QWEN_MODEL = str(Path(ROOT_DIR) / "checkpoints" / "qwen" / "Qwen3-VL-4B-Instruct")


def _find_ffmpeg() -> str:
	ffmpeg = shutil.which("ffmpeg")
	if ffmpeg is not None:
		return ffmpeg
	try:
		import imageio_ffmpeg
		return imageio_ffmpeg.get_ffmpeg_exe()
	except Exception as error:
		raise RuntimeError("ffmpeg is required by the Lyra GUI server.") from error


def _encode_video_bytes(images_thwc: np.ndarray, fps: float) -> bytes:
	"""Encode RGB uint8 frames as browser/QuickTime-compatible H.264."""
	ffmpeg = _find_ffmpeg()
	t, h, w, _ = images_thwc.shape
	cmd = [
		ffmpeg, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
		"-s", f"{w}x{h}", "-r", str(float(fps)), "-i", "pipe:0", "-an",
		"-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart+frag_keyframe+empty_moov",
		"-preset", "fast", "-crf", "18", "-f", "mp4", "pipe:1",
	]
	proc = subprocess.run(cmd, input=np.ascontiguousarray(images_thwc).tobytes(), capture_output=True)
	if proc.returncode != 0:
		raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.decode(errors='replace')}")
	return proc.stdout


class LyraModel(InferenceModel):
	def __init__(self, *, checkpoint_path: str | None = None, **kwargs):
		super().__init__(checkpoint_path=checkpoint_path, **kwargs)
		from lyra_persistent import Lyra2PersistentModel

		checkpoint_path = checkpoint_path or os.environ.get("LYRA_GUI_CHECKPOINT_PATH", "checkpoints/model")
		output_root = os.environ.get("LYRA_GUI_OUTPUT_DIR", os.path.join(ROOT_DIR, "outputs", "gui_sessions"))
		qwen_model = os.environ.get("LYRA_GUI_QWEN_MODEL", DEFAULT_QWEN_MODEL)
		self.model = Lyra2PersistentModel(
			checkpoint_path=checkpoint_path,
			output_root=output_root,
			qwen_model=qwen_model,
		)
		self.pose_history_w2c: list[np.ndarray] = []
		self.intrinsics_history: list[np.ndarray] = []
		self.aabb_min = np.array([-16.0, -16.0, -16.0])
		self.aabb_max = np.array([16.0, 16.0, 16.0])

	async def make_test_image(self):
		raise NotImplementedError

	async def seed_model(self, req: SeedingRequest) -> SeedingResult:
		async with self.inference_lock:
			result = self.model.seed_model_from_values(
				images_np=req.images,
				depths_np=req.depths,
				masks_np=req.masks,
				world_to_cameras_np=req.world_to_cameras(),
				focal_lengths_np=req.focal_lengths,
				principal_point_rel_np=req.principal_points,
				resolutions=req.resolutions,
				request_id=req.request_id,
			)
		self.pose_history_w2c.clear()
		self.intrinsics_history.clear()
		self.model_seeded = True
		return SeedingResult(request_id=req.request_id, **result)

	async def run_inference(self, req: InferenceRequest) -> InferenceResult:
		import torch

		async with self.inference_lock:
			start = time.monotonic()
			w2c = req.world_to_cameras().astype(np.float32)
			target_res = np.tile(
				[[self.model.target_w, self.model.target_h]], (len(req), 1)
			)
			intrinsics = req.intrinsics_matrix(for_resolutions=target_res).astype(np.float32)
			self.pose_history_w2c.append(w2c.copy())
			self.intrinsics_history.append(intrinsics.copy())
			try:
				model_result = self.model.inference_on_cameras(
					w2c,
					intrinsics,
					fps=req.framerate,
					region_hint=req.region_hint,
					save_buffer=req.show_cache_renderings,
					request_id=req.request_id,
				)
			except Exception:
				self.pose_history_w2c.pop()
				self.intrinsics_history.pop()
				raise

		video = model_result["video_no_overlap"]
		if isinstance(video, torch.Tensor):
			video = video.detach().cpu().float().numpy()
		if video.ndim == 5:
			video = video[0]
		images = np.transpose(video, (0, 2, 3, 1))
		n_frames, height, width, _ = images.shape

		pred_depth = model_result.get("predicted_depth")
		if isinstance(pred_depth, torch.Tensor):
			pred_depth = pred_depth.detach().cpu().numpy()
		if isinstance(pred_depth, np.ndarray) and pred_depth.ndim == 4:
			pred_depth = pred_depth[:, 0]
		pred_mask = model_result.get("predicted_mask")
		if isinstance(pred_mask, torch.Tensor):
			pred_mask = pred_mask.detach().cpu().numpy()
		pred_indices = np.asarray(
			model_result.get("predicted_depth_indices", np.empty((0,), dtype=np.int32)), dtype=np.int32
		)

		c2w_out = req.cameras_to_world.copy()
		updated_c2w = model_result.get("updated_last_camera_c2w")
		if isinstance(updated_c2w, np.ndarray) and len(c2w_out):
			c2w_out[-1] = updated_c2w[:3]

		kwargs = dict(
			request_id=req.request_id,
			result_ids=[f"{req.request_id}__frame_{i}" for i in range(n_frames)],
			timestamps=np.zeros((n_frames,), dtype=np.float64),
			cameras_to_world=c2w_out,
			focal_lengths=req.focal_lengths,
			principal_points=req.principal_points,
			resolutions=np.tile([[width, height]], (n_frames, 1)),
			frame_count_without_padding=req.frame_count_without_padding,
			runtime_ms=(time.monotonic() - start) * 1000.0,
			predicted_depths=pred_depth,
			predicted_depth_indices=pred_indices,
			predicted_masks=pred_mask,
			updated_last_camera_c2w=updated_c2w,
			updated_seed_depth=model_result.get("updated_seed_depth"),
			updated_seed_mask=model_result.get("updated_seed_mask"),
		)

		if self.compress_inference_results:
			chunk_path = Path(model_result["video_save_path"])
			video_bytes = chunk_path.read_bytes()
			return CompressedInferenceResult(
				images=None,
				depths=None,
				images_compressed=[video_bytes],
				images_format=CompressionFormat.MP4,
				depths_compressed=None,
				depths_format=None,
				warp_video_compressed=None,
				warp_video_format=None,
				**kwargs,
			)
		return InferenceResult(images=images, depths=None, **kwargs)

	async def revert_last_generation(self) -> dict:
		async with self.inference_lock:
			result = self.model.revert_last_generation()
			if result.get("success"):
				if self.pose_history_w2c:
					self.pose_history_w2c.pop()
				if self.intrinsics_history:
					self.intrinsics_history.pop()
			return result

	def min_frames_per_request(self) -> int:
		return self.model.frames_per_batch

	def max_frames_per_request(self) -> int:
		return self.model.frames_per_batch * 100

	def inference_time_per_frame(self) -> float:
		return 0.5 if self.args_use_dmd else 7.0

	@property
	def args_use_dmd(self) -> bool:
		return bool(self.model.args.use_dmd_scheduler)

	def inference_resolution(self):
		return [(self.model.target_w, self.model.target_h)]

	def default_framerate(self):
		return 16.0

	def requires_seeding(self):
		return True

	def metadata(self) -> dict:
		return {
			"model_name": "Lyra2",
			"model_version": (2, 0, 0),
			"caption_model": self.model.captioner.model_name,
			"aabb_min": self.aabb_min.tolist(),
			"aabb_max": self.aabb_max.tolist(),
			"min_frames_per_request": self.min_frames_per_request(),
			"max_frames_per_request": self.max_frames_per_request(),
			"inference_resolution": self.inference_resolution(),
			"inference_time_per_frame": self.inference_time_per_frame(),
			"default_framerate": self.default_framerate(),
			"requires_seeding": True,
			"session_output_dir": str(self.model.output_root),
		}

	def cleanup(self):
		self.model.cleanup()


class DummyLyraModel(InferenceModel):
	"""Lightweight protocol-compatible model for server/client smoke tests."""

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.width = int(os.environ.get("LYRA_GUI_DUMMY_WIDTH", 96))
		self.height = int(os.environ.get("LYRA_GUI_DUMMY_HEIGHT", 64))
		self.seed_image: np.ndarray | None = None

	async def make_test_image(self):
		raise NotImplementedError

	async def seed_model(self, req: SeedingRequest) -> SeedingResult:
		image = cv2.resize(req.images[0, ..., :3], (self.width, self.height), interpolation=cv2.INTER_AREA)
		self.seed_image = image.astype(np.float32)
		self.model_seeded = True
		depth = np.ones((1, self.height, self.width), dtype=np.float32) * 2.0
		return SeedingResult(
			request_id=req.request_id,
			cameras_to_world=req.cameras_to_world[:1],
			focal_lengths=np.array([[80.0, 80.0]], dtype=np.float32),
			principal_points=np.array([[0.5, 0.5]], dtype=np.float32),
			resolutions=np.array([[self.width, self.height]], dtype=np.int32),
			depths=depth,
			masks=np.ones_like(depth, dtype=np.bool_),
		)

	async def run_inference(self, req: InferenceRequest) -> InferenceResult:
		assert self.seed_image is not None
		start = time.monotonic()
		frames = []
		for i in range(len(req)):
			frame = np.roll(self.seed_image, i % self.width, axis=1)
			frames.append(np.clip(frame * (0.85 + 0.15 * i / max(len(req) - 1, 1)), 0, 1))
		images = np.stack(frames).astype(np.float32)
		video_bytes = _encode_video_bytes((images * 255.0 + 0.5).astype(np.uint8), req.framerate)
		depth = np.ones((1, self.height, self.width), dtype=np.float32) * 2.0
		return CompressedInferenceResult(
			request_id=req.request_id,
			result_ids=[f"{req.request_id}__frame_{i}" for i in range(len(req))],
			timestamps=np.zeros((len(req),), dtype=np.float64),
			cameras_to_world=req.cameras_to_world,
			focal_lengths=req.focal_lengths,
			principal_points=req.principal_points,
			resolutions=np.tile([[self.width, self.height]], (len(req), 1)),
			frame_count_without_padding=req.frame_count_without_padding,
			images=None,
			depths=None,
			runtime_ms=(time.monotonic() - start) * 1000.0,
			predicted_depths=depth,
			predicted_depth_indices=np.array([len(req) - 1], dtype=np.int32),
			predicted_masks=np.ones_like(depth, dtype=np.bool_),
			images_compressed=[video_bytes],
			images_format=CompressionFormat.MP4,
			depths_compressed=None,
			depths_format=None,
		)

	async def revert_last_generation(self):
		return {"success": True, "message": "Dummy state reverted."}

	def min_frames_per_request(self):
		return 80

	def max_frames_per_request(self):
		return 8000

	def inference_time_per_frame(self):
		return 0.001

	def inference_resolution(self):
		return [(self.width, self.height)]

	def default_framerate(self):
		return 16.0

	def requires_seeding(self):
		return True

	def metadata(self):
		return {
			"model_name": "Lyra2Dummy",
			"model_version": (2, 0, 0),
			"caption_model": "dummy",
			"aabb_min": [-16.0] * 3,
			"aabb_max": [16.0] * 3,
			"min_frames_per_request": self.min_frames_per_request(),
			"max_frames_per_request": self.max_frames_per_request(),
			"inference_resolution": self.inference_resolution(),
			"inference_time_per_frame": self.inference_time_per_frame(),
			"default_framerate": self.default_framerate(),
			"requires_seeding": True,
		}
