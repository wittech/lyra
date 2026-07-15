#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import code
from copy import deepcopy
from datetime import datetime
import glob
import os
from os.path import realpath, dirname, join
import subprocess
import sys
import time
import math
import threading
import tempfile

import cv2
import httpx
import numpy as np
import pyexr
import tqdm

ROOT_DIR = realpath(dirname(dirname(__file__)))
DATA_DIR = join(ROOT_DIR, "data")

sys.path.append(join(ROOT_DIR, "scripts"))

# Search for pyngp in the build folder.
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.pyd"), recursive=True)]
sys.path += [os.path.dirname(pyd) for pyd in glob.iglob(os.path.join(ROOT_DIR, "build*", "**/*.so"), recursive=True)]

import pyngp as ngp
from pyngp import tlog

from api_serialization import API_MEDIA_TYPE, dumps_api_message, loads_api_message
from api_types import SeedingRequest, CompressedSeedingRequest, SeedingResult, \
					  InferenceRequest, InferenceResult, CompressedInferenceResult, \
					  RequestState, PendingRequest, RevertResult
from httpx_utils import httpx_request
from v2v_utils import load_v2v_seeding_data, ensure_alpha_channel



def repl(testbed):
	print("-------------------\npress Ctrl-Z to return to gui\n---------------------------")
	code.InteractiveConsole(locals=locals()).interact()
	print("------- returning to gui...")


def open_file_with_default_app(video_path: str) -> None:
	"""Open the saved video file with the default video player application."""
	try:
		if sys.platform == "win32":
			# Windows
			os.startfile(video_path)
		else:
			# Avoid venv, etc interfering with the application that will open.
			env = os.environ.copy()
			for k in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "LD_LIBRARY_PATH"):
				if k in env:
					del env[k]
			if sys.platform == "darwin":
				# macOS
				subprocess.run(["open", video_path], check=True, env=env)
			else:
				# Linux, etc.
				subprocess.run(["xdg-open", video_path], check=True, env=env)
	except Exception as e:
		tlog.error(f"Failed to open video file: {e}")


class Lyra2Client():
	def __init__(
		self,
		files: list[str],
		host: str,
		port: int,
		width: int = 1920,
		height: int = 1080,
		vr: bool = False,
		request_latency_ms: int = 100,
		inference_resolution: tuple[int, int] = (1920, 1080),
		# max_pending_requests: int = 2,
		max_pending_requests: int = 1,
		request_timeous_seconds: float = 3600,
		seed_max_frames: int | None = None,
		seed_stride: int = 1,
		output_dir: str | None = None,
	):
		self.url = f"http://{host}:{port}"
		self.client_id = f"lyra2-{os.getpid()}"
		self.request_latency_ms = request_latency_ms
		self.inference_resolution = inference_resolution
		self.max_pending_requests = max_pending_requests
		self.req_timeout_s = request_timeous_seconds
		self.seed_max_frames = seed_max_frames
		self.seed_stride = seed_stride

		testbed = ngp.Testbed(ngp.TestbedMode.Gen3c)
		testbed.root_dir = ROOT_DIR
		testbed.set_gen3c_cb(self.gui_callback)
		testbed.file_drop_callback = self.file_drop_callback

		if output_dir is not None:
			os.makedirs(output_dir, exist_ok=True)
		else:
			output_dir = join(ROOT_DIR, "outputs")
		testbed.gen3c_output_dir = output_dir
		testbed.video_path = join(output_dir, "lyra2_%Y-%m-%d_%H-%M-%S.mp4")

		# --- Check metadata from server to ensure compatibility
		testbed.reproject_visualize_src_views = False
		testbed.render_aabb.min = np.array([-16.0, -16.0, -16.0])
		testbed.render_aabb.max = np.array([16.0, 16.0, 16.0])
		try:
			tlog.info(f"Requesting metadata from server {host}:{port}")
			metadata = self.request_metadata_sync()
			testbed.render_aabb.min = np.array(metadata["aabb_min"]).astype(np.float32)
			testbed.render_aabb.max = np.array(metadata["aabb_max"]).astype(np.float32)
			testbed.aabb = ngp.BoundingBox(testbed.render_aabb.min, testbed.render_aabb.max)
			testbed.gen3c_info = f"Connected to server {host}:{port}, model name: {metadata.get('model_name')}"

			model_inference_res: list[tuple[int, int]] | None = metadata.get("inference_resolution")
			if model_inference_res is not None:
				for supported_res in model_inference_res:
					if tuple(supported_res) == self.inference_resolution:
						break
				else:
					r = tuple(model_inference_res[0])
					tlog.warning(f"Client inference resolution {self.inference_resolution} is not"
						         f" supported by the inference server, adopting resolution {r} instead.")
					self.inference_resolution = r
			testbed.camera_path.render_settings.resolution = self.inference_resolution

			testbed.gen3c_inference_is_connected = True
			testbed.gen3c_render_with_gen3c = True

		except httpx.ConnectError as e:
			# The metadata-based setup happens only once at startup. Since we failed to
			# get the metadata from the server, it's easier to just raise and exit here.
			raise RuntimeError(
				f"Connection error! Make sure the server was started at: {host}:{port}\n{e}"
			) from e

		testbed.camera_path.render_settings.fps = metadata.get("default_framerate") or 24.0
		self.min_frames_per_request: int = metadata.get("min_frames_per_request", 1)
		self.max_frames_per_request: int = metadata.get("max_frames_per_request", 1)
		if self.min_frames_per_request > 1:
			# Set default render settings such that the model can generate it
			# in a single batch exactly.
			testbed.camera_path.default_duration_seconds = \
				self.min_frames_per_request / testbed.camera_path.render_settings.fps
			testbed.camera_path.duration_seconds = testbed.camera_path.default_duration_seconds

		# Expected time that the model will take to generate each frame, in seconds
		self.inference_time_per_frame: float = metadata.get("inference_time_per_frame", 0.0)
		# Don't automatically request new frames all the time if inference is slow
		testbed.gen3c_auto_inference &= (self.inference_time_per_frame < 1.0)

		self.seeding_pending: bool = False
		self.model_requires_seeding: bool = metadata.get("requires_seeding", True)
		if self.model_requires_seeding:
			testbed.gen3c_info += "\nThis model requires seeding data."
		# Generation workflow state (see `self._update_generate_video_enabled()`).
		self._seeding_complete: bool = (not self.model_requires_seeding)
		self._seed_baseline_keyframes: int = 0
		self._has_generated_once: bool = False
		# One-level undo state for reverting the last generation.
		self._pre_gen_snapshot: dict | None = None
		# Seed frame info for predicted-pose correction (updated_seed_depth).
		self._seed_c2w: np.ndarray | None = None
		self._seed_focal_lengths: np.ndarray | None = None
		self._seed_principal_points: np.ndarray | None = None
		self._seed_image_rgba: np.ndarray | None = None
		self._seed_depth_hw: np.ndarray | None = None

		# Pick a sensible GUI resolution depending on arguments.
		sw = width
		sh = height
		while sw * sh > 1920 * 1080 * 4:
			sw = int(sw / 2)
			sh = int(sh / 2)

		testbed.init_window(sw, sh)
		if vr:
			testbed.init_vr()
		self.testbed: ngp.Testbed = testbed

		self.lens = ngp.Lens()
		self.lens.mode = ngp.LensMode.Perspective

		self.client = httpx.AsyncClient()
		self._http_stream_chunk_size = int(os.environ.get("LYRA_GUI_HTTP_STREAM_CHUNK_SIZE", 1024 * 1024))  # 1MB
		self._progress_interval_s = float(os.environ.get("LYRA_GUI_HTTP_PROGRESS_INTERVAL_S", 0.10))  # 100ms

		self.last_request_id: int = 0
		self.start_t: float = None
		self.last_request_t: float = None
		self.pending_requests: dict[str, PendingRequest] = {}

		# Handle files given as command-line arguments.
		if files:
			self.file_drop_callback(files)

		# Default: disabled until we have seeding + at least one new camera/keyframe.
		self.testbed.gen3c_generate_video_enabled = False
		self.testbed.gen3c_revert_available = False

	def _duplicate_endpoints_for_sampling(self, keyframes: list) -> list:
		"""
		Method A: duplicate the first and last keyframes as control points so that the spline
		sampling includes exact endpoints (p0 and pn) without manually overwriting sampled frames.
		"""
		if len(keyframes) < 2:
			return keyframes
		first = keyframes[0]
		last = keyframes[-1]
		first_dup = ngp.CameraKeyframe(first.m(), float(first.fov), float(first.timestamp), first.up_dir)
		last_dup = ngp.CameraKeyframe(last.m(), float(last.fov), float(last.timestamp), last.up_dir)
		return [first_dup] + list(keyframes) + [last_dup]

	def _sample_authored_camera_path(self, n_frames: int, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
		"""
		Sample the current authored camera path into (cameras_to_world, focal_lengths) with
		endpoints included via duplicated control points.
		"""
		cp = self.testbed.camera_path
		orig_keyframes = [ngp.CameraKeyframe(kf.m(), float(kf.fov), float(kf.timestamp), kf.up_dir) for kf in cp.keyframes]
		sampling_keyframes = self._duplicate_endpoints_for_sampling(orig_keyframes)

		out_c2w: np.ndarray | None = None
		out_fl: np.ndarray | None = None

		def _make_uniform_ts_by_arc_length(ts: np.ndarray, centers_xyz: np.ndarray) -> np.ndarray:
			"""
			Given a dense sampling (ts, centers), return `n_frames` values of t that are
			approximately equally spaced by translation arc length.
			"""
			if len(ts) <= 1:
				return ts
			d = np.linalg.norm(np.diff(centers_xyz, axis=0), axis=1)
			total = float(np.sum(d))
			if not np.isfinite(total) or total <= 1e-8:
				return np.linspace(0.0, 1.0, n_frames, endpoint=True)
			cum = np.concatenate([[0.0], np.cumsum(d)])
			target = np.linspace(0.0, total, n_frames, endpoint=True)
			idx = np.searchsorted(cum, target, side="left")
			idx = np.clip(idx, 1, len(cum) - 1)
			t0 = ts[idx - 1]
			t1 = ts[idx]
			c0 = cum[idx - 1]
			c1 = cum[idx]
			alpha = (target - c0) / np.maximum(c1 - c0, 1e-12)
			return (1.0 - alpha) * t0 + alpha * t1

		def do_sample():
			nonlocal out_c2w, out_fl
			# First, densely sample the path uniformly in parameter space.
			dense_n = max(n_frames * 10, n_frames + 1)
			ts_dense = np.linspace(0.0, 1.0, dense_n, endpoint=True)
			kfs_dense = [self.testbed.camera_path.eval_camera_path(float(t)) for t in ts_dense]
			# Use matrix translation instead of `kf.T` to avoid any pybind vec3 conversion quirks.
			ms_dense = [np.asarray(kf.m(), dtype=np.float32) for kf in kfs_dense]
			centers_dense = np.stack([m[:, 3] for m in ms_dense], axis=0)

			# Re-sample with approximately uniform translation arc length per frame.
			ts = _make_uniform_ts_by_arc_length(ts_dense, centers_dense)
			# Ensure endpoints are included.
			if len(ts) > 0:
				ts[0] = 0.0
				ts[-1] = 1.0
			kfs = [self.testbed.camera_path.eval_camera_path(float(t)) for t in ts]

			ms = [np.asarray(kf.m(), dtype=np.float32) for kf in kfs]
			out_c2w = np.stack(ms, axis=0)
			out_fl = np.stack([
				[ngp.fov_to_focal_length(self.inference_resolution[self.testbed.fov_axis], kf.fov)] * 2
				for kf in kfs
			], axis=0)

		self._with_temporary_camera_path(
			keyframes=sampling_keyframes,
			segment_duration_s=duration_s,
			fn=do_sample,
		)
		assert out_c2w is not None and out_fl is not None
		return out_c2w, out_fl



	async def run(self):
		testbed = self.testbed

		self.start_t = time.monotonic()
		self.last_request_t = time.monotonic()
		# TODO: any way to make the rendering itself async? (pybind11 support?)
		while testbed.frame():
			# --- At each frame
			if testbed.want_repl():
				repl(testbed)

			if self.model_requires_seeding and self.seeding_pending and self.testbed.gen3c_seed_path:
				tlog.info(f"Loading seeding data with path: {self.testbed.gen3c_seed_path}")
				# Load the seeding data.
				seed_req = self.load_seeding_data(self.testbed.gen3c_seed_path)
				if seed_req is not None:
					self.adapt_view_to_cameras(seed_req.cameras_to_world)
					# Send the seeding request over to the server (could be a slow upload).
					self.send_seeding_request(seed_req)
				self.seeding_pending = False

			# Give coroutines a chance to run (especially if there are pending HTTP requests).
			# This is essentially a "yield".
			# TODO: how can we sleep only for the minimum needed time?
			#       Probably we would need to request the `testbed`'s frame in an
			#       async way as well? Something like:
			#    await testbed.frame()
			await asyncio.sleep(0.003 if self._transfer_in_progress() else 0.0001)

			# Check pending inference requests
			self.get_request_results()
			self._update_generate_video_enabled()

			# New inference request
			# TODO: if there are too many pending requests, cancel the oldest one
			#       instead of continuing to wait.
			now = time.monotonic()
			if ((1000 * (now - self.last_request_t) > self.request_latency_ms)
				and testbed.gen3c_auto_inference
				and testbed.is_rendering
				and len(self.pending_requests) < self.max_pending_requests):

				self.request_frames()

	def _update_generate_video_enabled(self) -> None:
		"""
		Controls whether the GUI "Generate video" button is enabled.

		Workflow:
		- After seeding: disabled until the user adds >=1 new camera keyframe
		  (i.e. keyframe count grows beyond the seeding baseline).
		- After each successful generation: freeze existing keyframes (via `locked_prefix`)
		  and disable until the user appends >=1 new keyframe.
		"""
		cp = self.testbed.camera_path
		if len(self.pending_requests) > 0:
			self.testbed.gen3c_generate_video_enabled = False
			return
		if self.model_requires_seeding and (not self._seeding_complete):
			self.testbed.gen3c_generate_video_enabled = False
			return

		n_kf = len(cp.keyframes)
		if not self._has_generated_once:
			self.testbed.gen3c_generate_video_enabled = (n_kf > self._seed_baseline_keyframes)
		else:
			self.testbed.gen3c_generate_video_enabled = (n_kf > int(cp.locked_prefix))

	def _with_temporary_camera_path(self, keyframes: list, segment_duration_s: float | None, fn: callable) -> None:
		"""
		Temporarily replaces `testbed.camera_path.keyframes` so that `request_frames()`
		can reuse the built-in C++ `CameraPath::eval_camera_path()` logic.
		"""
		cp = self.testbed.camera_path
		# Avoid `deepcopy()` on pybind objects (can be unreliable depending on bindings).
		orig_keyframes = [ngp.CameraKeyframe(kf.m(), float(kf.fov), float(kf.timestamp), kf.up_dir) for kf in cp.keyframes]
		try:
			cp.keyframes = [ngp.CameraKeyframe(kf.m(), float(kf.fov), float(kf.timestamp), kf.up_dir) for kf in keyframes]
			if segment_duration_s is not None and segment_duration_s > 0:
				# Re-timestamp keyframes to match the requested segment duration.
				cp.make_keyframe_timestamps_equidistant(float(segment_duration_s))
			cp.sanitize_keyframes()
			fn()
		finally:
			cp.keyframes = orig_keyframes
			cp.sanitize_keyframes()

	def get_request_results(self):
		to_remove = set()
		for req_id, state in self.pending_requests.items():

			if state.state in (RequestState.FAILED, RequestState.COMPLETE):
				# Cleanup requests that are done one way or another
				to_remove.add(req_id)

			elif state.state == RequestState.REQUEST_PENDING:
				# Before checking the results, we wait for the inference request to have
				# been received by the server at least.
				self.testbed.gen3c_inference_info = f"Waiting for inference request {req_id} to be received by the server..."
				continue

			elif state.state == RequestState.REQUEST_SENT:
				# Server has received the inference request, we should now start checking results
				def on_result_received(result: InferenceResult | None,
									   response: httpx.Response, failed: bool = False):
					if failed:
						tlog.error(f"Results request for inference {req_id} failed!\n"
								   f"{response.content}")
						self.testbed.gen3c_inference_info = f"Error: {response.content}"
						state.state = RequestState.FAILED
						state.task = None
						return

					if result is None:
						# Result not ready yet, check again soon
						state.state = RequestState.REQUEST_SENT
						state.task = None
						return

					# Actual result received!
					assert isinstance(result, InferenceResult)
					state.state = RequestState.COMPLETE
					self.testbed.gen3c_inference_info = ""

					tlog.success(f"Received results {req_id}: took {result.runtime_ms:.1f} ms to generate.")

					need_frames = self.testbed.gen3c_save_frames
					result.trim_to_original_frame_count()
					pred_depths = getattr(result, "predicted_depths", None)
					pred_depth_indices = getattr(result, "predicted_depth_indices", None)
					decoded_pred_rgb = None
					if isinstance(result, CompressedInferenceResult):
						# Save the compressed video straight to disk
						video_path = datetime.now().strftime(self.testbed.video_path)
						result.save_images(video_path)
						tlog.success(f"[+] Wrote generated video to: {video_path}")

						tlog.info(f"Opening file with default application: {video_path}")
						open_file_with_default_app(video_path)
						if need_frames:
							result.decompress()

						# If the server returned sparse predicted depth(s), decode the corresponding
						# RGB frame(s) from the saved video so we can render pointclouds.
						if (pred_depths is not None) and (pred_depth_indices is not None):
							try:
								indices = np.asarray(pred_depth_indices).reshape(-1).astype(np.int32)
								if indices.size > 0:
									cap = cv2.VideoCapture(video_path)
									rgbs = []
									for i_idx in indices.tolist():
										cap.set(cv2.CAP_PROP_POS_FRAMES, int(i_idx))
										ok, frame_bgr = cap.read()
										if not ok or frame_bgr is None:
											tlog.warning(f"Failed to decode frame {i_idx} from {video_path}")
											rgbs.append(None)
											continue
										frame_rgb = frame_bgr[..., ::-1].astype(np.float32) / 255.0
										rgbs.append(frame_rgb)
									cap.release()
									decoded_pred_rgb = rgbs
							except Exception as e:
								tlog.warning(f"Failed to decode predicted-depth RGB frame(s) from video: {e}")

					# Add all received frames to the viewer.
					if self.testbed.gen3c_save_frames:
						os.makedirs(self.testbed.gen3c_output_dir, exist_ok=True)

					view_ids = set(self.testbed.src_view_ids())
					for res_i in range(len(result)):
						if not need_frames:
							continue

						# Only display the result if we don't already have it shown
						res_id = result.result_ids[res_i]

						if (res_id is not None) and (res_id in view_ids):
							tlog.debug(f"Skipping result since id {res_id} is already displayed")
							continue

						# Allow alpha channel to be omitted for faster transfers
						image = ensure_alpha_channel(result.images[res_i, ...])

						if self.testbed.gen3c_save_frames:
							safe_res_id = (res_id or f"{res_i:04d}").replace(":", "_")
							fname = join(self.testbed.gen3c_output_dir,
										 f"rgb_{safe_res_id}.exr")
							pyexr.write(fname, image)

							fname = join(self.testbed.gen3c_output_dir,
										 f"depth_{safe_res_id}.exr")
							pyexr.write(fname, result.depths[res_i, ...].astype(np.float32))
							tlog.success(f"[+] Wrote inference result to: {fname}")

					# Add predicted-depth pointcloud(s) to the viewer (even if we didn't request per-frame depths).
					if (pred_depths is not None) and (pred_depth_indices is not None):
						try:
							indices = np.asarray(pred_depth_indices).reshape(-1).astype(np.int32)
							pd = np.asarray(pred_depths)
							pm = getattr(result, "predicted_masks", None)
							pm = np.asarray(pm) if pm is not None else None
							# Allow [k,1,H,W] or [k,H,W]
							if pd.ndim == 4 and pd.shape[1] == 1:
								pd = pd[:, 0, ...]
							assert pd.ndim == 3, pd.shape
							assert pd.shape[0] == indices.shape[0], (pd.shape, indices.shape)
							if pm is not None:
								if pm.ndim == 4 and pm.shape[1] == 1:
									pm = pm[:, 0, ...]
								# Expect [k,H,W]
								if pm.ndim != 3 or pm.shape[0] != indices.shape[0]:
									pm = None

							for j, frame_idx in enumerate(indices.tolist()):
								if frame_idx < 0 or frame_idx >= len(result):
									tlog.warning(f"Skipping predicted depth frame_idx={frame_idx} (out of range for {len(result)} frames)")
									continue
								depth_hw = pd[j].astype(np.float32)
								if pm is not None:
									mask_hw = pm[j].astype(np.bool_)
									if mask_hw.shape == depth_hw.shape:
										depth_hw = depth_hw.copy()
										depth_hw[~mask_hw] = 0.0

								# Choose an RGB image for coloring the pointcloud.
								# - If compressed video: use decoded frame (if available)
								# - Else: use result.images[frame_idx]
								img = None
								if isinstance(result, CompressedInferenceResult):
									if decoded_pred_rgb is not None and j < len(decoded_pred_rgb):
										img = decoded_pred_rgb[j]
								else:
									img = result.images[frame_idx, ...]

								if img is None:
									# Fallback: gray image
									h, w = depth_hw.shape
									img = np.full((h, w, 3), 0.5, dtype=np.float32)

								# Normalize common model output ranges to [0,1] if needed.
								if img.dtype != np.float32 and img.dtype != np.float64:
									img = img.astype(np.float32)
								if img.max() > 1.5:
									img = img / 255.0
								if img.min() < -0.5:
									img = (img + 1.0) * 0.5

								image_rgba = ensure_alpha_channel(img)
								has_valid_depth = np.any(np.isfinite(depth_hw))
								if not has_valid_depth:
									continue

								self.testbed.add_src_view(
									result.cameras_to_world[frame_idx, ...],
									result.focal_lengths[frame_idx][0],
									result.focal_lengths[frame_idx][1],
									result.principal_points[frame_idx][0],
									result.principal_points[frame_idx][1],
									self.lens,
									image_rgba,
									depth_hw,
									result.timestamps[frame_idx],
									is_srgb=True,
								)
								self.testbed.reset_accumulation(reset_pip=True)
								valid_points = int(np.count_nonzero(np.isfinite(depth_hw) & (depth_hw > 0)))
								total_points = int(depth_hw.size)
								tlog.info(
									f"Added predicted-depth pointcloud for frame {frame_idx}: "
									f"{valid_points}/{total_points} valid pixels "
									f"({100.0 * valid_points / max(total_points, 1):.1f}%)"
								)
						except Exception as e:
							tlog.warning(f"Failed to add predicted-depth pointcloud(s): {e}")

					# Don't display more than 8 views at once by default to avoid
					# slowing down the rendering too much.
					self.set_max_number_of_displayed_views(100)

					# --- Predicted-pose correction: update keyframe + seed depth ---
					try:
						updated_c2w = getattr(result, "updated_last_camera_c2w", None)
						if updated_c2w is not None and isinstance(updated_c2w, np.ndarray):
							cp = self.testbed.camera_path
							kf_idx = len(cp.keyframes) - 1
							if kf_idx >= 0:
								old_kf = cp.keyframes[kf_idx]
								c2w_34 = updated_c2w.astype(np.float32)
								if c2w_34.shape == (4, 4):
									c2w_34 = c2w_34[:3, :]
								# pybind11 returns a *copy* of the C++ vector for
								# def_readwrite std::vector members, so element
								# assignment on the proxy has no effect.  Replace
								# the whole list instead.
								kfs = list(cp.keyframes)
								kfs[kf_idx] = ngp.CameraKeyframe(
									c2w_34,
									float(old_kf.fov),
									float(old_kf.timestamp),
									old_kf.up_dir,
								)
								cp.keyframes = kfs
								tlog.info(f"Updated last keyframe (idx={kf_idx}) with predicted-pose corrected c2w")

						updated_seed_d = getattr(result, "updated_seed_depth", None)
						if updated_seed_d is not None and isinstance(updated_seed_d, np.ndarray):
							seed_c2w = self._seed_c2w
							seed_fl = self._seed_focal_lengths
							seed_pp = self._seed_principal_points
							seed_img = self._seed_image_rgba
							if seed_c2w is not None and seed_fl is not None and seed_pp is not None and seed_img is not None:
								depth_hw = updated_seed_d.astype(np.float32)
								updated_seed_m = getattr(result, "updated_seed_mask", None)
								if updated_seed_m is not None and isinstance(updated_seed_m, np.ndarray):
									mask_hw = updated_seed_m.astype(np.bool_)
									if mask_hw.shape == depth_hw.shape:
										depth_hw = depth_hw.copy()
										depth_hw[~mask_hw] = 0.0
								# Hide the old seed view (index 0) so its stale depth
								# doesn't overlap with the corrected one.
								if self.testbed.reproject_min_src_view_index == 0:
									self.testbed.reproject_min_src_view_index = 1
								self.testbed.add_src_view(
									seed_c2w,
									float(seed_fl[0]),
									float(seed_fl[1]),
									float(seed_pp[0]),
									float(seed_pp[1]),
									self.lens,
									seed_img,
									depth_hw,
									0.0,
									is_srgb=True,
								)
								self.testbed.reproject_max_src_view_index = self.testbed.reproject_src_views_count()
								self.testbed.reset_accumulation(reset_pip=True)
								tlog.info("Replaced seed depth pointcloud with predicted-pose corrected version")
					except Exception as e:
						tlog.warning(f"Failed to apply predicted-pose corrections: {e}")

					self.testbed.camera_path.locked_prefix = len(self.testbed.camera_path.keyframes)
					self._has_generated_once = True
					self.testbed.gen3c_revert_available = True
					self._update_generate_video_enabled()

				tlog.debug(f"Checking results of request {req_id}...")
				state.state = RequestState.RESULT_PENDING
				state.task = self._get_inference_results(req_id, on_result_received)

			elif state.state == RequestState.RESULT_PENDING:
				# We already sent a request to check on the results, let's wait until
				# a response comes back (through the `on_result_received` cb).
				if self.testbed.gen3c_inference_progress < 0:
					# Only show the spinner if downloading the results hasn't started yet.
					spinner = "|/-\\"[int(4 * time.time()) % 4]
					self.testbed.gen3c_inference_info = f"[{spinner}] Waiting for server to complete inference..."
				pass


		for k in to_remove:
			del self.pending_requests[k]
		self.testbed.camera_path.rendering = len(self.pending_requests) > 0

	# ----------

	def request_metadata_sync(self) -> InferenceResult:
		# Synchronous request (no need to `await`)
		return httpx_request("get", self.url + "/metadata", timeout=self.req_timeout_s).json()


	def request_frames(self, sync: bool = False) -> asyncio.Task | InferenceResult:
		# The user wants a certain number of frames, but the model can only generate
		# `self.min_frames_per_request` per request. Pad to get there.
		n_desired_frames = int(np.ceil(self.testbed.camera_path.duration_seconds
							           * self.testbed.camera_path.render_settings.fps))
		n_frames_padded = max(
			int(np.ceil(n_desired_frames / self.min_frames_per_request) * self.min_frames_per_request),
			self.min_frames_per_request
		)
		self.testbed.gen3c_inference_info = (
			f"Requesting {n_desired_frames} frames ({n_frames_padded} total with padding, "
			f"model has min batch size {self.min_frames_per_request})."
		)
		tlog.info(self.testbed.gen3c_inference_info)
		# TODO: enforce `max_frames_per_request` from the server, too (with a clear error message)
		now = time.monotonic()

		cameras_to_world = np.repeat(self.testbed.camera_matrix[None, ...],
									 repeats=n_desired_frames, axis=0)

		# By default, use the preview camera focal length.
		# We assume square pixels, so horizontal and vertical focal lengths are equal.
		default_focal_length = self.testbed.relative_focal_length * self.inference_resolution[self.testbed.fov_axis]
		focal_lengths = np.array([default_focal_length] * n_desired_frames)

		match self.testbed.gen3c_camera_source:
			case ngp.Gen3cCameraSource.Fake:
				# --- Camera movement: fake based on fixed translation and rotation speeds
				counter = np.arange(n_desired_frames)[..., None]

				if np.any(self.testbed.gen3c_rotation_speed != 0):
					angles = counter * self.testbed.gen3c_rotation_speed[None, ...]
					alphas = angles[:, 0]
					betas = angles[:, 1]
					gammas = angles[:, 2]

					# TODO: nicer way to build the rotation matrix
					fake_rotation = np.tile(np.eye(3, 3)[None, ...], (n_desired_frames, 1, 1))
					fake_rotation[:, 0, 0] = np.cos(betas) * np.cos(gammas)
					fake_rotation[:, 0, 1] = (
						np.sin(alphas) * np.sin(betas) * np.cos(gammas)
						- np.cos(alphas) * np.sin(gammas)
					)
					fake_rotation[:, 0, 2] = (
						np.cos(alphas) * np.sin(betas) * np.cos(gammas)
						+ np.sin(alphas) * np.sin(gammas)
					)

					fake_rotation[:, 1, 0] = np.cos(betas) * np.sin(gammas)
					fake_rotation[:, 1, 1] = (
						np.sin(alphas) * np.sin(betas) * np.sin(gammas)
						+ np.cos(alphas) * np.cos(gammas)
					)
					fake_rotation[:, 1, 2] = (
						np.cos(alphas) * np.sin(betas) * np.sin(gammas)
						- np.sin(alphas) * np.cos(gammas)
					)

					fake_rotation[:, 2, 0] = -np.sin(betas)
					fake_rotation[:, 2, 1] = np.sin(alphas) * np.cos(betas)
					fake_rotation[:, 2, 2] = np.cos(alphas) * np.cos(betas)

					cameras_to_world[:, :3, :3] @= fake_rotation

				if np.any(self.testbed.gen3c_translation_speed != 0):
					fake_translation = counter * self.testbed.gen3c_translation_speed[None, ...]
					cameras_to_world[:, :, 3] += fake_translation

			case ngp.Gen3cCameraSource.Viewpoint:
				# --- Camera movement: based on the current viewpoint + predicted movement
				tlog.error("Not implemented: Lyra-2 camera movement source: Viewpoint")
				return

			case ngp.Gen3cCameraSource.Authored:
				cameras_to_world, focal_lengths = self._sample_authored_camera_path(
					n_frames=n_desired_frames,
					duration_s=float(self.testbed.camera_path.duration_seconds),
				)

			case _:
				raise ValueError("Unsupported Lyra-2 camera movement source:",
								 self.testbed.gen3c_camera_source)
		t0 = now - self.start_t
		timestamps = [t0 + i * self.inference_time_per_frame
					  for i in range(n_desired_frames)]

		request_id = f"{self.client_id}:{self.last_request_id + 1}"

		tlog.debug(f"Creating new request {request_id}")
		req = InferenceRequest(
			request_id=request_id,
			timestamps=np.array(timestamps),
			cameras_to_world=cameras_to_world,
			focal_lengths=focal_lengths,
			principal_points=np.array([self.testbed.screen_center] * n_desired_frames),
			resolutions=np.array([self.inference_resolution] * n_desired_frames),
			framerate=self.testbed.camera_path.render_settings.fps,
			return_depths=self.testbed.gen3c_save_frames,
			video_encoding_quality=self.testbed.camera_path.render_settings.quality,
			show_cache_renderings=False,
			region_hint=getattr(self.testbed, "gen3c_region_hint", ""),
		)
		# Add any necessary padding to the request to match the server's batch size.
		req.pad_to_frame_count(n_frames_padded)

		# Send an inference request to the server and add it to the
		# list of pending requests.
		self.request_frame(req, sync=sync)

		tlog.info("Waiting for inference results (this may take a while)...")
		self.last_request_t = now
		self.last_request_id += 1


	def request_frame(self, req: InferenceRequest, sync: bool = False) -> asyncio.Task | InferenceResult:
		qp = "?sync=1" if sync else ""
		url = self.url + "/request-inference" + qp
		data = dumps_api_message(req)

		def req_done_cb(task_or_res: asyncio.Task | httpx.Response) -> None:
			if sync:
				res: httpx.Response = task_or_res
			else:
				try:
					res: httpx.Response = task_or_res.result()
				except RuntimeError as e:
					tlog.error(f"Inference request task failed!\n{e}")

			if res.status_code != 202:
				tlog.error(f"Inference request failed!\n{res.content}")

			if sync:
				return loads_api_message(
					res.content, allowed_types=(InferenceResult, CompressedInferenceResult)
				)
			else:
				if req.request_id not in self.pending_requests:
					tlog.error(f"Inference request {req.request_id} was created on the server,"
							   f" but it is not part of our pending requests"
							   f" (pending: {list(self.pending_requests.keys())})")

				state = self.pending_requests[req.request_id]
				state.state = RequestState.REQUEST_SENT
				state.task = None

		task_or_res = httpx_request(
			"post", url, data=data, timeout=self.req_timeout_s,
			headers={"Content-Type": API_MEDIA_TYPE, "Accept": API_MEDIA_TYPE},
			async_client=(None if sync else self.client),
			callback=req_done_cb
		)
		if not sync:
			self.pending_requests[req.request_id] = PendingRequest(
				request_id=req.request_id,
				state=RequestState.REQUEST_PENDING,
				task=task_or_res,
			)
		return task_or_res


	def _get_inference_results(self, request_id: str, on_result_received: callable) -> asyncio.Task:
		# ------------------------------------------------------------------
		# Stream inference results in a background thread so downloads
		# are not throttled by the synchronous `testbed.frame()` render loop.
		#
		# Also avoid allocating a giant `res.content` buffer during transfer: write
		# the allowlisted JSON payload to a temporary file before decoding it.
		progress_state = {"p": 0.0}
		progress_lock = threading.Lock()

		url = self.url + f"/inference-result?request_id={request_id}"
		timeout_s = float(10 * self.req_timeout_s)
		chunk_size = int(self._http_stream_chunk_size)
		progress_interval_s = float(self._progress_interval_s)

		def _download_serialized_result() -> tuple[object | None, httpx.Response, bool]:
			"""
			Returns (result_or_none, response, failed_flag).
			- 503 => (None, response, False)
			- non-200/non-503 => (None, response, True)
			- 200 => (result, response, False)
			"""
			with httpx.Client(timeout=timeout_s) as client:
				with client.stream("GET", url, headers={"Accept": API_MEDIA_TYPE}) as res:
					status = int(res.status_code)
					# For non-200 we only need a small error body for logs/UI.
					if status != 200:
						try:
							msg = res.read().decode("utf-8", errors="replace")
						except Exception:
							msg = ""
						resp = httpx.Response(status_code=status, content=msg.encode("utf-8", errors="ignore"))
						return None, resp, (status != 503)

					# 200: stream to a temporary file, then decode the safe JSON message.
					total = res.headers.get("Content-Length", None)
					try:
						total_i = int(total) if total is not None else None
					except Exception:
						total_i = None

					last_cb_t = 0.0
					downloaded = 0
					tmp_fd, tmp_path = tempfile.mkstemp(prefix="lyra2_inference_", suffix=".json")
					try:
						with os.fdopen(tmp_fd, "wb") as f:
							for chunk in res.iter_bytes(chunk_size=chunk_size):
								if not chunk:
									continue
								f.write(chunk)
								downloaded += len(chunk)
								if total_i and total_i > 0:
									now = time.monotonic()
									if (now - last_cb_t) >= progress_interval_s:
										with progress_lock:
											progress_state["p"] = min(1.0, float(downloaded) / float(total_i))
										last_cb_t = now
						# Final progress update
						with progress_lock:
							progress_state["p"] = 1.0

						with open(tmp_path, "rb") as rf:
							result = loads_api_message(
								rf.read(), allowed_types=(InferenceResult, CompressedInferenceResult)
							)
					finally:
						try:
							os.remove(tmp_path)
						except Exception:
							pass

					# Re-create a lightweight response for downstream error handling / headers access.
					resp = httpx.Response(status_code=200, headers=dict(res.headers))
					return result, resp, False

		async def _pump_progress_until_done(task: asyncio.Task) -> None:
			# If Content-Length is unknown, we still show an indeterminate-ish progress (0..1 not meaningful).
			while not task.done():
				with progress_lock:
					p = float(progress_state["p"])
				self.testbed.gen3c_inference_progress = p
				await asyncio.sleep(0.05)

		async def _runner() -> None:
			# Show progress bar while downloading.
			self.testbed.gen3c_inference_progress = 0.0
			self.testbed.gen3c_inference_info = "Downloading inference results..."
			try:
				download_task = asyncio.create_task(asyncio.to_thread(_download_serialized_result))
				pump_task = asyncio.create_task(_pump_progress_until_done(download_task))
				result, response, failed = await download_task
				try:
					await pump_task
				except Exception:
					pass
			except Exception as e:
				# Treat thread failure as a hard failure.
				response = httpx.Response(status_code=500, content=str(e).encode("utf-8", errors="ignore"))
				result = None
				failed = True
			finally:
				# Hide the progress bar (regardless of success or failure)
				self.testbed.gen3c_inference_progress = -1.0

			# Preserve old semantics for caller.
			if response.status_code == 503:
				on_result_received(result=None, response=response)
			elif response.status_code != 200:
				on_result_received(result=None, response=response, failed=True)
			else:
				on_result_received(result, response=response)

		return asyncio.create_task(_runner())

	# ----------

	def load_seeding_data(self, seeding_data_path: str, display: bool = True,
						  normalize_cameras: bool = False) -> SeedingRequest:

		if not os.path.exists(seeding_data_path):
			tlog.error(f"Cannot seed with invalid path: \"{seeding_data_path}\"")
			return None

		tlog.info(f"Seeding model from \"{seeding_data_path}\"")

		req = load_v2v_seeding_data(seeding_data_path, max_frames=self.seed_max_frames,
								    frames_stride=self.seed_stride)

		# Seed-from-image: crop the loaded image in-memory to match server inference aspect ratio.
		# (Avoids rewriting the user's file; `load_v2v_seeding_data()` already produced `req.images`.)
		if (not os.path.isdir(seeding_data_path)) and (not isinstance(req, CompressedSeedingRequest)) and (len(req) == 1):
			try:
				img = req.images[0]
				h, w = int(img.shape[0]), int(img.shape[1])
				tw, th = int(self.inference_resolution[0]), int(self.inference_resolution[1])
				if h > 0 and w > 0 and tw > 0 and th > 0 and (w * th) != (h * tw):
					g = math.gcd(tw, th)
					rw = max(1, tw // g)
					rh = max(1, th // g)
					k = min(w // rw, h // rh)
					if k > 0:
						new_w = int(k * rw)
						new_h = int(k * rh)
						x0 = int((w - new_w) // 2)
						y0 = int((h - new_h) // 2)

						req.images = req.images[:, y0:y0 + new_h, x0:x0 + new_w, ...]
						if req.masks is not None:
							req.masks = req.masks[:, y0:y0 + new_h, x0:x0 + new_w]
						if req.depths is not None:
							req.depths = req.depths[:, y0:y0 + new_h, x0:x0 + new_w]

						# Update intrinsics/resolution (same logic as v2v_utils.py for still images).
						req.resolutions = np.array([[new_w, new_h]], dtype=req.resolutions.dtype)
						fov_y_rad = np.pi * (50.625 / 180.0)
						f = 0.5 / (np.tan(fov_y_rad / 2.0)) * float(new_h)
						req.focal_lengths = np.array([[f, f]], dtype=np.float32)
						req.principal_points = np.full((1, 2), 0.5, dtype=np.float32)

						tlog.info(f"Seed image cropped to inference aspect {rw}:{rh} ({w}x{h} -> {new_w}x{new_h}).")
			except Exception as e:
				tlog.warning(f"Seed image crop skipped due to error: {e}")

		if normalize_cameras:
			if isinstance(req, CompressedSeedingRequest):
				raise NotImplementedError("Normalizing cameras not implemented for compressed seeding data")

			# Post-process the cameras so that they are centered at (0.5, 0.5, 0.5)
			# and so that they fit within a reasonable scale.
			current_origins = req.cameras_to_world[:, :3, 3]
			current_center = np.mean(current_origins, axis=0)
			current_scale = np.mean(np.linalg.norm(current_origins, axis=1))
			# TODO: robust scale estimation using the median depth as well
			if req.depths is not None:
				median_depth = np.nanmedian(req.depths)
				current_scale = max(current_scale, median_depth)

			# tlog.debug(f"Current scale: {current_scale}")

			if current_scale != 0.0:
				normalized_origins = (current_origins - current_center) / current_scale

				new_center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
				# aabb_scale = np.linalg.norm(self.testbed.render_aabb.max - self.testbed.render_aabb.min)
				# new_scale = aabb_scale / 4
				new_scale = 1.0
				req.cameras_to_world[:, :3, 3] = (normalized_origins * new_scale) + new_center

				# Rescale the depth values by the same
				req.depths *= new_scale / current_scale
			# TODO: retain this information so that we can undo the transform when
			# communicating with the server or saving stuff out.

		if display and (req.depths is not None):
			# If there's not depth data available at this point, we'll download it from the server
			# when seeding is done, and display the frames then.
			self.display_seeding_data(req, save_frames=self.testbed.gen3c_save_frames)

		return req


	def display_seeding_data(self, req: SeedingRequest, res: SeedingResult | None = None,
							 save_frames: bool = False) -> None:
		self.testbed.clear_src_views()

		if isinstance(req, CompressedSeedingRequest):
			# Since the de-compression is done inline, we make sure not to
			# populate uncompressed data in the request before sending it over.
			req = deepcopy(req)
			req.decompress()

		images = req.images
		depths = req.depths
		masks = getattr(req, "masks", None)
		if res is not None:
			# Adopt extrinsics and intrinsics from the server, the model might
			# have estimated them better than our hardcoded guess.
			focal_lengths = res.focal_lengths.copy()
			cameras_to_world = res.cameras_to_world
			principal_points = res.principal_points

			if res.depths is not None:
				# TODO: the depth estimated by the server may have a completely different scale.
				depths = res.depths
				if res.depths.shape[1:] != images.shape[1:3]:
					# Depth prediction took place on the server at a different resolution,
					# let's resize the RGB images to match.
					tlog.debug(f"Resizing seeding images for display to match depth resolution {depths.shape[1:3]}")
					resized = []
					for i in range(len(req)):
						resized.append(
							cv2.resize(images[i, ...], (depths.shape[2], depths.shape[1]),
									   interpolation=cv2.INTER_CUBIC)
						)
						# Let's assume that the inference server already adjusted the intrinsics
						# to match the requested inference resolution.
						# focal_lengths[i, 0] *= depths.shape[2] / images.shape[2]
						# focal_lengths[i, 1] *= depths.shape[1] / images.shape[1]
					images = np.stack(resized, axis=0)
			if getattr(res, "masks", None) is not None:
				masks = res.masks
		else:
			focal_lengths = req.focal_lengths.copy()
			cameras_to_world = req.cameras_to_world
			principal_points = req.principal_points


		if save_frames:
			os.makedirs(self.testbed.gen3c_output_dir, exist_ok=True)
		for seed_i in range(len(req)):
			res_id = f"seeding_{seed_i:04d}"
			image = ensure_alpha_channel(images[seed_i, ...])

			if save_frames:
				safe_res_id = res_id
				fname = join(self.testbed.gen3c_output_dir, f"rgb_{safe_res_id}.exr")
				pyexr.write(fname, image)

				if depths is not None:
					fname = join(self.testbed.gen3c_output_dir, f"depth_{safe_res_id}.exr")
					pyexr.write(fname, depths[seed_i, ...].astype(np.float32))
					tlog.success(f"[+] Wrote seeding frame to: {fname}")

			if depths is None:
				# Still no depth values available, cannot display
				continue

			depth_hw = depths[seed_i, ...]
			if masks is not None and seed_i < masks.shape[0]:
				mask_hw = masks[seed_i, ...].astype(np.bool_)
				if mask_hw.shape == depth_hw.shape:
					depth_hw = depth_hw.copy()
					depth_hw[~mask_hw] = 0.0

			self.testbed.add_src_view(
				cameras_to_world[seed_i, ...],
				focal_lengths[seed_i][0],
				focal_lengths[seed_i][1],
				principal_points[seed_i][0],
				principal_points[seed_i][1],
				self.lens,
				image,
				depth_hw,
				# TODO: seeding request could also have timestamps
				seed_i * 1 / 30,
				is_srgb=True,
			)
			tlog.success(f"[+] Displaying seeding view: {res_id}")

		tlog.info(f"Setting camera path from seeding view.")
		# First, initialize the camera path from all seeding view.
		self.set_max_number_of_displayed_views(len(req))
		self.testbed.init_camera_path_from_reproject_src_cameras()
		# Then, limit the number of displayed views so that rendering doesn't slow down too much.
		self.set_max_number_of_displayed_views(100)
		self.testbed.reset_accumulation(reset_pip=True)

		# Store seed frame info for predicted-pose updates and revert.
		if cameras_to_world is not None and len(cameras_to_world) > 0:
			self._seed_c2w = cameras_to_world[0].copy()
			self._seed_focal_lengths = focal_lengths[0].copy()
			self._seed_principal_points = principal_points[0].copy()
			if images is not None and len(images) > 0:
				self._seed_image_rgba = ensure_alpha_channel(images[0])
			if depths is not None and len(depths) > 0:
				d = depths[0].copy()
				if masks is not None and 0 < masks.shape[0]:
					m = masks[0].astype(np.bool_)
					if m.shape == d.shape:
						d = d.copy()
						d[~m] = 0.0
				self._seed_depth_hw = d.astype(np.float32)
			else:
				self._seed_depth_hw = None

		# Reset generation workflow state: after (re-)seeding, disable "Generate video" until the
		# user adds >=1 new camera/keyframe on top of this baseline.
		self.testbed.camera_path.locked_prefix = 0
		self._has_generated_once = False
		self._pre_gen_snapshot = None
		self.testbed.gen3c_revert_available = False
		self._seed_baseline_keyframes = len(self.testbed.camera_path.keyframes)
		self._update_generate_video_enabled()

		# Update the viewport FOV to match the seeding data
		if len(req) > 0:
			# Assume constant focal length across seeding frames for the viewport setup
			axis = self.testbed.fov_axis
			# Use the resolution that corresponds to the focal lengths
			if res is not None:
				dim = res.resolutions[0, axis]
			else:
				dim = req.resolutions[0, axis]

			fl = focal_lengths[0][axis]
			if fl > 0:
				fov_rad = 2 * np.arctan(dim / (2 * fl))
				self.testbed.fov = np.degrees(fov_rad)


	def send_seeding_request(self, req: SeedingRequest, sync: bool = False) -> asyncio.Task | None:
		"""
		Note: we do seeding requests synchronously by default so that we don't have to implement
		eager checking, etc.
		"""

		qp = "?sync=1" if sync else ""
		url = self.url + "/seed-model" + qp
		depth_was_missing = (req.depths is None)

		def req_done_cb(task_or_res: asyncio.Task | httpx.Response) -> None:
			# Hide the progress bar (regardless of success or failure)
			self.testbed.gen3c_seeding_progress = -1.0
			if sync:
				res: httpx.Response = task_or_res
			else:
				try:
					res: httpx.Response = task_or_res.result()
				except RuntimeError as e:
					tlog.error(f"Seeding request task failed!\n{e}")
					return

			if res.status_code >= 300:
				tlog.error(f"Seeding request failed!\n{res.content}")
				return None

			if depth_was_missing:
				response: SeedingResult = loads_api_message(res.content, allowed_types=(SeedingResult,))
				self.display_seeding_data(req, res=response, save_frames=self.testbed.gen3c_save_frames)

			message = "Model seeded."
			self.testbed.gen3c_info = "\n".join([
				self.testbed.gen3c_info.split("\n")[0],
				message
			])
			tlog.success(message)
			self._seeding_complete = True
			self._update_generate_video_enabled()

		def progress_cb(progress: float, **kwargs):
			self.testbed.gen3c_seeding_progress = progress

		if not isinstance(req, CompressedSeedingRequest):
			req = req.compress()

		data = dumps_api_message(req)
		try:
			progress_direction = "both" if depth_was_missing else "auto"
			return httpx_request("post", url, data=data, timeout=self.req_timeout_s,
								 headers={"Content-Type": API_MEDIA_TYPE, "Accept": API_MEDIA_TYPE},
								 progress=True, progress_direction=progress_direction,
								 desc="Seeding",
								 async_client=(None if sync else self.client),
								 callback=req_done_cb,
								 progress_callback=progress_cb)
		except (httpx.TimeoutException, httpx.ConnectError) as e:
			tlog.error(f"Seeding request failed (timeout or connection error)!\n{e}")
			return None

	# ----------

	def set_max_number_of_displayed_views(self, n_views: int) -> None:
		tlog.info(f"Setting max number of displayed views to {n_views}")
		# Jump to the last view.
		self.testbed.reproject_max_src_view_index = min(self.testbed.reproject_src_views_count(), n_views)

	def _transfer_in_progress(self) -> bool:
		return (self.testbed.gen3c_inference_progress >= 0.0) or (self.testbed.gen3c_seeding_progress >= 0.0)

	# ----------

	def adapt_view_to_cameras(self, cameras_to_world: np.ndarray,
							  go_to_default_camera: bool = True) -> None:
		"""
		Analyzes the given set of cameras, and tries to adapt the current
		up vector, default camera pose, etc to match.

		Note: this hasn't been tested very thoroughly yet and could easily
		do the wrong thing depending on the inputs.
		"""
		assert cameras_to_world.shape[1:] == (3, 4)

		# Note: up_dir is intentionally not modified here. The testbed's m_up_dir default
		# ({0, 1, 0}) is set correctly in C++. Auto-detecting it from camera matrices is
		# unreliable because the result depends on the camera convention (OpenCV Y-down vs
		# OpenGL Y-up), and getting it wrong reverses left-right camera controls.

		# --- Default camera pose
		default_c2w = cameras_to_world[0, :3, :]

		# Note: `default_camera` is technically a 4x3 camera, but the bindings
		# expose it as a 3x4 matrix, so we can set it as normal here.
		self.testbed.default_camera = default_c2w
		tlog.debug(f"Based on the seeding data, setting up dir to {self.testbed.up_dir}"
				   f" and default camera to:\n{self.testbed.default_camera}")

		if go_to_default_camera:
			self.testbed.reset_camera()



	# ------------------------------------------------------------------ #
	# One-level undo for the last generation
	# ------------------------------------------------------------------ #

	def _save_pre_generate_state(self) -> None:
		"""Capture client-side state that will be reverted on undo."""
		cp = self.testbed.camera_path
		self._pre_gen_snapshot = {
			"keyframes": [
				ngp.CameraKeyframe(kf.m(), float(kf.fov), float(kf.timestamp), kf.up_dir)
				for kf in cp.keyframes
			],
			"locked_prefix": int(cp.locked_prefix),
			"has_generated_once": self._has_generated_once,
			"src_view_count": int(self.testbed.reproject_src_views_count()),
			"reproject_min_src_view_index": int(self.testbed.reproject_min_src_view_index),
			"reproject_max_src_view_index": int(self.testbed.reproject_max_src_view_index),
		}

	def _revert_last_generation(self) -> None:
		"""Revert client + server to the state before the last generation."""
		snap = getattr(self, "_pre_gen_snapshot", None)
		if snap is None:
			tlog.warning("No pre-generation snapshot available to revert.")
			return

		# 1) Ask the server to revert its pipeline state.
		url = self.url + "/revert-last-generation"
		try:
			res = httpx_request(
				"post", url, timeout=self.req_timeout_s, headers={"Accept": API_MEDIA_TYPE}
			)
			if res.status_code == 200:
				result: RevertResult = loads_api_message(res.content, allowed_types=(RevertResult,))
				if result.success:
					tlog.success("Server reverted successfully.")
				else:
					tlog.error(f"Server revert failed: {result.message}")
					return
			else:
				tlog.error(f"Server revert HTTP error {res.status_code}: {res.content!r}")
				return
		except Exception as e:
			tlog.error(f"Server revert request failed: {e}")
			return

		# 2) Restore client-side keyframes.
		cp = self.testbed.camera_path
		n_kf_before = len(cp.keyframes)
		cp.keyframes = snap["keyframes"]
		cp.locked_prefix = snap["locked_prefix"]
		cp.sanitize_keyframes()
		self._has_generated_once = snap["has_generated_once"]
		tlog.info(
			f"Keyframes: {n_kf_before} -> {len(cp.keyframes)}, "
			f"locked_prefix: {cp.locked_prefix}, has_generated_once: {self._has_generated_once}"
		)

		# 3) Trim src_views back to the pre-generation count (keeps views from
		#    earlier generations intact) and restore min/max display indices.
		#    trim_src_views also invalidates PatchMatch state to prevent stale
		#    view indices from causing out-of-bounds GPU reads.
		old_count = snap["src_view_count"]
		cur_count = int(self.testbed.reproject_src_views_count())
		self.testbed.trim_src_views(old_count)
		self.testbed.reproject_min_src_view_index = snap["reproject_min_src_view_index"]
		self.testbed.reproject_max_src_view_index = snap["reproject_max_src_view_index"]
		self.testbed.cuda_device_synchronize()
		tlog.info(
			f"Src views: {cur_count} -> {old_count}, "
			f"display range: [{snap['reproject_min_src_view_index']}, {snap['reproject_max_src_view_index']})"
		)

		self.testbed.reset_accumulation(reset_pip=True)
		self.testbed.gen3c_revert_available = False
		self._update_generate_video_enabled()

		# Consume the snapshot (single-level undo).
		self._pre_gen_snapshot = None
		tlog.success("Reverted to pre-generation state.")

	def gui_callback(self, event: str) -> bool:
		match event:
			case "seed_model":
				seed_req = self.load_seeding_data(self.testbed.gen3c_seed_path)
				if seed_req is not None:
					self.adapt_view_to_cameras(seed_req.cameras_to_world)
					self.send_seeding_request(seed_req)
				# "True" means we handled the event, not that seeding was successful.
				return True

			case "request_inference":
				self.testbed.gen3c_camera_source = ngp.Gen3cCameraSource.Authored

				self._update_generate_video_enabled()
				if not self.testbed.gen3c_generate_video_enabled:
					self.testbed.gen3c_inference_info = (
						"Generate video is disabled: add at least one new camera keyframe first."
					)
					return True

				# Save client-side state for one-level undo.
				self._save_pre_generate_state()

				cp = self.testbed.camera_path
				keyframes = list(cp.keyframes)

				if not self._has_generated_once:
					self.request_frames(sync=False)
					return True

				prefix = int(cp.locked_prefix)
				if prefix <= 0 or prefix >= len(keyframes):
					self.testbed.gen3c_inference_info = (
						"Generate video is disabled: add at least one new camera keyframe first."
					)
					self._update_generate_video_enabled()
					return True

				anchor_i = max(prefix - 1, 0)
				segment_keyframes = [ngp.CameraKeyframe(kf.m(), float(kf.fov), float(kf.timestamp), kf.up_dir) for kf in keyframes[anchor_i:]]
				segment_keyframes = self._duplicate_endpoints_for_sampling(segment_keyframes)
				t0 = float(segment_keyframes[0].timestamp)
				for kf in segment_keyframes:
					kf.timestamp = float(kf.timestamp) - t0
				segment_duration_s = float(cp.duration_seconds)
				if segment_duration_s <= 0:
					segment_duration_s = None

				self._with_temporary_camera_path(
					keyframes=segment_keyframes,
					segment_duration_s=segment_duration_s,
					fn=lambda: self.request_frames(sync=False),
				)
				return True

			case "abort_inference":
				tlog.info("Aborting inference request...")
				tlog.error("Not implemented yet: aborting an ongoing inference request. Ignoring.")
				return True

			case "revert_last_generation":
				self._revert_last_generation()
				return True

		return False

	def file_drop_callback(self, paths: list[str]) -> bool:
		tlog.info(f"Received {len(paths)} file{'s' if len(paths) > 1 else ''} via drag & drop: {paths}")
		for path in paths:
			ext = os.path.splitext(path)[1].lower()
			if os.path.isdir(path) or ext in (".jpg", ".jpeg", ".png", ".exr"):
				self.testbed.gen3c_seed_path = path
				self.seeding_pending = True
			elif ext == ".json":
				try:
					self.testbed.load_camera_path(path)
				except RuntimeError as e:
					tlog.error(f"Error loading camera path, perhaps the formata is incorrect?\n\t{e}")
			else:
				tlog.error(f"Don't know how to handle given file: {path}")
		return True



if __name__ == "__main__":
	parser = argparse.ArgumentParser("client.py")
	parser.add_argument("files", nargs="*",
						help="Files to be loaded. Can be a camera path, scene name,"
							 " seed image, or pre-processed video directory.")
	parser.add_argument("--host", default="127.0.0.1")
	parser.add_argument("--port", type=int, default=8000)
	parser.add_argument("--request-latency-ms", "--latency", type=int, default=250)
	parser.add_argument("--inference-resolution", nargs=2, type=int, default=(576, 320))
	parser.add_argument("--vr", action="store_true")
	parser.add_argument("--seed-max-frames", type=int, default=None,
						help="If seeding from a video, maximum number of frames to use.")
	parser.add_argument("--seed-stride", type=int, default=1,
						help="If seeding from a video, number of frames to skip when reading (stride).")
	parser.add_argument("--output-dir", "-o", type=str, default=None,
						help="Directory in which to save the inference results.")
	args = parser.parse_args()

	client = Lyra2Client(**vars(args))
	asyncio.run(client.run())
