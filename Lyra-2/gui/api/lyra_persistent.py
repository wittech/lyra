# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful Lyra-2 adapter used by the interactive GUI server."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import cv2
from loguru import logger as log
import numpy as np
import torch
import torch.nn.functional as F

from lyra_2._ext.imaginaire.utils import misc
from lyra_2._src.datasets.forward_warp_utils_pytorch import forward_warp_multiframes
from lyra_2._src.inference.lyra2_ar_inference import Lyra2InferencePipeline
_cudnn_enabled_before_inference_imports = torch.backends.cudnn.enabled
from lyra_2._src.inference.lyra2_custom_traj_inference import DMD_LORA_PATH, DMD_LORA_WEIGHT
from lyra_2._src.inference.lyra2_zoomgs_inference import _da3_infer_depth_intrinsics_single
# Both inference modules disable cuDNN globally at import time.  The GUI only
# reuses constants and a DA3 helper from them, so preserve the server's prior
# backend state instead of forcing the VAE Conv3D layers onto the high-memory
# native fallback.
torch.backends.cudnn.enabled = _cudnn_enabled_before_inference_imports
from lyra_2._src.utils.model_loader import load_model_from_checkpoint

from qwen_captioner import DEFAULT_QWEN_MODEL, QwenCaptioner


def _env_bool(name: str, default: bool = False) -> bool:
	value = os.environ.get(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str) -> str:
	return "".join(c if c.isalnum() or c in "-_" else "_" for c in value).strip("_") or "session"


def _find_ffmpeg() -> str:
	ffmpeg = shutil.which("ffmpeg")
	if ffmpeg is not None:
		return ffmpeg
	try:
		import imageio_ffmpeg
		return imageio_ffmpeg.get_ffmpeg_exe()
	except Exception as error:
		raise RuntimeError("ffmpeg is required by the Lyra GUI server.") from error


class Lyra2PersistentModel:
	"""Keep a Lyra2InferencePipeline alive across GUI inference requests."""

	def __init__(
		self,
		*,
		checkpoint_path: str,
		output_root: str,
		qwen_model: str = DEFAULT_QWEN_MODEL,
	) -> None:
		self.checkpoint_path = checkpoint_path
		self.output_root = Path(output_root).resolve()
		self.output_root.mkdir(parents=True, exist_ok=True)
		# Lyra-2 was trained and released with this fixed GUI inference size.
		# Do not silently lower it to work around memory pressure.
		self.target_h = 448
		self.target_w = 768
		self.frames_per_batch = 80
		self.inference_overlap_frames = 0
		self.pipeline: Lyra2InferencePipeline | None = None
		self.seed_image_bchw: torch.Tensor | None = None
		self.seed_depth_hw: torch.Tensor | None = None
		self.seed_mask_hw: torch.Tensor | None = None
		self.seed_w2c: torch.Tensor | None = None
		self.seed_intrinsics: torch.Tensor | None = None
		self.manual_t5: torch.Tensor | None = None
		self._session_dir: Path | None = None
		self._chunk_paths: list[Path] = []
		self._pre_generate_chunk_count = 0

		use_dmd = _env_bool("LYRA_GUI_USE_DMD", False)
		offload = _env_bool("LYRA_GUI_OFFLOAD", True)
		self.args = argparse.Namespace(
			experiment=os.environ.get("LYRA_GUI_EXPERIMENT", "lyra2"),
			guidance=float(os.environ.get("LYRA_GUI_GUIDANCE", 5.0)),
			shift=float(os.environ.get("LYRA_GUI_SHIFT", 5.0)),
			num_sampling_step=int(os.environ.get("LYRA_GUI_SAMPLING_STEPS", 35)),
			seed=int(os.environ.get("LYRA_GUI_SEED", 1)),
			offload=offload,
			offload_vae=offload and _env_bool("LYRA_GUI_OFFLOAD_VAE", True),
			vae_decode_device=os.environ.get(
				"LYRA_GUI_VAE_DECODE_DEVICE",
				"cuda:0",
			),
			use_dmd_scheduler=use_dmd,
			depth_backend="da3",
			num_retrieval_views=int(os.environ.get("LYRA_GUI_RETRIEVAL_VIEWS", 1)),
			multiview_ids=None,
			da3_model_name=os.environ.get(
				"LYRA_GUI_DA3_MODEL", "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
			),
			da3_model_path_custom=os.environ.get("LYRA_GUI_DA3_CHECKPOINT_PATH"),
			da3_frame_interval=int(os.environ.get("LYRA_GUI_DA3_FRAME_INTERVAL", 4)),
			da3_max_history_frames=int(os.environ.get("LYRA_GUI_DA3_MAX_HISTORY", 40)),
			da3_include_ar_chunk_last_frames=True,
			da3_use_predicted_pose=_env_bool("LYRA_GUI_DA3_USE_PREDICTED_POSE", True),
			da3_predicted_pose_continuation=_env_bool(
				"LYRA_GUI_DA3_PREDICTED_POSE_CONTINUATION", False
			),
			disable_cache_update=False,
			offload_da3_diffusion=True,
			offload_da3_model=True,
			dynamic_prompt_num_warps=int(
				os.environ.get("LYRA_GUI_DYNAMIC_PROMPT_NUM_WARPS", 4)
			),
			dynamic_prompt_num_source_frames=int(
				os.environ.get("LYRA_GUI_DYNAMIC_PROMPT_NUM_SOURCE_FRAMES", 3)
			),
		)

		if self.args.da3_model_path_custom is None:
			local_da3 = Path("checkpoints/recon/model.pt")
			if local_da3.is_file():
				self.args.da3_model_path_custom = str(local_da3.resolve())

		self.video_model = self._load_video_model(use_dmd=use_dmd)
		# Keep this process on cuDNN's heuristic convolution path.  Restoring
		# benchmark mode between VAE calls lets unrelated Lyra convolutions cache
		# Conv3D plans with workspaces that exceed an 80GB H100 during AR decode.
		torch.backends.cudnn.enabled = True
		torch.backends.cudnn.benchmark = False
		self.device = torch.device(self.video_model.tensor_kwargs.get("device", "cuda"))
		self.dtype = self.video_model.tensor_kwargs.get("dtype", torch.bfloat16)

		negative_path = Path(os.environ.get(
			"LYRA_GUI_NEGATIVE_PROMPT_PATH", "checkpoints/text_encoder/negative_prompt.pt"
		))
		if not negative_path.is_file():
			raise FileNotFoundError(
				f"Missing negative prompt embedding: {negative_path}. Download the public Lyra-2 checkpoints first."
			)
		self.negative_prompt_data = torch.load(negative_path, map_location="cpu", weights_only=False)

		from lyra_2._src.inference.depth_utils import load_da3_model
		self.da3_model = load_da3_model(
			da3_model_name=self.args.da3_model_name,
			da3_model_path_custom=self.args.da3_model_path_custom,
			device=self.device,
		)
		self.da3_model.eval()

		log.info(f"Loading public VLM {qwen_model} on CPU.")
		self.captioner = QwenCaptioner(model_name=qwen_model, device=str(self.device))
		self.manual_prompt = os.environ.get("LYRA_GUI_PROMPT", "").strip()

	def _load_video_model(self, *, use_dmd: bool):
		experiment_opts = [
			"model.config.use_mp_policy_fsdp=False",
			"model.config.keep_original_net_dtype=False",
		]
		if use_dmd:
			experiment_opts += ["model.config.net.postpone_checkpoint=True"]

		model, _ = load_model_from_checkpoint(
			config_file="lyra_2/_src/configs/config.py",
			experiment_name=self.args.experiment,
			checkpoint_path=self.checkpoint_path,
			enable_fsdp=False,
			instantiate_ema=False,
			load_ema_to_reg=False,
			experiment_opts=experiment_opts,
		)
		if use_dmd:
			if not Path(DMD_LORA_PATH).is_file():
				raise FileNotFoundError(f"LYRA_GUI_USE_DMD=1 but {DMD_LORA_PATH} is missing.")
			name = model.load_lora_weights(DMD_LORA_PATH)
			model.set_weights_and_activate_adapters([name], [DMD_LORA_WEIGHT])
			if hasattr(model.net, "enable_selective_checkpoint"):
				model.net.enable_selective_checkpoint(model.net.sac_config, model.net.blocks)

		dtype = model.tensor_kwargs.get("dtype")
		device = model.tensor_kwargs.get("device")
		if dtype is not None:
			model.net = model.net.to(device=device, dtype=dtype)
		model.eval()
		return model

	@contextmanager
	def _video_net_on_cpu(self, *, also_da3: bool):
		"""Temporarily free GPU memory for DA3, Qwen, or UMT5."""
		clip_modules = []
		for embedder in getattr(self.video_model.conditioner, "embedders", {}).values():
			clip = getattr(embedder, "clip_model", None)
			clip_module = getattr(clip, "model", None)
			if isinstance(clip_module, torch.nn.Module):
				try:
					original_device = next(clip_module.parameters()).device
				except StopIteration:
					original_device = torch.device("cpu")
				clip_modules.append((clip_module, original_device))
		da3_device = None
		if also_da3:
			try:
				da3_device = next(self.da3_model.parameters()).device
			except StopIteration:
				da3_device = torch.device("cpu")
		self.video_model.net.to("cpu")
		for clip_module, _ in clip_modules:
			clip_module.to("cpu")
		if also_da3:
			self.da3_model.to("cpu")
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		try:
			yield
		finally:
			if also_da3 and da3_device is not None:
				self.da3_model.to(da3_device)
			self.video_model.net.to(device=self.device, dtype=self.dtype)
			for clip_module, original_device in clip_modules:
				clip_module.to(original_device)
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	def _embed_prompt(self, caption: str) -> torch.Tensor:
		from lyra_2._src.inference.get_t5_emb import get_umt5_embedding_offloaded

		t5 = get_umt5_embedding_offloaded(caption, device=str(self.device)).to(dtype=self.dtype)
		if t5.dim() == 2:
			t5 = t5.unsqueeze(0)
		return t5

	def _build_dynamic_caption_images(
		self,
		cam_w2c: torch.Tensor,
		intrinsics: torch.Tensor,
	) -> list[torch.Tensor]:
		"""Build seed/current/trajectory-projection inputs used by dynamic captioning."""
		assert self.pipeline is not None
		assert self.seed_image_bchw is not None
		assert self.seed_depth_hw is not None

		warp_device = self.device
		last_frame = self.pipeline.history_frames[:, :, -1].to(
			device=warp_device, dtype=torch.float32
		)
		_, _, height, width = last_frame.shape
		num_frames = int(cam_w2c.shape[1])
		num_warps = max(int(self.args.dynamic_prompt_num_warps), 1)
		points = torch.linspace(0, num_frames - 1, num_warps + 1).long().tolist()
		target_indices = sorted(set(points[1:]))
		if not target_indices:
			target_indices = [num_frames - 1]

		retrieved_frame_ids: list[int] = []
		num_source = max(int(self.args.dynamic_prompt_num_source_frames), 1)
		cache = self.pipeline.retrieval_cache
		use_multiframe = (
			num_source > 1
			and cache is not None
			and len(cache._world_points) > 0
		)
		if use_multiframe:
			try:
				retrieved = cache.retrieve(
					cam_w2c[:, target_indices[-1]].to(warp_device),
					intrinsics[:, target_indices[-1]].to(warp_device),
					(height, width),
					num_latents=num_source - 1,
					skip_last_n=0,
					max_coverage=True,
				)
				retrieved_frame_ids = [int(frame_id) for _, frame_id in retrieved]
			except Exception as error:
				log.warning(f"Dynamic prompt cache retrieval failed; using latest frame only: {error}")
				retrieved_frame_ids = []
			use_multiframe = bool(retrieved_frame_ids)

		def as_b1hw(value: torch.Tensor) -> torch.Tensor:
			if value.dim() == 2:
				value = value[None, None]
			elif value.dim() == 3:
				value = value[:, None]
			return value.to(device=warp_device, dtype=torch.float32)

		last_depth = self.pipeline.buffer_depth_latest
		if last_depth is None:
			last_depth = self.seed_depth_hw[None, None]
		last_depth = as_b1hw(last_depth)
		last_mask = self.pipeline.buffer_mask_latest
		last_mask = as_b1hw(last_mask) if isinstance(last_mask, torch.Tensor) else None
		last_w2c = self.pipeline.cam_w2c[:, -1].to(warp_device, dtype=torch.float32)
		last_intrinsics = self.pipeline.intrinsics[:, -1].to(warp_device, dtype=torch.float32)

		warped_images: list[torch.Tensor] = []
		if use_multiframe:
			assert cache is not None
			source_rgb = []
			source_depth = []
			source_w2c = []
			source_intrinsics = []
			for frame_id in retrieved_frame_ids:
				if frame_id < 0:
					rgb = cache.get_rgb_by_frame_id(frame_id)
				else:
					history_index = int(self.pipeline.start_index) + frame_id
					rgb = self.pipeline.history_frames[:, :, history_index]
				depth, w2c, camera_intrinsics = cache.get_rgbd_by_frame_id(frame_id)
				depth = as_b1hw(depth)
				if depth.shape[-2:] != (height, width):
					depth = F.interpolate(depth, (height, width), mode="nearest")
				source_rgb.append(rgb.to(warp_device, dtype=torch.float32))
				source_depth.append(depth)
				source_w2c.append(w2c.to(warp_device, dtype=torch.float32))
				source_intrinsics.append(camera_intrinsics.to(warp_device, dtype=torch.float32))

			source_rgb.append(last_frame)
			source_depth.append(last_depth)
			source_w2c.append(last_w2c)
			source_intrinsics.append(last_intrinsics)
			rgb_bvchw = torch.stack(source_rgb, dim=1)
			depth_bv1hw = torch.stack(source_depth, dim=1)
			w2c_bv44 = torch.stack(source_w2c, dim=1)
			intrinsics_bv33 = torch.stack(source_intrinsics, dim=1)

			for target_index in target_indices:
				warped, _, _, _ = forward_warp_multiframes(
					frame1=rgb_bvchw,
					mask1=None,
					depth1=depth_bv1hw,
					transformation1=w2c_bv44,
					transformation2=cam_w2c[:, target_index].to(warp_device),
					intrinsic1=intrinsics_bv33,
					intrinsic2=intrinsics[:, target_index].to(warp_device),
					is_image=True,
					clean_points=True,
					clean_points_continuity=True,
				)
				warped_images.append(warped[0].detach().cpu())
		else:
			for target_index in target_indices:
				warped, _, _, _ = forward_warp_multiframes(
					frame1=last_frame[:, None],
					mask1=None if last_mask is None else last_mask[:, None],
					depth1=last_depth[:, None],
					transformation1=last_w2c[:, None],
					transformation2=cam_w2c[:, target_index].to(warp_device),
					intrinsic1=last_intrinsics[:, None],
					intrinsic2=intrinsics[:, target_index].to(warp_device),
					is_image=True,
					clean_points=True,
					clean_points_continuity=True,
				)
				warped_images.append(warped[0].detach().cpu())

		log.info(
			f"Dynamic caption inputs: seed + latest + {len(warped_images)} trajectory projections "
			f"from {len(retrieved_frame_ids) + 1 if use_multiframe else 1} source frame(s)."
		)
		return [
			self.seed_image_bchw[0].detach().float().cpu(),
			last_frame[0].detach().cpu(),
			*warped_images,
		]

	def _dynamic_caption_and_embed(
		self,
		cam_w2c: torch.Tensor,
		intrinsics: torch.Tensor,
		region_hint: str,
	) -> tuple[str, torch.Tensor]:
		if self.manual_prompt:
			if self.manual_t5 is None:
				with self._video_net_on_cpu(also_da3=True):
					self.manual_t5 = self._embed_prompt(self.manual_prompt)
			return self.manual_prompt, self.manual_t5

		with self._video_net_on_cpu(also_da3=True):
			images = self._build_dynamic_caption_images(cam_w2c, intrinsics)
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
			with self.captioner.on_device():
				caption = self.captioner.caption(images, region_hint=region_hint)
			log.info(f"Video prompt: {caption}")
			t5 = self._embed_prompt(caption)
		return caption, t5

	def _resize_seed(self, images_np: np.ndarray):
		image = np.asarray(images_np[0])
		if image.dtype != np.uint8:
			image = np.clip(image, 0.0, 1.0)
			image = (image * 255.0 + 0.5).astype(np.uint8)
		image = cv2.resize(image[..., :3], (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
		image_chw01 = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
		return image, image_chw01

	def seed_model_from_values(
		self,
		*,
		images_np: np.ndarray,
		depths_np: np.ndarray | None,
		masks_np: np.ndarray | None,
		world_to_cameras_np: np.ndarray,
		focal_lengths_np: np.ndarray,
		principal_point_rel_np: np.ndarray,
		resolutions: np.ndarray,
		request_id: str,
	) -> dict[str, np.ndarray]:
		if len(images_np) != 1:
			raise ValueError("The Lyra-2 GUI currently supports exactly one seed image.")
		self.clear_cache()
		image_u8, image_chw01 = self._resize_seed(images_np)

		if depths_np is None:
			with self._video_net_on_cpu(also_da3=False):
				image_chw01, depth_hw, K_33, mask_hw = _da3_infer_depth_intrinsics_single(
					self.da3_model,
					torch.from_numpy(image_u8),
					(self.target_h, self.target_w),
				)
		else:
			depth = torch.from_numpy(np.asarray(depths_np[0], dtype=np.float32))[None, None]
			depth_hw = F.interpolate(depth, (self.target_h, self.target_w), mode="bilinear", align_corners=False)[0, 0]
			mask_hw = (depth_hw > 0).float()
			if masks_np is not None:
				mask = torch.from_numpy(np.asarray(masks_np[0], dtype=np.float32))[None, None]
				mask_hw = F.interpolate(mask, (self.target_h, self.target_w), mode="nearest")[0, 0]
			K_np = np.zeros((3, 3), dtype=np.float32)
			K_np[0, 0], K_np[1, 1] = focal_lengths_np[0]
			K_np[0, 2] = principal_point_rel_np[0, 0] * resolutions[0, 0]
			K_np[1, 2] = principal_point_rel_np[0, 1] * resolutions[0, 1]
			K_np[2, 2] = 1.0
			K_np[0] *= self.target_w / float(resolutions[0, 0])
			K_np[1] *= self.target_h / float(resolutions[0, 1])
			K_33 = torch.from_numpy(K_np)

		self.seed_image_bchw = (image_chw01.to(self.device) * 2.0 - 1.0).to(self.dtype)
		self.seed_depth_hw = depth_hw.to(self.device, dtype=torch.float32)
		self.seed_mask_hw = mask_hw.to(self.device, dtype=torch.float32)
		self.seed_w2c = torch.from_numpy(world_to_cameras_np[0].astype(np.float32)).unsqueeze(0).to(self.device)
		self.seed_intrinsics = K_33.unsqueeze(0).to(self.device, dtype=torch.float32)

		negative_t5 = misc.to(
			self.negative_prompt_data["t5_text_embeddings"], **self.video_model.tensor_kwargs
		)
		padding_mask = torch.zeros(
			(1, 1, self.target_h, self.target_w), dtype=self.dtype, device=self.device
		)
		fps = torch.tensor([16], dtype=torch.int32, device=self.device)
		self.pipeline = Lyra2InferencePipeline(
			model=self.video_model,
			args=self.args,
			first_frame=self.seed_image_bchw.unsqueeze(2),
			first_depth=self.seed_depth_hw.unsqueeze(0).unsqueeze(0),
			first_cam_w2c=self.seed_w2c,
			first_intrinsics=self.seed_intrinsics,
			da3_model=self.da3_model,
			cp_group=None,
			base_t5_text_embeddings=None,
			base_neg_t5_text_embeddings=negative_t5,
			padding_mask=padding_mask,
			fps=fps,
		)
		# DA3 is only needed during seeding and post-generation cache updates.  Keeping it
		# on CPU during diffusion/VAE avoids overlapping its weights with the VAE peak.
		self.da3_model.to("cpu")
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

		timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
		self._session_dir = self.output_root / f"{timestamp}_{_safe_name(request_id)}"
		self._session_dir.mkdir(parents=True, exist_ok=True)
		self._chunk_paths.clear()

		K_out = K_33.cpu().numpy()
		return {
			"cameras_to_world": np.linalg.inv(world_to_cameras_np[:1])[:, :3],
			"focal_lengths": np.array([[K_out[0, 0], K_out[1, 1]]], dtype=np.float32),
			"principal_points": np.array(
				[[K_out[0, 2] / self.target_w, K_out[1, 2] / self.target_h]], dtype=np.float32
			),
			"resolutions": np.array([[self.target_w, self.target_h]], dtype=np.int32),
			"depths": self.seed_depth_hw.detach().cpu().numpy()[None],
			"masks": (self.seed_mask_hw.detach().cpu().numpy()[None] > 0.5),
		}

	def inference_on_cameras(
		self,
		view_cameras_w2cs: np.ndarray,
		view_camera_intrinsics: np.ndarray,
		*,
		fps: int | float,
		region_hint: str = "",
		save_buffer: bool = True,
		request_id: str,
	) -> dict[str, Any]:
		if self.pipeline is None:
			raise RuntimeError("Seed the model before inference.")
		num_frames = int(view_cameras_w2cs.shape[0])
		step_size = int(self.video_model.framepack_num_new_video_frames)
		if num_frames % step_size:
			raise ValueError(f"Expected a multiple of {step_size} camera poses, got {num_frames}.")

		cam_w2c = torch.from_numpy(view_cameras_w2cs.astype(np.float32)).unsqueeze(0).to(self.device)
		intrinsics = torch.from_numpy(view_camera_intrinsics.astype(np.float32)).unsqueeze(0).to(self.device)
		self.pipeline.fps = torch.tensor([fps], dtype=torch.int32, device=self.device)
		self.pipeline._predicted_pose_last_w2c = None
		self.pipeline.save_snapshot()
		self._pre_generate_chunk_count = len(self._chunk_paths)

		num_chunks = num_frames // step_size
		try:
			caption, dynamic_t5 = self._dynamic_caption_and_embed(
				cam_w2c, intrinsics, region_hint=region_hint
			)
			log.info(f"Generate request prompt: {caption}")
			for chunk_i in range(num_chunks):
				start = chunk_i * step_size
				end = start + step_size
				result = self.pipeline.autoregressive_step(
					cam_w2c_chunk=cam_w2c[:, start:end],
					intrinsics_chunk=intrinsics[:, start:end],
					t5_text_embeddings=dynamic_t5,
					neg_t5_text_embeddings=None,
				)
				if result.get("abort") or result.get("stop"):
					break
		except Exception:
			self.pipeline.revert_to_snapshot()
			raise

		video = self.pipeline.history_frames[:, :, -num_frames:].float().cpu().permute(0, 2, 1, 3, 4)
		warp_video = None
		if save_buffer and self.pipeline.warp_video_collect:
			warp_video = torch.cat(self.pipeline.warp_video_collect, dim=2)
			warp_video = warp_video[:, :, -num_frames:].float().cpu().permute(0, 2, 1, 3, 4)

		last_depth = self.pipeline.buffer_depth_latest
		last_mask = self.pipeline.buffer_mask_latest
		predicted_depth = None
		predicted_mask = None
		predicted_depth_indices = np.empty((0,), dtype=np.int32)
		if isinstance(last_depth, torch.Tensor):
			d = last_depth.detach().float()
			if d.dim() == 2:
				d = d[None, None]
			elif d.dim() == 3:
				d = d[:, None]
			d = F.interpolate(d, (self.target_h, self.target_w), mode="bilinear", align_corners=False)
			predicted_depth = d[:1].cpu()
			predicted_depth_indices = np.array([num_frames - 1], dtype=np.int32)
			if isinstance(last_mask, torch.Tensor):
				m = last_mask.detach().float()
				if m.dim() == 2:
					m = m[None, None]
				elif m.dim() == 3:
					m = m[:, None]
				predicted_mask = F.interpolate(m, (self.target_h, self.target_w), mode="nearest")[:1, 0].bool().cpu()
			valid_depth = torch.isfinite(predicted_depth) & (predicted_depth > 0)
			if predicted_mask is not None:
				valid_depth &= predicted_mask[:, None]
			valid_count = int(valid_depth.sum().item())
			total_count = int(valid_depth.numel())
			log.info(
				f"Point-cloud depth coverage: {valid_count}/{total_count} "
				f"pixels ({100.0 * valid_count / max(total_count, 1):.1f}%)."
			)

		chunk_path = self._save_chunk(video[0], fps=fps, request_id=request_id)
		full_video_path = self._rebuild_full_video()
		output = {
			"video_no_overlap": video,
			"rendered_warp_images": warp_video,
			"predicted_depth": predicted_depth,
			"predicted_mask": predicted_mask,
			"predicted_depth_indices": predicted_depth_indices,
			"video_save_path": str(chunk_path),
			"session_video_path": str(full_video_path),
		}
		last_w2c = self.pipeline._predicted_pose_last_w2c
		if isinstance(last_w2c, torch.Tensor):
			w2c = last_w2c.detach().cpu().float()
			if w2c.dim() == 3:
				w2c = w2c[0]
			output["updated_last_camera_c2w"] = torch.linalg.inv(w2c).numpy()
		for source, key in (
			("_predicted_pose_updated_seed_depth", "updated_seed_depth"),
			("_predicted_pose_updated_seed_mask", "updated_seed_mask"),
		):
			value = getattr(self.pipeline, source, None)
			if isinstance(value, torch.Tensor):
				value = value.detach().cpu()
				while value.dim() > 2:
					value = value[0]
				output[key] = value.numpy() if "mask" not in key else (value > 0.5).numpy()
				setattr(self.pipeline, source, None)
		return output

	def _save_chunk(self, video_tchw: torch.Tensor, *, fps: float, request_id: str) -> Path:
		assert self._session_dir is not None
		path = self._session_dir / f"chunk_{len(self._chunk_paths):04d}_{_safe_name(request_id)}.mp4"
		frames = video_tchw.permute(0, 2, 3, 1).clamp(-1, 1)
		frames = ((frames + 1.0) * 127.5).byte().numpy()
		ffmpeg = _find_ffmpeg()
		cmd = [
			ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
			"-s", f"{self.target_w}x{self.target_h}", "-r", str(float(fps)), "-i", "pipe:0",
			"-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
			"-preset", "fast", "-crf", "18", str(path),
		]
		proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
		_, stderr = proc.communicate(input=np.ascontiguousarray(frames).tobytes())
		if proc.returncode != 0:
			raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {stderr.decode(errors='replace')}")
		self._chunk_paths.append(path)
		return path

	def _rebuild_full_video(self) -> Path:
		assert self._session_dir is not None and self._chunk_paths
		output = self._session_dir / "full_video.mp4"
		if len(self._chunk_paths) == 1:
			shutil.copy2(self._chunk_paths[0], output)
			return output
		concat_file = self._session_dir / "chunks.txt"
		with concat_file.open("w", encoding="utf-8") as handle:
			for path in self._chunk_paths:
				handle.write(f"file '{path.name}'\n")
		cmd = [
			_find_ffmpeg(), "-loglevel", "error", "-y", "-f", "concat",
			"-safe", "0", "-i", str(concat_file), "-c", "copy", "-movflags", "+faststart", str(output),
		]
		subprocess.run(cmd, check=True, cwd=self._session_dir)
		return output

	def get_cache_input_depths(self):
		return None if self.seed_depth_hw is None else self.seed_depth_hw[None]

	def get_cache_input_masks(self):
		return None if self.seed_mask_hw is None else self.seed_mask_hw[None]

	def revert_last_generation(self) -> dict[str, Any]:
		if self.pipeline is None or not self.pipeline.revert_to_snapshot():
			return {"success": False, "message": "No generation is available to revert."}
		removed_paths = self._chunk_paths[self._pre_generate_chunk_count:]
		self._chunk_paths = self._chunk_paths[:self._pre_generate_chunk_count]
		for path in removed_paths:
			path.unlink(missing_ok=True)
		if self._chunk_paths:
			full_video = self._rebuild_full_video()
			message = f"Reverted. Full video: {full_video}"
		else:
			if self._session_dir is not None:
				(self._session_dir / "full_video.mp4").unlink(missing_ok=True)
				(self._session_dir / "chunks.txt").unlink(missing_ok=True)
			message = "Reverted to the seeded image."
		return {"success": True, "message": message}

	def clear_cache(self) -> None:
		self.pipeline = None
		self._chunk_paths.clear()
		self._session_dir = None
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def cleanup(self) -> None:
		self.clear_cache()
