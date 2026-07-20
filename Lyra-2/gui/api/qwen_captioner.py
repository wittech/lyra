# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local, public VLM captioning for the Lyra GUI server."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_QWEN_MODEL = str(
	Path(__file__).resolve().parents[2] / "checkpoints" / "qwen" / "Qwen3-VL-4B-Instruct"
)

CAPTION_SYSTEM_PROMPT = """You are a video captioning specialist writing one English prompt for a
camera-controlled video generator. You receive three labeled groups: the original seed frame defines
the environment identity; the latest generated frame shows the current scene; and trajectory
projections show upcoming camera views, where black areas have no observed content. Describe visible
objects and spatial relationships faithfully, infer coherent content for newly revealed areas using
the seed's architecture, materials, lighting, palette, and atmosphere, and describe the camera
movement suggested by the projections. Write 80-100 natural, concrete words. Never mention images,
projections, occlusion, missing or black areas, instructions, or uncertainty. Return only the prompt."""


def _as_pil(image: np.ndarray | torch.Tensor) -> Image.Image:
	if isinstance(image, torch.Tensor):
		image = image.detach().float().cpu().numpy()
	if image.ndim == 3 and image.shape[0] in (1, 3, 4):
		image = np.transpose(image, (1, 2, 0))
	if image.shape[-1] == 4:
		image = image[..., :3]
	if np.issubdtype(image.dtype, np.floating):
		if float(np.nanmin(image)) < -0.1:
			image = image * 0.5 + 0.5
		image = np.clip(image, 0.0, 1.0) * 255.0
	return Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")


class QwenCaptioner:
	"""Qwen3-VL captioner kept on CPU except while generating a caption."""

	def __init__(
		self,
		model_name: str = DEFAULT_QWEN_MODEL,
		device: str = "cuda",
		max_new_tokens: int = 160,
	) -> None:
		self.model_name = model_name
		self.device = torch.device(device)
		self.max_new_tokens = max_new_tokens
		self.processor = AutoProcessor.from_pretrained(model_name)

		attn_implementation = os.environ.get("LYRA_GUI_QWEN_ATTN", "sdpa")
		self.model = AutoModelForImageTextToText.from_pretrained(
			model_name,
			dtype=torch.bfloat16,
			attn_implementation=attn_implementation,
			low_cpu_mem_usage=True,
		)
		self.model.eval().requires_grad_(False)
		self.model.to("cpu")

	@contextmanager
	def on_device(self):
		self.model.to(self.device)
		try:
			yield
		finally:
			self.model.to("cpu")
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	@torch.inference_mode()
	def caption(self, images: Iterable[np.ndarray | torch.Tensor], region_hint: str = "") -> str:
		pil_images = [_as_pil(image) for image in images]
		if len(pil_images) < 2:
			raise ValueError("Dynamic Qwen captioning requires a seed and latest frame.")

		instruction = CAPTION_SYSTEM_PROMPT
		if region_hint.strip():
			instruction += (
				"\nThe user specifically wants newly revealed areas to contain: "
				f'"{region_hint.strip()}". Integrate that request while preserving scene coherence.'
			)

		content = [
			{"type": "text", "text": instruction},
			{
				"type": "text",
				"text": "[SCENE IDENTITY] Original seed frame; use it for style and environment identity:",
			},
			{"type": "image", "image": pil_images[0]},
			{
				"type": "text",
				"text": "[CURRENT SCENE] Latest generated frame; describe its actual visible content:",
			},
			{"type": "image", "image": pil_images[1]},
		]
		if len(pil_images) > 2:
			content.append({
				"type": "text",
				"text": "[TRAJECTORY PROJECTIONS] Upcoming camera views in chronological order:",
			})
			content.extend({"type": "image", "image": image} for image in pil_images[2:])
		messages = [{"role": "user", "content": content}]
		inputs = self.processor.apply_chat_template(
			messages,
			tokenize=True,
			add_generation_prompt=True,
			return_dict=True,
			return_tensors="pt",
		)
		inputs = inputs.to(self.device)
		generated = self.model.generate(
			**inputs,
			max_new_tokens=self.max_new_tokens,
			do_sample=False,
			use_cache=True,
		)
		trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
		caption = self.processor.batch_decode(
			trimmed,
			skip_special_tokens=True,
			clean_up_tokenization_spaces=False,
		)[0].strip()
		if not caption:
			raise RuntimeError("Qwen returned an empty caption.")
		return caption
