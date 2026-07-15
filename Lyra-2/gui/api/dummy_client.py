# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small headless client that smoke-tests seed + synchronous Lyra inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import httpx
import numpy as np

from api_serialization import API_MEDIA_TYPE, dumps_api_message, loads_api_message
from api_types import CompressedInferenceResult, InferenceRequest, SeedingRequest, SeedingResult


def _camera_batch(count: int) -> np.ndarray:
	c2w = np.tile(np.eye(4, dtype=np.float32)[None], (count, 1, 1))
	c2w[:, 0, 3] = np.linspace(0.0, 0.25, count, dtype=np.float32)
	return c2w[:, :3]


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("--url", default="http://127.0.0.1:8000")
	parser.add_argument("--image", default=None, help="Optional seed image. A synthetic image is used by default.")
	parser.add_argument("--output", default="outputs/dummy_client.mp4")
	parser.add_argument("--timeout", type=float, default=7200.0)
	args = parser.parse_args()

	with httpx.Client(base_url=args.url, timeout=args.timeout) as client:
		metadata = client.get("/metadata").raise_for_status().json()
		width, height = metadata["inference_resolution"][0]
		frame_count = int(metadata["min_frames_per_request"])

		if args.image:
			bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
			if bgr is None:
				raise FileNotFoundError(args.image)
			rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
			rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
		else:
			x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
			y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
			rgb = np.concatenate([
				np.broadcast_to(x, (height, width, 1)),
				np.broadcast_to(y, (height, width, 1)),
				np.full((height, width, 1), 0.35, dtype=np.float32),
			], axis=-1)
			rgb = (rgb * 255.0 + 0.5).astype(np.uint8)

		focal = float(max(width, height))
		seed = SeedingRequest(
			request_id="dummy-seed",
			images=rgb[None].astype(np.float32) / 255.0,
			depths=None,
			masks=None,
			cameras_to_world=_camera_batch(1),
			focal_lengths=np.array([[focal, focal]], dtype=np.float32),
			principal_points=np.array([[0.5, 0.5]], dtype=np.float32),
			resolutions=np.array([[width, height]], dtype=np.int32),
		)
		seed_response = client.post(
			"/seed-model",
			content=dumps_api_message(seed),
			headers={"Content-Type": API_MEDIA_TYPE, "Accept": API_MEDIA_TYPE},
		)
		seed_response.raise_for_status()
		seed_result = loads_api_message(seed_response.content, allowed_types=(SeedingResult,))
		if seed_result.depths is None:
			raise RuntimeError("Server did not return seed depth.")

		request = InferenceRequest(
			request_id="dummy-inference",
			timestamps=np.arange(frame_count, dtype=np.float64) / 16.0,
			cameras_to_world=_camera_batch(frame_count),
			focal_lengths=np.full((frame_count, 2), focal, dtype=np.float32),
			principal_points=np.full((frame_count, 2), 0.5, dtype=np.float32),
			resolutions=np.tile([[width, height]], (frame_count, 1)),
			framerate=16.0,
			return_depths=False,
			show_cache_renderings=False,
			region_hint="",
		)
		response = client.post(
			"/request-inference?sync=1",
			content=dumps_api_message(request),
			headers={"Content-Type": API_MEDIA_TYPE, "Accept": API_MEDIA_TYPE},
		)
		response.raise_for_status()
		result = loads_api_message(response.content, allowed_types=(CompressedInferenceResult,))
		if not isinstance(result, CompressedInferenceResult):
			raise TypeError(f"Expected CompressedInferenceResult, got {type(result).__name__}")
		if result.images_format.value != "mp4" or len(result.images_compressed) != 1:
			raise RuntimeError("Server returned an invalid compressed video result.")

		output = Path(args.output)
		output.parent.mkdir(parents=True, exist_ok=True)
		output.write_bytes(result.images_compressed[0])
		print(
			f"OK: model={metadata['model_name']}, frames={frame_count}, "
			f"seed_depth={seed_result.depths.shape}, video={output.resolve()}"
		)


if __name__ == "__main__":
	main()
