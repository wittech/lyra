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

import asyncio
from io import BytesIO
import time
from typing import Callable
import os

import httpx
from tqdm import tqdm


DEFAULT_STREAM_CHUNK_SIZE = int(os.environ.get("LYRA_GUI_HTTP_STREAM_CHUNK_SIZE", 256 * 1024))  # 256KB
DEFAULT_PROGRESS_INTERVAL_S = float(os.environ.get("LYRA_GUI_HTTP_PROGRESS_INTERVAL_S", 0.10))  # 100ms


def content_with_progress(
	content,
	chunk_size: int = DEFAULT_STREAM_CHUNK_SIZE,
	desc: str = "Upload",
	progress_callback: Callable[[str, float, tqdm], None] | None = None,
	progress_interval_s: float = DEFAULT_PROGRESS_INTERVAL_S,
):
	total = len(content)
	with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit="B", desc=desc) as progress:
		last_cb_t = 0.0
		for i in range(0, total, chunk_size):
			chunk = content[i:i + chunk_size]
			yield chunk
			last_cb_t = report_progress(
				"upload",
				len(chunk),
				progress,
				callback=progress_callback,
				last_callback_t=last_cb_t,
				progress_interval_s=progress_interval_s,
			)

async def async_content_with_progress(*args, **kwargs):
	for chunk in content_with_progress(*args, **kwargs):
		yield chunk


def streaming_response_to_response(response: httpx.Response, content_bytes: BytesIO) -> httpx.Response:
	"""
	Convert a streaming response to a non-streaming response.
	"""
	# TODO: is there a nicer way to get a non-streaming-style Response object, despite
	# having used the streaming API above? (for uniform consumption by the caller).
	to_remove = set(["is_stream_consumed", "next_request", "is_closed", "content", "stream"] + [
		k for k in response.__dict__ if k.startswith("_")
	])
	kwargs = { k: v for k, v in response.__dict__.items() if k not in to_remove }

	content_bytes.seek(0)
	kwargs["content"] = content_bytes.read()
	return httpx.Response(**kwargs)


def report_progress(
	direction: str,
	progress_absolute: int | float,
	bar: tqdm,
	callback: Callable[[str, float, tqdm], None] | None = None,
	*,
	last_callback_t: float = 0.0,
	progress_interval_s: float = DEFAULT_PROGRESS_INTERVAL_S,
) -> float:
	bar.update(progress_absolute)
	if callback is None:
		return last_callback_t

	# Throttle callback frequency to avoid per-chunk Python/UI overhead becoming a throughput bottleneck.
	now = time.monotonic()
	is_done = (bar.total is not None) and (bar.n >= bar.total)
	if is_done or (now - last_callback_t) >= float(progress_interval_s):
		progress_percent = (bar.n / bar.total) if bar.total else 0.0
		callback(direction=direction, progress=progress_percent, bar=bar)
		return now
	return last_callback_t



def httpx_request(method: str,
				  *args,
				  progress: bool = False,
				  progress_direction: str = "auto",
				  desc: str | None = None,
				  async_client: httpx.AsyncClient | None = None,
				  callback: Callable | None = None,
				  progress_callback: Callable[[str, float, tqdm], None] | None = None,
				  stream_chunk_size: int = DEFAULT_STREAM_CHUNK_SIZE,
				  progress_interval_s: float = DEFAULT_PROGRESS_INTERVAL_S,
				  **kwargs) -> httpx.Response | asyncio.Task[httpx.Response]:
	is_async = async_client is not None

	progress_download = progress and (
		progress_direction in ("both", "download")
		or (progress_direction == "auto" and method.lower() == "get")
	)
	progress_upload = progress and (
		progress_direction in ("both", "upload")
		or (progress_direction == "auto" and method.lower() == "post")
	)

	if progress_upload:
		for key in ("content", "data"):
			if key in kwargs:
				upload_desc = f"{desc} (upload)" if desc else "Upload"
				wrapper = async_content_with_progress if is_async else content_with_progress
				kwargs[key] = wrapper(
					kwargs[key],
					chunk_size=int(stream_chunk_size),
					desc=upload_desc,
					progress_callback=progress_callback,
					progress_interval_s=float(progress_interval_s),
				)

	if progress_download:
		# Progress bar requested for download, need to use streaming API

		if async_client is None:
			content_bytes = BytesIO()
			with httpx.stream(method, *args, **kwargs) as response:
				total = int(response.headers["Content-Length"])
				with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit="B", desc=desc) as progress:
					last_cb_t = 0.0
					for chunk in response.iter_bytes(chunk_size=int(stream_chunk_size)):
						# Avoid response.num_bytes_downloaded deltas (extra property reads);
						# len(chunk) is cheaper and accurate for progress.
						last_cb_t = report_progress(
							"download",
							len(chunk),
							progress,
							callback=progress_callback,
							last_callback_t=last_cb_t,
							progress_interval_s=float(progress_interval_s),
						)
						content_bytes.write(chunk)
			response = streaming_response_to_response(response, content_bytes)
			if callback is not None:
				callback(response)
			return response

		else:
			async def inner():
				content_bytes = BytesIO()
				async with async_client.stream(method, *args, **kwargs) as response:
					total = int(response.headers["Content-Length"])
					with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit="B", desc=desc) as progress:
						last_cb_t = 0.0
						async for chunk in response.aiter_bytes(chunk_size=int(stream_chunk_size)):
							last_cb_t = report_progress(
								"download",
								len(chunk),
								progress,
								callback=progress_callback,
								last_callback_t=last_cb_t,
								progress_interval_s=float(progress_interval_s),
							)
							content_bytes.write(chunk)
				response = streaming_response_to_response(response, content_bytes)
				return response

			task = asyncio.create_task(inner())
			if callback is not None:
				task.add_done_callback(callback)
			return task

	else:
		# No download progress bar needed, use standard httpx methods
		if is_async:
			task = asyncio.create_task(
				async_client.request(method, *args, **kwargs)
			)
			if callback is not None:
				task.add_done_callback(callback)
			return task
		else:
			res = httpx.request(method, *args, **kwargs)
			if callback is not None:
				callback(res)
			return res
