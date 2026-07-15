# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI entrypoint for the Lyra-2 interactive GUI."""

from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import os
import traceback

from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import Response

from api_serialization import API_MEDIA_TYPE, APIMessageError, dumps_api_message, loads_api_message
from api_types import CompressedSeedingRequest, InferenceRequest, RevertResult, SeedingRequest
from server_base import InferenceModel
from server_lyra import DummyLyraModel, LyraModel


def _env_bool(name: str, default: bool = False) -> bool:
	value = os.environ.get(name)
	if value is None:
		return default
	return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ServerSettings:
	checkpoint_path: str | None = os.environ.get("LYRA_GUI_CHECKPOINT_PATH")
	inference_latency: int = int(os.environ.get("LYRA_GUI_INFERENCE_LATENCY_MS", 0))
	inference_cache_size: int = int(os.environ.get("LYRA_GUI_INFERENCE_CACHE_SIZE", 15))
	dummy: bool = _env_bool("LYRA_GUI_DUMMY", False)


settings = ServerSettings()
model: InferenceModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
	global model
	cls = DummyLyraModel if settings.dummy else LyraModel
	model = cls(
		checkpoint_path=settings.checkpoint_path,
		fake_delay_ms=settings.inference_latency,
		inference_cache_size=settings.inference_cache_size,
	)
	yield
	model.cleanup()
	model = None


app = FastAPI(title="Lyra-2 GUI inference server", lifespan=lifespan)
logger = logging.getLogger("uvicorn.error")


def _get_model() -> InferenceModel:
	if model is None:
		raise RuntimeError("The inference model is not initialized.")
	return model


def get_bool_query_param(request: Request, name: str, default: bool) -> bool:
	value = request.query_params.get(name, "1" if default else "0")
	return value.lower() in ("1", "true", "yes", "")


async def read_api_message(request: Request, allowed_types: tuple[type, ...]):
	content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
	if content_type not in (API_MEDIA_TYPE, "application/json"):
		raise APIMessageError(f"Unsupported Content-Type: {content_type or '<missing>'}")
	return loads_api_message(await request.body(), allowed_types=allowed_types)


@app.post("/request-inference", response_class=Response, response_model=None)
async def request_inference(request: Request):
	sync = get_bool_query_param(request, "sync", default=False)
	try:
		req = await read_api_message(request, allowed_types=(InferenceRequest,))
	except APIMessageError as error:
		logger.warning("Invalid inference request: %s", error)
		return Response(str(error), status_code=400)
	try:
		if sync:
			result = await _get_model().request_inference_sync(req)
			return Response(dumps_api_message(result), media_type=API_MEDIA_TYPE)
		_get_model().request_inference(req)
		return Response("Request accepted.", status_code=202)
	except Exception as error:
		logging.error("Inference request failed:\n%s\n%s", error, traceback.format_exc())
		return Response(str(error), status_code=400)


@app.post("/seed-model", response_class=Response, response_model=None)
async def seed_model(request: Request):
	try:
		req = await read_api_message(
			request, allowed_types=(CompressedSeedingRequest, SeedingRequest)
		)
	except APIMessageError as error:
		logger.warning("Invalid seeding request: %s", error)
		return Response(str(error), status_code=400)
	try:
		if isinstance(req, CompressedSeedingRequest):
			req.decompress()
		result = await _get_model().seed_model(req)
		return Response(dumps_api_message(result), media_type=API_MEDIA_TYPE)
	except Exception as error:
		logging.error("Seeding request failed:\n%s\n%s", error, traceback.format_exc())
		return Response(str(error), status_code=400)


@app.post("/revert-last-generation", response_class=Response, response_model=None)
async def revert_last_generation():
	try:
		active = _get_model()
		if not hasattr(active, "revert_last_generation"):
			return Response("Model does not support revert.", status_code=501)
		result = await active.revert_last_generation()
		return Response(dumps_api_message(RevertResult(**result)), media_type=API_MEDIA_TYPE)
	except Exception as error:
		logging.error("Revert failed:\n%s\n%s", error, traceback.format_exc())
		return Response(str(error), status_code=500)


@app.get("/inference-result", response_class=Response, response_model=None)
async def inference_result_or_none(request_id: str):
	try:
		result = _get_model().inference_result_or_none(request_id)
		if result is None:
			return Response("Result not ready", status_code=503)
		return Response(dumps_api_message(result), media_type=API_MEDIA_TYPE)
	except Exception as error:
		logging.error("Result request failed:\n%s\n%s", error, traceback.format_exc())
		return Response(str(error), status_code=500)


@app.get("/metadata")
def metadata():
	return _get_model().metadata()


@app.get("/healthz")
def healthz():
	active = _get_model()
	return {"status": "ok", "model": active.metadata()["model_name"]}
