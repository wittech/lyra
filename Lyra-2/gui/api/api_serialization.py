# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allowlisted JSON serialization for messages crossing the GUI HTTP API."""

import base64
import binascii
from dataclasses import fields, is_dataclass
from enum import Enum
import json
import math
from typing import Any

import numpy as np

from api_types import (
	CompressedInferenceResult,
	CompressedSeedingRequest,
	InferenceRequest,
	InferenceResult,
	RevertResult,
	SeedingRequest,
	SeedingResult,
)
from encoding import CompressionFormat


API_MEDIA_TYPE = "application/vnd.lyra2.api+json"

_TYPE_KEY = "__lyra2_type__"
_VALUE_KEY = "value"

_ARRAY_TYPE = "ndarray"
_BYTES_TYPE = "bytes"
_ENUM_TYPE = "enum"
_MESSAGE_TYPES = (
	CompressedInferenceResult,
	CompressedSeedingRequest,
	InferenceRequest,
	InferenceResult,
	RevertResult,
	SeedingRequest,
	SeedingResult,
)
_MESSAGE_TYPES_BY_NAME = {cls.__name__: cls for cls in _MESSAGE_TYPES}
_ENUM_TYPES_BY_NAME = {CompressionFormat.__name__: CompressionFormat}


class APIMessageError(ValueError):
	"""Raised when an API message cannot be safely decoded."""


def dumps_api_message(message: Any) -> bytes:
	"""Serialize an allowlisted Lyra 2 API dataclass to JSON bytes."""
	if not is_dataclass(message) or isinstance(message, type):
		raise TypeError(f"Expected an API dataclass instance, got {type(message).__name__}")

	message_type = type(message)
	if message_type.__name__ not in _MESSAGE_TYPES_BY_NAME:
		raise TypeError(f"Unsupported API message type: {message_type.__name__}")

	payload = {
		"type": message_type.__name__,
		"fields": {
			field.name: _encode_value(getattr(message, field.name))
			for field in fields(message)
			if field.init
		},
	}
	return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def loads_api_message(data: bytes, allowed_types: tuple[type, ...] | None = None) -> Any:
	"""Deserialize an allowlisted Lyra 2 API dataclass from JSON bytes."""
	try:
		payload = json.loads(data.decode("utf-8"))
	except (UnicodeDecodeError, json.JSONDecodeError) as error:
		raise APIMessageError(f"Invalid API JSON payload: {error}") from error

	if not isinstance(payload, dict):
		raise APIMessageError("API payload must be a JSON object")

	message_type_name = payload.get("type")
	field_values = payload.get("fields")
	if not isinstance(message_type_name, str) or not isinstance(field_values, dict):
		raise APIMessageError("API payload must include string 'type' and object 'fields'")

	candidate_types = allowed_types or _MESSAGE_TYPES
	candidate_types_by_name = {cls.__name__: cls for cls in candidate_types}
	message_type = candidate_types_by_name.get(message_type_name)
	if message_type is None:
		allowed = ", ".join(sorted(candidate_types_by_name))
		raise APIMessageError(
			f"Unsupported API message type '{message_type_name}', expected one of: {allowed}"
		)

	allowed_field_names = {field.name for field in fields(message_type) if field.init}
	unknown_fields = set(field_values) - allowed_field_names
	if unknown_fields:
		unknown = ", ".join(sorted(unknown_fields))
		raise APIMessageError(f"Unexpected field(s) for {message_type_name}: {unknown}")

	kwargs = {
		field.name: _decode_value(field_values[field.name])
		for field in fields(message_type)
		if field.init and field.name in field_values
	}
	try:
		return message_type(**kwargs)
	except (AssertionError, TypeError, ValueError) as error:
		raise APIMessageError(f"Invalid {message_type_name} payload: {error}") from error


def _encode_value(value: Any) -> Any:
	if isinstance(value, np.ndarray):
		array = np.ascontiguousarray(value)
		if array.dtype.hasobject:
			raise TypeError("Object dtype arrays are not supported")
		return {
			_TYPE_KEY: _ARRAY_TYPE,
			"dtype": array.dtype.str,
			"shape": list(array.shape),
			_VALUE_KEY: base64.b64encode(array.tobytes()).decode("ascii"),
		}

	if isinstance(value, bytes):
		return {_TYPE_KEY: _BYTES_TYPE, _VALUE_KEY: base64.b64encode(value).decode("ascii")}

	if isinstance(value, Enum):
		return {
			_TYPE_KEY: _ENUM_TYPE,
			"enum": type(value).__name__,
			_VALUE_KEY: value.value,
		}

	if isinstance(value, (list, tuple)):
		return [_encode_value(item) for item in value]

	if value is None or isinstance(value, (str, int, float, bool)):
		return value

	raise TypeError(f"Unsupported API value type: {type(value).__name__}")


def _decode_value(value: Any) -> Any:
	if isinstance(value, dict):
		type_name = value.get(_TYPE_KEY)
		if type_name == _ARRAY_TYPE:
			return _decode_array(value)
		if type_name == _BYTES_TYPE:
			return _decode_base64(value.get(_VALUE_KEY), "bytes")
		if type_name == _ENUM_TYPE:
			return _decode_enum(value)
		raise APIMessageError("Unexpected JSON object in API payload")

	if isinstance(value, list):
		return [_decode_value(item) for item in value]

	if value is None or isinstance(value, (str, int, float, bool)):
		return value

	raise APIMessageError(f"Unsupported JSON value in API payload: {type(value).__name__}")


def _decode_array(value: dict[str, Any]) -> np.ndarray:
	dtype_name = value.get("dtype")
	shape = value.get("shape")
	encoded = value.get(_VALUE_KEY)
	if not isinstance(dtype_name, str) or not isinstance(shape, list):
		raise APIMessageError("Array payload must include string 'dtype' and list 'shape'")

	try:
		dtype = np.dtype(dtype_name)
	except (TypeError, ValueError) as error:
		raise APIMessageError(f"Invalid numpy dtype '{dtype_name}'") from error
	if dtype.hasobject:
		raise APIMessageError("Object dtype arrays are not supported")
	if not all(isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0 for dim in shape):
		raise APIMessageError("Array shape must contain non-negative integers")

	raw = _decode_base64(encoded, "array")
	expected_size = math.prod(shape) * dtype.itemsize
	if len(raw) != expected_size:
		raise APIMessageError(f"Array byte length mismatch: expected {expected_size}, got {len(raw)}")
	return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()


def _decode_base64(encoded: Any, name: str) -> bytes:
	if not isinstance(encoded, str):
		raise APIMessageError(f"{name} payload must be a base64 string")
	try:
		return base64.b64decode(encoded.encode("ascii"), validate=True)
	except (UnicodeEncodeError, binascii.Error) as error:
		raise APIMessageError(f"Invalid base64 data in {name} payload") from error


def _decode_enum(value: dict[str, Any]) -> Enum:
	enum_type_name = value.get("enum")
	enum_value = value.get(_VALUE_KEY)
	if not isinstance(enum_type_name, str):
		raise APIMessageError("Enum payload must include string 'enum'")

	enum_type = _ENUM_TYPES_BY_NAME.get(enum_type_name)
	if enum_type is None:
		raise APIMessageError(f"Unsupported enum type: {enum_type_name}")
	try:
		return enum_type(enum_value)
	except ValueError as error:
		raise APIMessageError(f"Invalid {enum_type_name} value: {enum_value}") from error
