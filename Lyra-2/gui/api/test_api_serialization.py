# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import unittest

import numpy as np

from api_serialization import APIMessageError, dumps_api_message, loads_api_message
from api_types import CompressedInferenceResult, InferenceRequest, RevertResult
from encoding import CompressionFormat


class APIMessageSerializationTest(unittest.TestCase):
	def test_inference_request_round_trip(self):
		message = InferenceRequest(
			request_id="request-1",
			cameras_to_world=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
			focal_lengths=np.full((2, 2), 500.0, dtype=np.float32),
			principal_points=np.full((2, 2), 0.5, dtype=np.float32),
			resolutions=np.full((2, 2), [832, 480], dtype=np.int32),
			timestamps=np.array([0.0, 1.0 / 16.0], dtype=np.float64),
			region_hint="a dog",
		)

		decoded = loads_api_message(dumps_api_message(message), allowed_types=(InferenceRequest,))

		self.assertIsInstance(decoded, InferenceRequest)
		self.assertEqual(decoded.region_hint, "a dog")
		np.testing.assert_array_equal(decoded.cameras_to_world, message.cameras_to_world)
		np.testing.assert_array_equal(decoded.timestamps, message.timestamps)

	def test_binary_media_and_enum_round_trip(self):
		message = CompressedInferenceResult(
			request_id="result-1",
			cameras_to_world=np.zeros((2, 3, 4), dtype=np.float32),
			focal_lengths=np.ones((2, 2), dtype=np.float32),
			principal_points=np.full((2, 2), 0.5, dtype=np.float32),
			resolutions=np.full((2, 2), [832, 480], dtype=np.int32),
			result_ids=["0", "1"],
			timestamps=[0.0, 0.0625],
			images=None,
			depths=None,
			runtime_ms=10.0,
			images_compressed=[b"video-bytes"],
			images_format=CompressionFormat.MP4,
			depths_compressed=None,
			depths_format=None,
		)

		decoded = loads_api_message(
			dumps_api_message(message), allowed_types=(CompressedInferenceResult,)
		)

		self.assertEqual(decoded.images_compressed, [b"video-bytes"])
		self.assertEqual(decoded.images_format, CompressionFormat.MP4)

	def test_rejects_legacy_binary_payload(self):
		with self.assertRaises(APIMessageError):
			loads_api_message(b"\x80\x04legacy-binary-payload")

	def test_endpoint_type_allowlist_and_unknown_fields(self):
		payload = dumps_api_message(RevertResult(success=True, message="done"))
		with self.assertRaises(APIMessageError):
			loads_api_message(payload, allowed_types=(InferenceRequest,))

		decoded = json.loads(payload)
		decoded["fields"]["unexpected"] = True
		with self.assertRaises(APIMessageError):
			loads_api_message(json.dumps(decoded).encode(), allowed_types=(RevertResult,))


if __name__ == "__main__":
	unittest.main()
