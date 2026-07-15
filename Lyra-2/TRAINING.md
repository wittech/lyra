# Training Lyra 2.0

## 1. Download pretrained weights

First follow the main [checkpoint download instructions](README.md#download-checkpoints).

Training additionally initializes the diffusion transformer from the public
[Wan2.1 I2V 14B 480P checkpoint](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P). Download only
its diffusion shards and convert them to the bfloat16 `.pth` format consumed by the training entry
point:

```bash
conda activate lyra2
pip install -U huggingface_hub

WAN_ROOT=checkpoints/pretrained/Wan2.1-I2V-14B-480P
hf download Wan-AI/Wan2.1-I2V-14B-480P \
  --include "diffusion_pytorch_model-*.safetensors" \
  --local-dir "$WAN_ROOT"

python -m scripts.convert_wan_checkpoint \
  --input-dir "$WAN_ROOT" \
  --output "$WAN_ROOT/Wan2.1-I2V-14B-480P.pth"
```

The converter requires roughly 45 GB of CPU memory and writes an approximately 33 GB file.

## 2. Prepare the dataset

Lyra 2 training requires long videos that capture static scenes under camera motion.
The public checkpoint was trained on
[DL3DV-10K](https://dl3dv-10k.github.io/DL3DV-10K/). We sample every other frame from each source
video (temporal stride 2) and retain only videos with more than 1,000 frames after subsampling.

Process each retained video with [VIPE](https://github.com/nv-tlabs/vipe) to estimate camera poses
and metric depth. Divide the video into 80-frame chunks, caption each chunk with a VLM, and encode
each caption with the T5 text encoder. RGB frames, poses, intrinsics, depths, captions, and T5
embeddings must remain frame-aligned.

To illustrate the dataset layout and chunk-level captions expected by Lyra 2 training, we provide
two examples. Download them from the repository root:

```bash
hf download nvidia/Lyra-2.0 \
  --include "assets/training_data/**" \
  --local-dir .
```

```text
assets/training_data/
├── all_filter.lst
├── rgb/<video-id>.mp4
├── pose/<video-id>.npz
├── intrinsics/<video-id>.npz
├── depth/<video-id>.zip
└── framepack-caption-qwen3-chunk-80-overlap-0/<video-id>.pkl
```

Each sample uses the same `<video-id>` basename across all directories:

- `all_filter.lst` lists the video IDs included in the dataset, one per line.
- `rgb/<video-id>.mp4` contains the temporally subsampled RGB video.
- `pose/<video-id>.npz` contains aligned frame indices and camera-to-world poses.
- `intrinsics/<video-id>.npz` contains aligned frame indices and camera intrinsics.
- `depth/<video-id>.zip` contains one metric-depth EXR image per frame.
- `framepack-caption-qwen3-chunk-80-overlap-0/<video-id>.pkl` contains a dictionary of chunk-level
  records. Each record stores its `frame_range`, human-readable `caption`, and bfloat16 T5
  `embedding`.

To inspect the expected chunk-level prompts:

```python
import pickle
from pathlib import Path

root = Path("assets/training_data/framepack-caption-qwen3-chunk-80-overlap-0")
for path in sorted(root.glob("*.pkl")):
    chunks = pickle.loads(path.read_bytes())
    print(path.stem)
    for chunk in chunks.values():
        print(chunk["frame_range"], chunk["caption"])
```

Make sure every video in your training dataset follows this layout and that all modalities use the
same frame indexing.

To train on your own preprocessed dataset, add a named entry to
`lyra_2/_src/datasets/config_dataverse.py` by copying `lyra2_sample_data` and replacing its
`root_path`, `filter_list_path`, `data_name`, and `t5_embedding_path`. Every entry in
`DATAVERSE_CONFIG` is registered automatically. Then change the two dataset overrides in
`lyra_2/_src/configs/experiment.py`, for example:

```python
{"override /data_train": "my_preprocessed_dataset"},
{"override /data_val": "my_preprocessed_dataset"},
```

## 3. Train

Run this two-iteration test on one 8-GPU node to verify the sample dataset and visualization path:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LYRA2_DISABLE_CHECKPOINT_SAVE=1

torchrun --nproc_per_node=8 --master_port=12341 \
  -m scripts.train \
  --config=lyra_2/_src/configs/config.py -- \
  experiment=lyra2 \
  job.name=sample_training \
  trainer.max_iter=2 \
  trainer.logging_iter=1 \
  trainer.callbacks.frame_pack_viz_sampling.every_n=1 \
  trainer.callbacks.frame_pack_viz_sampling.num_sampling_step=1
```

For an actual run, first register the full preprocessed dataset as described in Step 2 and select it
in the experiment config. Then enable checkpoints and use the original experiment cadence:

```bash
unset LYRA2_DISABLE_CHECKPOINT_SAVE

torchrun --nproc_per_node=8 --master_port=12341 \
  -m scripts.train \
  --config=lyra_2/_src/configs/config.py -- \
  experiment=lyra2 \
  job.name=lyra2_training \
  trainer.max_iter=1000000 \
  checkpoint.save_iter=100 \
  trainer.logging_iter=50 \
  trainer.callbacks.frame_pack_viz_sampling.every_n=400 \
  trainer.callbacks.frame_pack_viz_sampling.num_sampling_step=35
```

Logs, TensorBoard events, visualizations, and checkpoints are written under
`outputs/lyra_2/lyra2/<job.name>/` in `stdout.log`, `tensorboard/`, `visualizations/`, and
`checkpoints/`, respectively. Training converges after approximately 5,000 iterations on 64 B200
GPUs.
