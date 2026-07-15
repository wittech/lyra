# Lyra 2.0 Interactive GUI

<video src="https://github.com/user-attachments/assets/76394f4d-b6c7-46f2-8133-5eabf5fd74e1" autoplay controls loop muted playsinline width="720"></video>

The Lyra 2.0 GUI lets you initialize a scene from one image, author a camera trajectory, and
autoregressively explore the generated 3D world. The viewer runs on your local workstation and
connects to a Lyra 2.0 inference server running on either the same or a remote machine.

## Starting the inference server

Follow the main [installation instructions](../INSTALL.md) and
[download the Lyra 2.0 checkpoints](../README.md#download-checkpoints). Then install the GUI server
requirements in the same `lyra2` conda environment and download Qwen3-VL:

```bash
conda activate lyra2
cd /path/to/Lyra-2

pip install -r gui/requirements.txt
pip install -U huggingface_hub
hf download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir checkpoints/qwen/Qwen3-VL-4B-Instruct
```

Start the server from the repository root:

```bash
export PYTHONPATH="$PWD:$PWD/gui/api"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m uvicorn server:app --app-dir gui/api --host 0.0.0.0 --port 8000
```

The server is ready when Uvicorn reports that it is listening on port 8000.

### Connecting to a remote server

If the inference server is on a remote compute node, forward its port to your workstation:

```bash
ssh -N -L 8000:<compute-node-hostname>:8000 <login-hostname>
```

The client and server exchange allowlisted JSON messages. An SSH tunnel is still recommended when
the compute node is not directly reachable from your workstation.

## Starting the GUI on your local machine

The viewer requires Linux, an NVIDIA GPU, and a CUDA toolkit. Create the `lyra2` environment by
following [INSTALL.md](../INSTALL.md), then build the GUI:

```bash
conda activate lyra2
cd /path/to/Lyra-2

conda install -c conda-forge -y glew libgl-devel xorg-libx11 xorg-libxrandr \
  xorg-libxinerama xorg-libxcursor xorg-libxi pkg-config
bash gui/scripts/setup_dependencies.sh
pip install -r gui/requirements.txt

export CMAKE_PREFIX_PATH="$CONDA_PREFIX${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
cmake gui -B gui/build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DNGP_BUILD_WITH_PYTHON_BINDINGS=ON
cmake --build gui/build --config RelWithDebInfo -j
```

With the server running, launch the client:

```bash
python gui/api/client.py --host 127.0.0.1 --port 8000
```

## Using the GUI

### Seeding the scene

Drag a `.jpg`, `.jpeg`, `.png`, or `.exr` image onto the viewport, or enter its path under
**Video generation server → Seeding** and click **Seed**. The server estimates depth and camera
intrinsics, initializes the Lyra 2.0 spatial cache, and displays the image as a point cloud.

### Authoring a camera trajectory

Use the **Camera path & video generation** window to define where the camera should move:

- **Record camera path** records motion from the viewport.
- **Clear** and **Init from views** reset or initialize the path from the current view.
- **Load** and **Save** read or write camera-path JSON files.
- **Add from cam**, **Split**, **Dup**, **Del**, and **Set** edit keyframes.
- The red, green, and blue gizmo edits translation or rotation in local or world coordinates.
- **Field of view** changes the current keyframe; **Apply to all keyframes** propagates it.
- **Start**, **Rev**, **Play**, and **End** preview the trajectory.

### Autoregressive generation

After seeding, add a camera trajectory and click **Generate video**. The GUI prompts for a short text
description of content that should appear in the outpainted region. It can be as simple as `a dog`:
the VLM rewrites it using the current scene context. If the prompt is left empty, the VLM imagines
coherent content for the newly revealed region.

The Lyra 2.0 server then generates a video following the specified camera trajectory. When the new
chunk is complete, it is displayed in the video player and the 3D cache is updated. If you do not
like the result, click **Revert** to return to the state immediately before the latest chunk was
generated—you can then define a new camera trajectory, try a different text prompt, or simply
randomize the output.

To continue exploring, add another camera trajectory using the same controls and click
**Generate video** again. Each new chunk extends the current world autoregressively.

The complete stitched video is saved server-side as
`outputs/gui_sessions/<timestamp>_<seed-request>/full_video.mp4`; use this file as the input to
[Step 2 — 3D Gaussian Splatting Reconstruction](../README.md#step-2--3d-gaussian-splatting-reconstruction).

## Useful configuration variables

Set these variables before starting the server when you need to override a default:

| Variable | Default | Purpose |
|---|---:|---|
| `LYRA_GUI_OUTPUT_DIR` | `outputs/gui_sessions` | Server-side session videos |
| `LYRA_GUI_OFFLOAD` | `1` | Offload inactive diffusion weights to reduce GPU memory use |
| `LYRA_GUI_OFFLOAD_VAE` | `1` | Keep the VAE on CPU while it is idle |
| `LYRA_GUI_RETRIEVAL_VIEWS` | `1` | Spatial-memory views retrieved per autoregressive step |
| `LYRA_GUI_USE_DMD` | `0` | Enable the optional four-step DMD LoRA |
| `LYRA_GUI_GUIDANCE` | `5.0` | Classifier-free guidance scale without DMD |
| `LYRA_GUI_SEED` | `1` | Generation seed |
