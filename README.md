# ComfyUI MOSS-SoundEffect v2 Nodes

Native V3 ComfyUI nodes for OpenMOSS MOSS-SoundEffect v2.0.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/envy-ai/ComfyUI-MOSS-SoundEffect-v2.git moss_soundeffect_v2
```

Install the upstream MOSS package into the same Python environment used to run ComfyUI:

```bash
cd /path/to/ComfyUI
pip install -r custom_nodes/moss_soundeffect_v2/requirements.txt
```

Restart ComfyUI after installing dependencies.

## Nodes

- **Load MOSS SoundEffect v2**: loads the pipeline and downloads `OpenMOSS-Team/MOSS-SoundEffect-v2.0` to `models/moss_soundeffect_v2/MOSS-SoundEffect-v2.0` if it is not already present.
- **MOSS SoundEffect v2 Generate**: generates native ComfyUI `AUDIO` from a prompt. Connect it to ComfyUI's built-in Preview Audio or Save Audio nodes.

The loader defaults `disable_torch_compile` to enabled. This follows the upstream inference script fallback and avoids TorchDynamo graph-break errors when SageAttention calls non-traceable CUDA helpers during `torch.compile`.

On CUDA 13 torch builds, the node also prepends and preloads the matching `nvidia/cu13/lib` runtime libraries before importing or running MOSS. This keeps NVRTC VAE decode kernels from failing when `libnvrtc-builtins.so.13.0` is installed but not visible to the ComfyUI process.

## Dependencies

The upstream package requires Python 3.12+ and has strict dependency pins. It uses the existing ComfyUI torch install because this requirements file does not install the upstream torch CUDA extra.

## Model Location

The loader prefers local models in:

```text
models/moss_soundeffect_v2/
```

When `auto_download` is enabled, the default Hugging Face model is downloaded into:

```text
models/moss_soundeffect_v2/MOSS-SoundEffect-v2.0/
```
