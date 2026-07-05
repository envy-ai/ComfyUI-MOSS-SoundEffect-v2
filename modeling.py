from __future__ import annotations

import gc
import ctypes
import json
import logging
import os
import site
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

import folder_paths

DEFAULT_REPO_ID = "OpenMOSS-Team/MOSS-SoundEffect-v2.0"
DEFAULT_MODEL_DIR_NAME = "MOSS-SoundEffect-v2.0"
MODEL_FOLDER_NAME = "moss_soundeffect_v2"
WEIGHT_QUANTIZATION_OPTIONS = ["auto", "off", "int8_convrot"]

_PIPELINE_CACHE: OrderedDict[tuple[str, str, str, str, bool, str], "MossSoundEffectV2Model"] = OrderedDict()
_MAX_CACHED_PIPELINES = 1
_PRELOADED_CUDA_LIBRARY_HANDLES: dict[Path, Any] = {}


@dataclass
class MossSoundEffectV2Model:
    pipeline: Any
    model_path: str
    device: str
    dtype: str
    sample_rate: int
    max_inference_seconds: int
    weight_quantization: str = "off"


@dataclass
class DitLoadResult:
    missing_keys: list[str]
    unexpected_keys: list[str]
    quantized_keys: list[str]


def register_model_folder() -> None:
    root = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    root.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(root), is_default=True)


register_model_folder()


def get_model_root() -> Path:
    paths = folder_paths.get_folder_paths(MODEL_FOLDER_NAME)
    if not paths:
        register_model_folder()
        paths = folder_paths.get_folder_paths(MODEL_FOLDER_NAME)
    root = Path(paths[0])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _has_model_index(path: Path) -> bool:
    return path.is_dir() and (path / "model_index.json").is_file()


def get_local_model_dirs() -> list[str]:
    roots = [Path(path) for path in folder_paths.get_folder_paths(MODEL_FOLDER_NAME)]
    names: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if _has_model_index(child):
                names.append(child.name)
    return list(dict.fromkeys(names))


def get_model_options() -> list[str]:
    return [DEFAULT_REPO_ID, *get_local_model_dirs()]


def _looks_like_hf_repo_id(value: str) -> bool:
    if os.path.isdir(os.path.expanduser(value)):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts) and not value.startswith((".", "~", os.sep))


def _repo_dir_name(repo_id: str) -> str:
    if repo_id == DEFAULT_REPO_ID:
        return DEFAULT_MODEL_DIR_NAME
    return repo_id.rstrip("/").split("/")[-1]


def _resolve_local_choice(model: str) -> Path | None:
    for root in folder_paths.get_folder_paths(MODEL_FOLDER_NAME):
        candidate = Path(root) / model
        if _has_model_index(candidate):
            return candidate
    return None


def resolve_model_path(
    model: str,
    manual_model_path: str,
    auto_download: bool,
    revision: str,
    local_files_only: bool,
    snapshot_download: Callable[..., str] | None = None,
) -> Path | str:
    selected = manual_model_path.strip() or model
    expanded = Path(selected).expanduser()
    if expanded.is_dir():
        if not _has_model_index(expanded):
            raise FileNotFoundError(f"MOSS SoundEffect v2 model directory is missing model_index.json: {expanded}")
        return expanded.resolve()

    local_choice = _resolve_local_choice(selected)
    if local_choice is not None:
        return local_choice.resolve()

    if not _looks_like_hf_repo_id(selected):
        raise FileNotFoundError(
            f"Could not find MOSS SoundEffect v2 model '{selected}'. "
            f"Use a local directory or a Hugging Face repo id like {DEFAULT_REPO_ID}."
        )

    target_dir = get_model_root() / _repo_dir_name(selected)
    if _has_model_index(target_dir):
        return target_dir.resolve()

    if local_files_only:
        raise FileNotFoundError(f"Local model files not found at {target_dir}")

    if not auto_download:
        return selected

    if snapshot_download is None:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        snapshot_download = hf_snapshot_download

    logging.info("Downloading %s to %s", selected, target_dir)
    downloaded = snapshot_download(
        repo_id=selected,
        local_dir=str(target_dir),
        revision=revision.strip() or None,
        local_files_only=False,
    )
    downloaded_path = Path(downloaded)
    return downloaded_path.resolve() if downloaded_path.exists() else target_dir.resolve()


def resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import comfy.model_management

            return str(comfy.model_management.get_torch_device())
        except Exception:
            return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA was requested for MOSS SoundEffect v2 but is unavailable; using CPU.")
        return "cpu"
    return device


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    return "float32"


def is_int8_convrot_available() -> bool:
    try:
        from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor
    except Exception:
        return False
    return "int8_tensorwise" in QUANT_ALGOS and hasattr(QuantizedTensor, "from_float")


def resolve_weight_quantization(weight_quantization: str, device: str) -> str:
    if weight_quantization not in WEIGHT_QUANTIZATION_OPTIONS:
        raise ValueError(
            f"Unsupported weight_quantization={weight_quantization!r}. "
            f"Expected one of: {', '.join(WEIGHT_QUANTIZATION_OPTIONS)}."
        )
    if weight_quantization == "off":
        return "off"
    if weight_quantization == "auto":
        if str(device).startswith("cuda") and is_int8_convrot_available():
            return "int8_convrot"
        return "off"
    if not is_int8_convrot_available():
        raise RuntimeError(
            "int8_convrot quantization requires ComfyUI's comfy_kitchen-backed int8 quantization support."
        )
    return "int8_convrot"


def _cuda_major_version() -> str | None:
    cuda_version = getattr(torch.version, "cuda", None)
    if not cuda_version:
        return None
    return str(cuda_version).split(".", 1)[0]


def _cuda_runtime_library_dirs() -> list[Path]:
    major = _cuda_major_version()
    if major is None:
        return []
    version_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.prefix) / "lib" / version_dir / "site-packages" / "nvidia" / f"cu{major}" / "lib",
    ]
    for package_root in site.getsitepackages():
        candidates.append(Path(package_root) / "nvidia" / f"cu{major}" / "lib")
    return list(dict.fromkeys(path for path in candidates if path.is_dir()))


def _cuda_runtime_library_names() -> list[str]:
    major = _cuda_major_version()
    if major is None:
        return []
    return [
        f"libnvJitLink.so.{major}",
        f"libnvrtc-builtins.so.{major}.0",
        f"libnvrtc.so.{major}",
    ]


def prepare_cuda_runtime_libraries(cdll_loader=ctypes.CDLL) -> list[Path]:
    loaded: list[Path] = []
    library_dirs = _cuda_runtime_library_dirs()
    if not library_dirs:
        return loaded

    existing_ld_paths = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    prepended = [str(path) for path in library_dirs if str(path) not in existing_ld_paths]
    if prepended:
        os.environ["LD_LIBRARY_PATH"] = ":".join([*prepended, *existing_ld_paths])

    for library_dir in library_dirs:
        for library_name in _cuda_runtime_library_names():
            library_path = library_dir / library_name
            if not library_path.is_file() or library_path in _PRELOADED_CUDA_LIBRARY_HANDLES:
                continue
            try:
                handle = cdll_loader(str(library_path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                logging.warning("Failed to preload CUDA runtime library %s", library_path, exc_info=True)
                continue
            _PRELOADED_CUDA_LIBRARY_HANDLES[library_path] = handle
            loaded.append(library_path)
    return loaded


def import_pipeline_class():
    prepare_cuda_runtime_libraries()
    try:
        from moss_soundeffect_v2 import MossSoundEffectPipeline
    except Exception as exc:
        raise RuntimeError(
            "MOSS-SoundEffect v2 is not installed in this ComfyUI Python environment. "
            "Install this node pack's requirements, then restart ComfyUI: "
            "pip install -r custom_nodes/moss_soundeffect_v2/requirements.txt"
        ) from exc
    return MossSoundEffectPipeline


def ensure_torchdynamo_disabled(disable: bool) -> None:
    if disable:
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


def unwrap_compiled_model_fn(pipeline: Any) -> bool:
    engine = getattr(pipeline, "engine", None)
    model_fn = getattr(engine, "model_fn", None)
    original_fn = getattr(model_fn, "__wrapped__", None)
    if engine is None or original_fn is None:
        return False
    engine.model_fn = original_fn
    return True


def quantize_weight_int8_convrot(
    weight: torch.Tensor,
    compute_dtype: torch.dtype,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    from comfy.quant_ops import QuantizedTensor

    return QuantizedTensor.from_float(
        weight.detach().to(device="cpu", dtype=compute_dtype),
        "TensorWiseINT8Layout",
        per_channel=True,
        convrot=True,
        convrot_groupsize=convrot_groupsize,
    )


def _get_child_module(module: torch.nn.Module, name: str) -> torch.nn.Module:
    if isinstance(module, (torch.nn.ModuleList, torch.nn.Sequential)) and name.isdigit():
        return module[int(name)]
    return getattr(module, name)


def _get_parent_module(root: torch.nn.Module, key: str) -> tuple[torch.nn.Module, str]:
    parts = key.split(".")
    module = root
    for part in parts[:-1]:
        module = _get_child_module(module, part)
    return module, parts[-1]


def _is_linear_weight_key(root: torch.nn.Module, key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    try:
        module, name = _get_parent_module(root, key)
    except AttributeError:
        return False
    return name == "weight" and isinstance(module, torch.nn.Linear)


def _assign_module_tensor(root: torch.nn.Module, key: str, tensor: torch.Tensor) -> None:
    module, name = _get_parent_module(root, key)
    if name in module._parameters:
        module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)
        return
    if name in module._buffers:
        module._buffers[name] = tensor
        return
    raise KeyError(f"Could not assign tensor for unknown DiT parameter or buffer: {key}")


def _hf_dit_key_to_custom_key(key: str) -> str:
    from moss_soundeffect_v2.diffsynth.pipelines import wan_audio

    converted = wan_audio._convert_hf_dit_state_dict({key: None})
    return next(iter(converted.keys()))


def stream_load_dit_state_dict(
    dit: torch.nn.Module,
    weights_path: Path,
    key_converter: Callable[[str], str],
    torch_dtype: torch.dtype,
    safe_open_fn: Callable[..., Any] | None = None,
    quantize_weight_fn: Callable[[torch.Tensor, torch.dtype, int], torch.Tensor] = quantize_weight_int8_convrot,
    convrot_groupsize: int = 256,
) -> DitLoadResult:
    if safe_open_fn is None:
        from safetensors import safe_open

        safe_open_fn = safe_open

    expected_keys = set(dit.state_dict().keys())
    loaded_keys: set[str] = set()
    unexpected_keys: list[str] = []
    quantized_keys: list[str] = []

    with safe_open_fn(str(weights_path), framework="pt", device="cpu") as reader:
        for hf_key in reader.keys():
            custom_key = key_converter(hf_key)
            if custom_key not in expected_keys:
                unexpected_keys.append(custom_key)
                continue

            tensor = reader.get_tensor(hf_key)
            if _is_linear_weight_key(dit, custom_key):
                loaded_tensor = quantize_weight_fn(tensor, torch_dtype, convrot_groupsize)
                quantized_keys.append(custom_key)
            else:
                loaded_tensor = tensor.detach()
                if loaded_tensor.is_floating_point():
                    loaded_tensor = loaded_tensor.to(device="cpu", dtype=torch_dtype)
                else:
                    loaded_tensor = loaded_tensor.to(device="cpu")

            _assign_module_tensor(dit, custom_key, loaded_tensor)
            loaded_keys.add(custom_key)
            del tensor, loaded_tensor

    missing_keys = sorted(expected_keys - loaded_keys)
    return DitLoadResult(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        quantized_keys=quantized_keys,
    )


def _materialize_wan_audio_freq_buffers(dit: torch.nn.Module, dit_cfg: dict[str, Any]) -> None:
    from moss_soundeffect_v2.diffsynth.models import wan_audio_dit

    head_dim = int(dit_cfg["dim"]) // int(dit_cfg["num_heads"])
    vae_type = dit_cfg.get("vae_type", "dac")
    if vae_type == "oobleck":
        freqs = wan_audio_dit.legacy_precompute_freqs_cis_1d(
            head_dim,
            base_tps=4.0,
            target_tps=44100 / 2048,
        )
    elif vae_type == "dac":
        freqs = wan_audio_dit.precompute_freqs_cis_1d(head_dim)
    else:
        raise ValueError(f"Invalid VAE type: {vae_type}")

    dit._buffers["freqs_cis_0"] = freqs[0]
    dit._buffers["freqs_cis_1"] = freqs[1]
    dit._buffers["freqs_cis_2"] = freqs[2]


def load_quantized_pipeline(
    model_path: Path,
    torch_dtype: torch.dtype,
    device: str,
    quantization: str,
) -> Any:
    if quantization != "int8_convrot":
        raise ValueError(f"Unsupported quantized MOSS loader mode: {quantization}")

    prepare_cuda_runtime_libraries()
    from moss_soundeffect_v2.diffsynth.pipelines import wan_audio
    from moss_soundeffect_v2.pipeline_moss_soundeffect import MossSoundEffectPipeline

    model_dir = Path(model_path)
    with open(model_dir / "model_index.json", encoding="utf-8") as f:
        index = json.load(f)
    with open(model_dir / "scheduler" / "scheduler_config.json", encoding="utf-8") as f:
        sched_cfg = json.load(f)
    with open(model_dir / "transformer" / "config.json", encoding="utf-8") as f:
        dit_cfg = json.load(f)

    print(f"Loading from: {model_dir}")
    print(f"  Pipeline: {index['_class_name']}, dit_variant: {index.get('dit_variant')}")

    te_path = model_dir / "text_encoder"
    print(f"  Loading text_encoder from {te_path} ...")
    text_encoder = wan_audio.Qwen3TextEncoder(str(te_path), torch_dtype=torch_dtype)
    text_encoder = text_encoder.to(device)
    print(f"  text_encoder: dim={text_encoder.dim}")

    tok_path = model_dir / "tokenizer"
    print(f"  Loading tokenizer from {tok_path} ...")
    prompter = wan_audio.WanPrompter(tokenizer_path=str(tok_path))
    prompter.fetch_models(text_encoder)

    vae_dir = model_dir / "vae"
    vae_pth = vae_dir / "vae_128d_48k.pth"
    vae_safetensors = vae_dir / "diffusion_pytorch_model.safetensors"
    if vae_pth.exists():
        print(f"  Loading DAC VAE from {vae_pth} ...")
        vae = wan_audio.DAC.load(str(vae_pth))
    elif vae_safetensors.exists():
        print(f"  Loading DAC VAE from {vae_safetensors} ...")
        vae = wan_audio.DAC.load(str(vae_safetensors))
    else:
        raise FileNotFoundError(f"No VAE found in {vae_dir}")

    print("  Building DiT on meta device for streaming int8_convrot load ...")
    with torch.device("meta"):
        dit = wan_audio.WanAudioModel(
            in_dim=dit_cfg["in_dim"],
            out_dim=dit_cfg["out_dim"],
            text_dim=dit_cfg["text_dim"],
            freq_dim=dit_cfg["freq_dim"],
            eps=dit_cfg["eps"],
            patch_size=tuple(dit_cfg["patch_size"]),
            has_image_input=dit_cfg["has_image_input"],
            dim=dit_cfg["dim"],
            ffn_dim=dit_cfg["ffn_dim"],
            num_heads=dit_cfg["num_heads"],
            num_layers=dit_cfg["num_layers"],
            vae_type=dit_cfg.get("vae_type", "dac"),
        )
    _materialize_wan_audio_freq_buffers(dit, dit_cfg)

    dit_weights_path = model_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    print(f"  Streaming DiT int8_convrot weights from {dit_weights_path} ...")
    load_result = stream_load_dit_state_dict(
        dit=dit,
        weights_path=dit_weights_path,
        key_converter=_hf_dit_key_to_custom_key,
        torch_dtype=torch_dtype,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Failed to stream-load quantized MOSS DiT: "
            f"missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}"
        )
    print(f"  DiT loaded with {len(load_result.quantized_keys)} int8_convrot linear weights")

    pipe = wan_audio.WanAudioPipeline(
        device=device,
        torch_dtype=torch_dtype,
        flow_shift=sched_cfg.get("shift", 5.0),
    )
    pipe.text_encoder = text_encoder
    pipe.prompter = prompter
    pipe.vae = vae
    pipe.dit = dit
    pipe.audio_latent_dim = dit_cfg["in_dim"]
    pipe.num_samples_division_factor = vae.hop_length
    pipe.dit_variant = index.get("dit_variant")
    pipe.to(device)
    print(f"  Pipeline assembled on {device}")

    return MossSoundEffectPipeline(
        engine=pipe,
        sample_rate=int(index.get("sample_rate", 48000)),
        max_inference_seconds=int(index.get("max_inference_seconds", 30)),
    )


def _evict_extra_cached_pipelines() -> None:
    while len(_PIPELINE_CACHE) > _MAX_CACHED_PIPELINES:
        _, model = _PIPELINE_CACHE.popitem(last=False)
        try:
            model.pipeline.to("cpu")
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            import comfy.model_management

            comfy.model_management.soft_empty_cache()
        except Exception:
            torch.cuda.empty_cache()


def clear_pipeline_cache() -> None:
    for model in _PIPELINE_CACHE.values():
        try:
            model.pipeline.to("cpu")
        except Exception:
            pass
    _PIPELINE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        try:
            import comfy.model_management

            comfy.model_management.soft_empty_cache()
        except Exception:
            torch.cuda.empty_cache()


def load_pipeline(
    model: str,
    manual_model_path: str,
    device: str,
    dtype: str,
    auto_download: bool,
    local_files_only: bool,
    revision: str,
    unload_comfy_models: bool,
    disable_torch_compile: bool,
    weight_quantization: str,
) -> MossSoundEffectV2Model:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
    resolved_weight_quantization = resolve_weight_quantization(weight_quantization, resolved_device)
    ensure_torchdynamo_disabled(disable_torch_compile)
    if resolved_device.startswith("cuda"):
        prepare_cuda_runtime_libraries()
    resolved_path = resolve_model_path(
        model=model,
        manual_model_path=manual_model_path,
        auto_download=auto_download,
        revision=revision,
        local_files_only=local_files_only,
    )
    cache_key = (
        str(resolved_path),
        resolved_device,
        dtype_name(resolved_dtype),
        revision.strip(),
        disable_torch_compile,
        resolved_weight_quantization,
    )
    cached = _PIPELINE_CACHE.get(cache_key)
    if cached is not None:
        _PIPELINE_CACHE.move_to_end(cache_key)
        return cached

    if unload_comfy_models:
        try:
            import comfy.model_management

            comfy.model_management.unload_all_models()
        except Exception:
            logging.exception("Failed to unload existing ComfyUI models before loading MOSS SoundEffect v2.")

    if resolved_weight_quantization == "int8_convrot":
        pipeline = load_quantized_pipeline(
            model_path=Path(resolved_path),
            torch_dtype=resolved_dtype,
            device=resolved_device,
            quantization=resolved_weight_quantization,
        )
    else:
        Pipeline = import_pipeline_class()
        pipeline = Pipeline.from_pretrained(
            str(resolved_path),
            torch_dtype=resolved_dtype,
            device=resolved_device,
        )
    if disable_torch_compile:
        unwrap_compiled_model_fn(pipeline)
    wrapper = MossSoundEffectV2Model(
        pipeline=pipeline,
        model_path=str(resolved_path),
        device=resolved_device,
        dtype=dtype_name(resolved_dtype),
        weight_quantization=resolved_weight_quantization,
        sample_rate=int(getattr(pipeline, "sample_rate", 48000)),
        max_inference_seconds=int(getattr(pipeline, "max_inference_seconds", 30)),
    )
    _PIPELINE_CACHE[cache_key] = wrapper
    _evict_extra_cached_pipelines()
    return wrapper


def to_comfy_audio(waveform: torch.Tensor, sample_rate: int) -> dict[str, Any]:
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().cpu().to(torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        raise ValueError(f"Expected waveform with 1, 2, or 3 dimensions; got shape {tuple(waveform.shape)}")
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


class ComfyProgress:
    def __call__(self, iterable):
        try:
            total = len(iterable)
        except TypeError:
            total = None
        pbar = None
        if total is not None:
            try:
                import comfy.utils

                pbar = comfy.utils.ProgressBar(total)
            except Exception:
                pbar = None
        for index, item in enumerate(iterable):
            yield item
            if pbar is not None:
                pbar.update_absolute(index + 1, total)
