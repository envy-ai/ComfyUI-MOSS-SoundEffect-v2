from __future__ import annotations

import gc
import ctypes
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

_PIPELINE_CACHE: OrderedDict[tuple[str, str, str, str, bool], "MossSoundEffectV2Model"] = OrderedDict()
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
) -> MossSoundEffectV2Model:
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_dtype(dtype, resolved_device)
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
    cache_key = (str(resolved_path), resolved_device, dtype_name(resolved_dtype), revision.strip(), disable_torch_compile)
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
