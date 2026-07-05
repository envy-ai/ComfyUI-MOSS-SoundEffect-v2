from __future__ import annotations

import asyncio
import os
from pathlib import Path

import torch


def test_extension_registers_v3_nodes():
    from custom_nodes.moss_soundeffect_v2 import comfy_entrypoint
    from custom_nodes.moss_soundeffect_v2.nodes import (
        MossSoundEffectV2Generate,
        MossSoundEffectV2Loader,
    )

    extension = asyncio.run(comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())

    assert nodes == [MossSoundEffectV2Loader, MossSoundEffectV2Generate]
    assert MossSoundEffectV2Loader.define_schema().node_id == "MOSS_SoundEffectV2Loader"
    assert MossSoundEffectV2Generate.define_schema().outputs[0].io_type == "AUDIO"


def test_model_folder_is_registered():
    import folder_paths
    from custom_nodes.moss_soundeffect_v2.modeling import MODEL_FOLDER_NAME

    assert MODEL_FOLDER_NAME in folder_paths.folder_names_and_paths
    registered_paths = folder_paths.get_folder_paths(MODEL_FOLDER_NAME)
    assert registered_paths
    assert Path(registered_paths[0]).name == MODEL_FOLDER_NAME


def test_resolve_model_path_downloads_default_repo_to_comfy_models(tmp_path, monkeypatch):
    from custom_nodes.moss_soundeffect_v2 import modeling

    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        (local_dir / "model_index.json").write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(modeling, "get_model_root", lambda: tmp_path)

    resolved = modeling.resolve_model_path(
        model=modeling.DEFAULT_REPO_ID,
        manual_model_path="",
        auto_download=True,
        revision="",
        local_files_only=False,
        snapshot_download=fake_snapshot_download,
    )

    assert resolved == tmp_path / modeling.DEFAULT_MODEL_DIR_NAME
    assert calls == [
        {
            "repo_id": modeling.DEFAULT_REPO_ID,
            "local_dir": str(tmp_path / modeling.DEFAULT_MODEL_DIR_NAME),
            "revision": None,
            "local_files_only": False,
        }
    ]


def test_loader_uses_lazy_pipeline_import_and_returns_model_wrapper(tmp_path, monkeypatch):
    from custom_nodes.moss_soundeffect_v2 import modeling
    from custom_nodes.moss_soundeffect_v2.nodes import MossSoundEffectV2Loader

    class FakePipeline:
        sample_rate = 48000
        max_inference_seconds = 30

        @classmethod
        def from_pretrained(cls, path, torch_dtype, device):
            cls.load_args = (path, torch_dtype, device)
            return cls()

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    monkeypatch.setattr(modeling, "import_pipeline_class", lambda: FakePipeline)
    monkeypatch.setattr(
        modeling,
        "resolve_model_path",
        lambda **kwargs: model_dir,
    )
    modeling.clear_pipeline_cache()

    out = MossSoundEffectV2Loader.execute(
        model=modeling.DEFAULT_REPO_ID,
        manual_model_path="",
        device="cpu",
        dtype="float32",
        auto_download=True,
        local_files_only=False,
        revision="",
        unload_comfy_models=False,
        disable_torch_compile=True,
    )
    wrapper = out[0]

    assert wrapper.pipeline.__class__ is FakePipeline
    assert wrapper.sample_rate == 48000
    assert wrapper.max_inference_seconds == 30
    assert FakePipeline.load_args == (str(model_dir), torch.float32, "cpu")


def test_disable_torch_compile_sets_env_and_unwraps_engine_model_fn(monkeypatch):
    from custom_nodes.moss_soundeffect_v2 import modeling

    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)

    def compiled():
        return "compiled"

    def original():
        return "original"

    compiled.__wrapped__ = original

    class FakeEngine:
        model_fn = compiled

    class FakePipeline:
        engine = FakeEngine()

    modeling.ensure_torchdynamo_disabled(True)
    was_unwrapped = modeling.unwrap_compiled_model_fn(FakePipeline())

    assert os.environ["TORCHDYNAMO_DISABLE"] == "1"
    assert was_unwrapped is True
    assert FakePipeline.engine.model_fn is original


def test_prepare_cuda_runtime_libraries_prepends_path_and_preloads(monkeypatch, tmp_path):
    from custom_nodes.moss_soundeffect_v2 import modeling

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    for name in [
        "libnvJitLink.so.13",
        "libnvrtc-builtins.so.13.0",
        "libnvrtc.so.13",
    ]:
        (lib_dir / name).write_text("", encoding="utf-8")

    calls = []

    def fake_cdll(path, mode):
        calls.append((Path(path).name, mode))

    monkeypatch.setattr(modeling.torch.version, "cuda", "13.0")
    monkeypatch.setattr(modeling, "_cuda_runtime_library_dirs", lambda: [lib_dir])
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")

    loaded = modeling.prepare_cuda_runtime_libraries(cdll_loader=fake_cdll)

    assert loaded == [lib_dir / name for name, _ in calls]
    assert os.environ["LD_LIBRARY_PATH"].startswith(f"{lib_dir}:/existing")
    assert [name for name, _ in calls] == [
        "libnvJitLink.so.13",
        "libnvrtc-builtins.so.13.0",
        "libnvrtc.so.13",
    ]


def test_to_comfy_audio_normalizes_waveform_shapes():
    from custom_nodes.moss_soundeffect_v2.modeling import to_comfy_audio

    one_dim = to_comfy_audio(torch.zeros(12), 48000)
    two_dim = to_comfy_audio(torch.zeros(2, 12), 48000)
    three_dim = to_comfy_audio(torch.zeros(3, 2, 12), 48000)

    assert one_dim["waveform"].shape == (1, 1, 12)
    assert two_dim["waveform"].shape == (1, 2, 12)
    assert three_dim["waveform"].shape == (3, 2, 12)
    assert one_dim["waveform"].dtype is torch.float32
    assert one_dim["sample_rate"] == 48000


def test_generate_outputs_native_audio_without_preview():
    from custom_nodes.moss_soundeffect_v2.modeling import MossSoundEffectV2Model
    from custom_nodes.moss_soundeffect_v2.nodes import MossSoundEffectV2Generate

    class FakePipeline:
        sample_rate = 48000
        max_inference_seconds = 30

        def __call__(self, **kwargs):
            self.call_kwargs = kwargs
            return torch.zeros(1, 1, 24)

    pipeline = FakePipeline()
    model = MossSoundEffectV2Model(
        pipeline=pipeline,
        model_path="/tmp/model",
        device="cpu",
        dtype="float32",
        sample_rate=48000,
        max_inference_seconds=30,
    )

    out = MossSoundEffectV2Generate.execute(
        moss_model=model,
        prompt="keyboard clicks",
        negative_prompt="",
        seconds=1.0,
        num_inference_steps=4,
        cfg_scale=4.0,
        sigma_shift=5.0,
        seed=123,
        append_duration_suffix=True,
        preview=False,
    )
    audio = out[0]

    assert audio["waveform"].shape == (1, 1, 24)
    assert audio["sample_rate"] == 48000
    assert pipeline.call_kwargs["prompt"] == "keyboard clicks"
    assert pipeline.call_kwargs["seconds"] == 1.0
    assert pipeline.call_kwargs["num_inference_steps"] == 4
    assert pipeline.call_kwargs["seed"] == 123
