from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from . import modeling

MossSoundEffectV2ModelType = io.Custom("MOSS_SOUNDEFFECT_V2_MODEL")


class MossSoundEffectV2Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MOSS_SoundEffectV2Loader",
            display_name="Load MOSS SoundEffect v2",
            category="audio/MOSS SoundEffect v2",
            description="Loads MOSS-SoundEffect v2.0 and downloads the Hugging Face model into ComfyUI/models/moss_soundeffect_v2 when needed.",
            inputs=[
                io.Combo.Input("model", options=modeling.get_model_options(), default=modeling.DEFAULT_REPO_ID),
                io.String.Input(
                    "manual_model_path",
                    default="",
                    advanced=True,
                    tooltip="Optional local model directory or Hugging Face repo id. Overrides the model menu when set.",
                ),
                io.Combo.Input("device", options=["auto", "cuda", "cpu"], default="auto"),
                io.Combo.Input("dtype", options=["auto", "bfloat16", "float16", "float32"], default="auto"),
                io.Boolean.Input("auto_download", default=True),
                io.Boolean.Input("local_files_only", default=False, advanced=True),
                io.String.Input("revision", default="", advanced=True),
                io.Boolean.Input(
                    "unload_comfy_models",
                    default=True,
                    advanced=True,
                    tooltip="Unload currently managed ComfyUI models before loading MOSS to free VRAM.",
                ),
                io.Boolean.Input(
                    "disable_torch_compile",
                    default=True,
                    advanced=True,
                    tooltip="Disables the upstream torch.compile path. Keep this enabled if SageAttention or TorchDynamo graph-break errors occur.",
                ),
            ],
            outputs=[MossSoundEffectV2ModelType.Output("moss_model")],
            search_aliases=["moss sound effect", "moss soundeffect", "text to audio"],
        )

    @classmethod
    def execute(
        cls,
        model: str,
        manual_model_path: str,
        device: str,
        dtype: str,
        auto_download: bool,
        local_files_only: bool,
        revision: str,
        unload_comfy_models: bool,
        disable_torch_compile: bool,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            modeling.load_pipeline(
                model=model,
                manual_model_path=manual_model_path,
                device=device,
                dtype=dtype,
                auto_download=auto_download,
                local_files_only=local_files_only,
                revision=revision,
                unload_comfy_models=unload_comfy_models,
                disable_torch_compile=disable_torch_compile,
            )
        )


class MossSoundEffectV2Generate(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MOSS_SoundEffectV2Generate",
            display_name="MOSS SoundEffect v2 Generate",
            category="audio/MOSS SoundEffect v2",
            description="Generates native ComfyUI AUDIO from a text prompt using MOSS-SoundEffect v2.0.",
            inputs=[
                MossSoundEffectV2ModelType.Input("moss_model"),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    default="The crisp, rhythmic click-clack of fast typing on a mechanical keyboard.",
                ),
                io.String.Input("negative_prompt", multiline=True, default="", advanced=True),
                io.Float.Input("seconds", default=10.0, min=0.1, max=30.0, step=0.1, round=0.1),
                io.Int.Input("num_inference_steps", default=100, min=1, max=300, step=1),
                io.Float.Input("cfg_scale", default=4.0, min=0.0, max=20.0, step=0.1, round=0.01),
                io.Float.Input("sigma_shift", default=5.0, min=0.1, max=20.0, step=0.1, round=0.01, advanced=True),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xffffffffffffffff,
                    step=1,
                    control_after_generate=True,
                ),
                io.Boolean.Input("append_duration_suffix", default=True, advanced=True),
                io.Boolean.Input("preview", default=True, advanced=True),
            ],
            outputs=[io.Audio.Output("audio")],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            search_aliases=["moss sound effect", "sound effect generator", "text to sound"],
        )

    @classmethod
    def execute(
        cls,
        moss_model: modeling.MossSoundEffectV2Model,
        prompt: str,
        negative_prompt: str,
        seconds: float,
        num_inference_steps: int,
        cfg_scale: float,
        sigma_shift: float,
        seed: int,
        append_duration_suffix: bool,
        preview: bool,
    ) -> io.NodeOutput:
        if moss_model is None or getattr(moss_model, "pipeline", None) is None:
            raise ValueError("MOSS SoundEffect v2 model is not loaded.")
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")
        if seconds <= 0:
            raise ValueError("seconds must be greater than 0.")
        if seconds > moss_model.max_inference_seconds:
            raise ValueError(
                f"seconds={seconds} exceeds this model's max_inference_seconds={moss_model.max_inference_seconds}."
            )

        if str(moss_model.device).startswith("cuda"):
            modeling.prepare_cuda_runtime_libraries()
        progress = modeling.ComfyProgress()
        with torch.inference_mode():
            waveform = moss_model.pipeline(
                prompt=prompt,
                seconds=seconds,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale,
                sigma_shift=sigma_shift,
                seed=seed,
                negative_prompt=negative_prompt,
                append_duration_suffix=append_duration_suffix,
                progress_bar_cmd=progress,
            )
        audio = modeling.to_comfy_audio(waveform, moss_model.sample_rate)
        if preview:
            return io.NodeOutput(audio, ui=ui.PreviewAudio(audio, cls=cls))
        return io.NodeOutput(audio)
