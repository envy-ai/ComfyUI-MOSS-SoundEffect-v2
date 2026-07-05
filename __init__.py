from __future__ import annotations

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import MossSoundEffectV2Generate, MossSoundEffectV2Loader


class MossSoundEffectV2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [MossSoundEffectV2Loader, MossSoundEffectV2Generate]


async def comfy_entrypoint() -> MossSoundEffectV2Extension:
    return MossSoundEffectV2Extension()


__all__ = ["comfy_entrypoint", "MossSoundEffectV2Extension"]
