from __future__ import annotations

"""Run llama-cpp-python with the MTMD vision projector on CPU memory.

The Qwen3.6 language model fits on a 24 GB GPU alongside the embedding service,
but its F32 projector does not. llama-cpp-python's CLI currently constructs the
MTMD handler with GPU offload enabled and exposes no flag for changing it. Patch
that constructor before starting the otherwise standard server CLI.
"""

from llama_cpp import llama_chat_format


_original_mtmd_init = llama_chat_format.MTMDChatHandler.__init__


def _cpu_mtmd_init(
    self: llama_chat_format.MTMDChatHandler,
    clip_model_path: str,
    verbose: bool = True,
    use_gpu: bool = True,
) -> None:
    _original_mtmd_init(
        self,
        clip_model_path=clip_model_path,
        verbose=verbose,
        use_gpu=False,
    )


llama_chat_format.MTMDChatHandler.__init__ = _cpu_mtmd_init  # type: ignore[method-assign]


if __name__ == "__main__":
    from llama_cpp.server.__main__ import main

    main()
