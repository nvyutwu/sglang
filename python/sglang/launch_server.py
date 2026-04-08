"""Launch the inference server."""

import asyncio
import os
import sys

from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import suppress_noisy_warnings

suppress_noisy_warnings()


def _patch_flashinfer_symlink_race():
    """Patch FlashInfer cubin_loader.py to handle symlink race condition.

    When TP > 1, multiple worker processes race to create the same symlink
    in FlashInfer's JIT cache (cubin_loader.py:ensure_symlink). The original
    code calls link.symlink_to(target) without handling FileExistsError,
    causing crashes on multi-GPU startup.

    This monkey-patches the function before workers are forked so all ranks
    inherit the fix.
    """
    try:
        from flashinfer.jit import cubin_loader

        _original_ensure_symlink = cubin_loader.ensure_symlink

        def _safe_ensure_symlink(link, target):
            import pathlib

            link = pathlib.Path(link)
            target = pathlib.Path(target)
            try:
                link.symlink_to(target)
            except FileExistsError:
                pass

        cubin_loader.ensure_symlink = _safe_ensure_symlink
    except (ImportError, AttributeError):
        pass


_patch_flashinfer_symlink_race()


def run_server(server_args):
    """Run the server based on server_args.grpc_mode and server_args.encoder_only."""
    if server_args.grpc_mode:
        from sglang.srt.entrypoints.grpc_server import serve_grpc

        asyncio.run(serve_grpc(server_args))
    elif server_args.encoder_only:
        from sglang.srt.disaggregation.encode_server import launch_server

        launch_server(server_args)
    else:
        # Default mode: HTTP mode.
        from sglang.srt.entrypoints.http_server import launch_server

        launch_server(server_args)


if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
