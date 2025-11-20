import argparse

from sglang.cli.generate import generate
from sglang.cli.serve import serve
from sglang.cli.utils import get_git_commit_hash
from sglang.version import __version__


def version(args, extra_argv):
    print(f"sglang version: {__version__}")
    print(f"git revision: {get_git_commit_hash()[:7]}")


def main():
    parser = argparse.ArgumentParser()

    # complex sub commands
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the SGLang server.",
        add_help=False,  # Defer help to the specific parser
    )
    def _serve_entry(args, extra_argv):
        from sglang.cli.serve import serve as _serve
        return _serve(args, extra_argv)
    serve_parser.set_defaults(func=_serve_entry)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,  # Defer help to the specific parser
    )
    def _generate_entry(args, extra_argv):
        from sglang.cli.generate import generate as _generate
        return _generate(args, extra_argv)
    generate_parser.set_defaults(func=_generate_entry)

    # simple commands
    version_parser = subparsers.add_parser(
        "version",
        help="Show the version information.",
    )
    version_parser.set_defaults(func=version)

    args, extra_argv = parser.parse_known_args()
    args.func(args, extra_argv)
