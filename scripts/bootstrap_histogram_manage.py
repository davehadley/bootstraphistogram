#!/usr/bin/env python3
from argparse import ArgumentParser, Namespace
from pathlib import Path
from subprocess import CalledProcessError, run

_PACKAGE_ROOT_DIRECTORY = Path(__file__).parent.parent


def _parseargs() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "build-docker",
            "run-docker",
            "build",
            "test",
            "build-documentation",
            "coverage",
        ],
    )
    args = parser.parse_args()
    return args


def _main() -> None:
    args = _parseargs()
    if args.command == "build-docker":
        _build_docker()
    if args.command == "run-docker":
        _run_docker()
    if args.command == "build":
        _build()
    if args.command == "test":
        _test()
    if args.command == "build-documentation":
        _build_documentation()
    if args.command == "coverage":
        _coverage()
    return


def _build_docker() -> None:
    run(
        ["docker", "build", "-tbootstraphistogram:latest", "."],
        cwd=_PACKAGE_ROOT_DIRECTORY,
        check=True,
    )


def _run_docker() -> None:
    try:
        run(["docker", "start", "bootstraphistogram"], check=True)
    except CalledProcessError:
        run(
            [
                "docker",
                "run",
                "--name",
                "bootstraphistogram",
                "-it",
                "-d",
                "bootstraphistogram:latest",
                "/bin/bash",
            ],
            check=True,
        )


def _build() -> None:
    run(["poetry", "install"], check=True, cwd=_PACKAGE_ROOT_DIRECTORY)


def _test() -> None:
    run(["poetry", "run", "pytest", "tests"], check=True, cwd=_PACKAGE_ROOT_DIRECTORY)


def _build_documentation() -> None:
    run(
        ["poetry", "run", "pip", "install", "-r", "docs/requirements.txt"],
        check=True,
        cwd=_PACKAGE_ROOT_DIRECTORY,
    )
    run(
        ["poetry", "run", "sphinx-build", "-W", "docs", "docs-build"],
        check=True,
        cwd=_PACKAGE_ROOT_DIRECTORY,
    )


def _coverage() -> None:
    run(
        ["poetry", "run", "coverage", "run", "-m", "pytest", "tests"],
        check=True,
        cwd=_PACKAGE_ROOT_DIRECTORY,
    )
    run(
        ["poetry", "run", "coverage", "report", "-m"],
        check=True,
        cwd=_PACKAGE_ROOT_DIRECTORY,
    )


if __name__ == "__main__":
    _main()
