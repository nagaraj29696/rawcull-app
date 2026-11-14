import sys


def _main() -> int:
    if "--cli" in sys.argv:
        sys.argv.remove("--cli")
        from .cli import main
        return main()
    from .ui import run
    return run()


if __name__ == "__main__":
    raise SystemExit(_main())
