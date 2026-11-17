"""cutmesh meshDict.json [-case DIR] [-noCheck]
(or python -m cutmesh ...; `pip install -e .` once makes both work from any folder)"""
import sys
from .mesher import run
from .config import ConfigError
from .geometry import STLError


def _usage_error(msg):
    print(f"cutmesh: {msg}\nusage: {__doc__.splitlines()[0]}", file=sys.stderr)
    return 2


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    case, check, files = None, True, []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-case":
            if i + 1 >= len(argv) or argv[i + 1].startswith("-"):
                return _usage_error("-case needs a directory")
            case = argv[i + 1]; i += 2; continue
        if a == "-noCheck":
            check = False
        elif a.startswith("-"):
            return _usage_error(f"unknown option {a!r}")
        else:
            files.append(a)
        i += 1
    if len(files) != 1:
        return _usage_error("give one meshDict.json" if not files else
                            f"one meshDict.json expected, got {len(files)}: {' '.join(files)}")
    try:
        run(files[0], case, check=check)
    except (ConfigError, STLError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
