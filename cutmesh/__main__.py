"""cutmesh meshDict.json [-case DIR] [-noCheck]
(or python -m cutmesh ...; `pip install -e .` once makes both work from any folder)"""
import sys
from .mesher import run


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0
    case = None
    if "-case" in argv:
        i = argv.index("-case"); case = argv[i + 1]; del argv[i:i + 2]
    check = "-noCheck" not in argv
    argv = [a for a in argv if a != "-noCheck"]
    run(argv[0], case, check=check)
    return 0


if __name__ == "__main__":
    sys.exit(main())
