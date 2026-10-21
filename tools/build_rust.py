"""Build the Rust extension cutmesh._core (rust/) and copy it into the package.

python tools/build_rust.py [--debug]

Needs a Rust toolchain (cargo).  Without the extension cutmesh falls back to the
pure Python/NumPy geometry queries.  Set CARGO_TARGET_DIR to share build artifacts.
"""
import os
import shutil
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main(argv):
    release = "--debug" not in argv
    crate = os.path.join(ROOT, "rust")
    env = dict(os.environ, PYO3_PYTHON=sys.executable)
    cmd = ["cargo", "build", "--manifest-path", os.path.join(crate, "Cargo.toml")]
    if release:
        cmd.append("--release")
    subprocess.run(cmd, check=True, env=env)
    target = env.get("CARGO_TARGET_DIR", os.path.join(crate, "target"))
    out = os.path.join(target, "release" if release else "debug")
    for name in ("_core.dll", "lib_core.so", "lib_core.dylib"):
        src = os.path.join(out, name)
        if os.path.isfile(src):
            break
    else:
        sys.exit(f"built library not found in {out}")
    ext = ".pyd" if os.name == "nt" else sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    dst = os.path.join(ROOT, "cutmesh", "_core" + ext)
    shutil.copyfile(src, dst)
    print(f"installed {dst}")


if __name__ == "__main__":
    main(sys.argv[1:])
