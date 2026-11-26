"""The version is kept in cutmesh/__init__.py only; rust/Cargo.toml carries the same."""
import contextlib
import io
import os
import re
import sys
import unittest

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import cutmesh                      # noqa: E402
from cutmesh.__main__ import main  # noqa: E402


class Version(unittest.TestCase):
    def test_format(self):
        self.assertRegex(cutmesh.__version__, r"^\d+\.\d+\.\d+$")

    def test_cargo(self):
        with open(os.path.join(ROOT, "rust", "Cargo.toml"), encoding="utf-8") as f:
            m = re.search(r'^version = "([^"]+)"', f.read(), re.M)
        self.assertEqual(m.group(1), cutmesh.__version__)

    def test_pyproject_reads_it(self):
        with open(os.path.join(ROOT, "pyproject.toml"), encoding="utf-8") as f:
            s = f.read()
        self.assertIn('dynamic = ["version"]', s)
        self.assertIn('version = {attr = "cutmesh.__version__"}', s)

    def test_cli(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(main(["--version"]), 0)
        self.assertEqual(out.getvalue().strip(), f"cutmesh {cutmesh.__version__}")


if __name__ == "__main__":
    unittest.main()
