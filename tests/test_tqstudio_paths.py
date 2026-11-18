"""tqstudio market data folder: the first valid of a list (``tqstudio/config.yaml``), an override, the download
writing to it (``mdbuild --root``)."""

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqstudio import paths  # noqa: E402
from tqstudio.paths import candidate_roots, is_valid_root, resolve_root  # noqa: E402


def make_root(folder: Path) -> Path:
    (folder / "config").mkdir(parents=True)
    (folder / "config" / "settings.yaml").write_text("bloomberg:\n  host: localhost\n", encoding="utf-8")
    return folder


class TestMarketDataRoot(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.saved = os.environ.pop(paths.ENV, None)

    def tearDown(self):
        if self.saved is not None:
            os.environ[paths.ENV] = self.saved
        self.tmp.cleanup()

    def config(self, *roots) -> Path:
        cfg = self.base / "config.yaml"
        cfg.write_text("marketdata_roots:\n" + "".join(f"  - {r}\n" for r in roots), encoding="utf-8")
        return cfg

    def test_first_valid_in_order(self):
        missing = self.base / "share_offline"                              # an unreachable share
        empty = self.base / "empty"                                        # a folder without config/settings.yaml
        empty.mkdir()
        first, second = make_root(self.base / "first"), make_root(self.base / "second")
        cfg = self.config(missing, empty, first, second)
        self.assertEqual(candidate_roots(cfg), [missing, empty, first, second])
        self.assertFalse(is_valid_root(missing) or is_valid_root(empty))
        self.assertEqual(resolve_root(config=cfg), first)

    def test_override(self):
        root, other = make_root(self.base / "root"), make_root(self.base / "other")
        cfg = self.config(root)
        self.assertEqual(resolve_root(str(other), config=cfg), other)       # explicit
        os.environ[paths.ENV] = str(other)
        self.assertEqual(resolve_root(config=cfg), other)                   # from the environment (the app's processes)
        with self.assertRaises(FileNotFoundError):
            resolve_root(str(self.base / "nothing"), config=cfg)            # an override must be valid

    def test_none_valid(self):
        cfg = self.config(self.base / "a", self.base / "b")
        with self.assertRaises(FileNotFoundError) as err:
            resolve_root(config=cfg)
        self.assertIn(str(self.base / "a"), str(err.exception))
        self.assertIn(str(self.base / "b"), str(err.exception))

    def test_this_machine(self):
        """The repository's config: the market data folder of this machine is valid."""
        self.assertTrue(is_valid_root(paths.MARKETDATA))

    def test_download_writes_to_the_root(self):
        """The download runs the repository's mdbuild with ``--root`` = the market data folder."""
        from tqstudio.views.universe import DownloadRuns
        runs = DownloadRuns(root=self.base / "share")
        captured = {}

        class Popen:                                                        # capture the command, start nothing
            def __init__(self, command, **kwargs):
                captured["command"], captured["cwd"] = command, kwargs["cwd"]
                self.stdout = io.StringIO("")                                # an empty output

            def poll(self):
                return 0

        import tqstudio.views.universe as module
        saved = (module.subprocess.Popen, module.resolve_python, module.bloomberg_endpoint, module.bloomberg_reachable)
        module.subprocess.Popen, module.resolve_python = Popen, lambda candidates: sys.executable
        module.bloomberg_endpoint, module.bloomberg_reachable = lambda root: ("localhost", 8194), lambda h, p: True
        try:
            runs.start()
        finally:
            (module.subprocess.Popen, module.resolve_python, module.bloomberg_endpoint,
             module.bloomberg_reachable) = saved
        self.assertEqual(captured["command"][-4:], ["-m", "mdbuild", "--root", str(self.base / "share")])
        self.assertEqual(Path(captured["cwd"]), paths.CODE_MARKETDATA)


if __name__ == "__main__":
    unittest.main()
