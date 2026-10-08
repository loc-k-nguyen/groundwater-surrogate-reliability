import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_repository
from make_manifest import rows


class ReleaseTools(unittest.TestCase):
    def test_manifest_uses_platform_independent_path_order(self):
        paths = [row[0] for row in rows()]
        self.assertEqual(paths, sorted(paths))

    def test_repository_audit(self):
        verify_repository.audit_contents()

    def test_existing_analysis_destination_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "preserve.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/build_v4_analysis.py"),
                 "--output-dir", directory], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Nonempty output directory", result.stderr)
            self.assertEqual(sentinel.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()
