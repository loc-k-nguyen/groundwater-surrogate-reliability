"""Check scoped licensing and metadata without importing model implementations."""

import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseNoticesTests(unittest.TestCase):
    def test_manifest_excludes_git_administration(self):
        spec = importlib.util.spec_from_file_location("release_manifest", ROOT / "scripts/make_manifest.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".git").mkdir()
            (root / ".git/config").write_text("private transport configuration", encoding="utf-8")
            (root / "retained.txt").write_text("retained", encoding="utf-8")
            module.ROOT = root
            module.MANIFEST = root / "MANIFEST_CODE_REVIEW.csv"
            self.assertEqual([row[0] for row in module.rows()], ["retained.txt"])

    def test_original_code_license_and_font_notice_are_separate(self):
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("Loc K. Nguyen", license_text)
        notice = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        self.assertIn("does not relicense third-party fonts", notice)
        self.assertIn("not the official", notice)
        font_notice = (ROOT / "DejaVu-fonts.txt").read_text(encoding="utf-8")
        for term in ("Bitstream", "Arev", "Permission"):
            self.assertIn(term, font_notice)

    def test_canonical_repository_metadata_and_unused_dependency(self):
        self.assertIn("* -text", (ROOT / ".gitattributes").read_text(encoding="utf-8"))
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        self.assertIn("https://github.com/loc-k-nguyen/groundwater-surrogate-reliability", citation)
        self.assertIn("license: MIT", citation)
        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("scikit-learn", requirements)
        for path in [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").glob("*.py")]:
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(any(name.split(".")[0] == "sklearn" for name in names), str(path))


if __name__ == "__main__":
    unittest.main()
