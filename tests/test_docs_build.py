import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]


class DocsBuildTest(unittest.TestCase):
    def test_html_build_with_and_without_local_metadata(self):
        for title in (None, "Fixture book"):
            with self.subTest(title=title), TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "docs" / "source"
                source.mkdir(parents=True)
                shutil.copyfile(ROOT / "docs" / "source" / "conf.py", source / "conf.py")
                shutil.copytree(ROOT / "doc_scripts", root / "doc_scripts")
                (source / "index.md").write_text(
                    "# Introduction\n\n"
                    '<callout emoji="tip">\n'
                    "**Reading route:**Start here.\n"
                    "- **Part one:**First.\n"
                    "- **Part two:**Second.\n\n"
                    "Read [the chapter](chapter.md).\n"
                    "</callout>\n\n"
                    "```{toctree}\n\nchapter\n```\n",
                    encoding="utf-8",
                )
                (source / "chapter.md").write_text(
                    "# Chapter\n\nA formula: $E = mc^2$.\n",
                    encoding="utf-8",
                )
                if title is not None:
                    (root / "docs" / "project.json").write_text(
                        json.dumps({"title": title}), encoding="utf-8",
                    )

                # No .venv launcher exists here; Make must use the selected Python.
                result = subprocess.run(
                    ["make", "-f", str(ROOT / "doc.mk"), "docs-html", f"PY={sys.executable}"],
                    cwd=root,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn("WARNING:", result.stderr)
                build = root / "docs" / "_build" / "html"
                index = (build / "index.html").read_text(encoding="utf-8")
                self.assertIn("chapter.html", index)
                self.assertNotIn("<callout", index)
                self.assertIn("<strong>Reading route:</strong>", index)
                self.assertIn("<strong>Part one:</strong>", index)
                self.assertIn('<div class="note admonition">', index)
                self.assertIn('<ul class="simple">', index)
                self.assertIn('<a class="reference internal" href="chapter.html">', index)
                if title is not None:
                    self.assertIn(title, index)
                chapter = (build / "chapter.html").read_text(encoding="utf-8")
                self.assertIn('class="math notranslate nohighlight"', chapter)


if __name__ == "__main__":
    unittest.main()
