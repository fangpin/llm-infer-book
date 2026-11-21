import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "doc_scripts"))

meta_path = Path(__file__).resolve().parent.parent / "project.json"
meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

project = meta.get("title", "LLM 推理引擎")
author = "Pin Fang"
copyright = "2026, Pin Fang"
html_show_sphinx = False
html_show_sourcelink = False

extensions = ["myst_parser", "lark_callouts"]
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "tasklist",
]
myst_heading_anchors = 3
source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_title = project
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
