"""Render a part's article.md into HTML you can paste straight into Medium.

    uv run --dev python tools/build_medium.py 01-ngrams

Medium's editor does not parse Markdown on paste — pasted `##`, `**` and backticks
arrive as literal characters. Pasting *rendered HTML* from a browser does work:
headings, bold, lists, links, blockquotes and code blocks all survive.

Open the generated file in a browser, select all, copy, paste into a new Medium draft.

Medium has no table support, so a table in the source is reported as an error rather
than silently pasted as a broken block.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent

PAGE = """<!doctype html>
<meta charset="utf-8">
<title>{title} — paste into Medium</title>
<style>
  body {{ max-width: 43rem; margin: 3rem auto; padding: 0 1.5rem;
         font: 19px/1.7 Charter, Georgia, serif; color: #222; }}
  h1 {{ font-size: 2.1rem; line-height: 1.2; }}
  h2 {{ font-size: 1.5rem; margin-top: 2.4rem; }}
  h3 {{ font-size: 1.2rem; color: #555; font-weight: 400; }}
  img {{ max-width: 100%; display: block; margin: 1.6rem auto 0.4rem; }}
  pre {{ background: #f6f8f8; padding: 1rem 1.1rem; overflow-x: auto;
         font: 15px/1.6 ui-monospace, Menlo, monospace; border-radius: 4px; }}
  code {{ font: 0.88em ui-monospace, Menlo, monospace; background: #f2f4f4;
          padding: 1px 5px; border-radius: 3px; }}
  pre code {{ background: none; padding: 0; }}
  blockquote {{ border-left: 3px solid #0C6F79; margin: 1.5rem 0;
                padding: 0.2rem 0 0.2rem 1.1rem; color: #555; }}
  hr {{ border: 0; border-top: 1px solid #ddd; margin: 2.5rem 0; }}
  .banner {{ font: 14px/1.5 system-ui, sans-serif; background: #FBEEDA;
             border: 1px solid #E0B462; padding: 0.9rem 1.1rem; border-radius: 6px;
             margin-bottom: 2.5rem; color: #5a4210; }}
</style>
<div class="banner">
  <b>Paste target: Medium.</b> Select all (Ctrl/Cmd-A), copy, paste into a new draft.
  This banner is outside the article body — delete it if it comes across.
  {figure_note}
</div>
{body}
"""


def inline_images(html: str, base: Path) -> tuple[str, int]:
    """Embed local images as data URIs so the page is self-contained."""
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        src = m.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = (base / src).resolve()
        if not path.exists():
            raise FileNotFoundError(f"image not found: {path}")
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode()
        count += 1
        return m.group(0).replace(src, f"data:{mime};base64,{data}")

    return re.sub(r'src="([^"]+)"', repl, html), count


def build(part: str) -> Path:
    src = ROOT / "parts" / part / "article.md"
    if not src.exists():
        sys.exit(f"no article at {src}")

    text = src.read_text()

    tables = [ln for ln in text.splitlines() if ln.lstrip().startswith("|")]
    if tables:
        sys.exit(
            f"{len(tables)} table line(s) found — Medium cannot render tables.\n"
            f"Rewrite them as lists before publishing. First: {tables[0][:70]}"
        )

    html = markdown.markdown(text, extensions=["fenced_code", "sane_lists", "attr_list"])
    html, n_img = inline_images(html, src.parent)

    title = next((ln[2:].strip() for ln in text.splitlines() if ln.startswith("# ")), part)
    note = (
        f"{n_img} figures are embedded. If any fail to carry across, upload them "
        f"manually from <code>parts/{part}/assets/images/</code>."
    )
    out = src.with_suffix(".medium.html")
    out.write_text(PAGE.format(title=title, body=html, figure_note=note))
    print(f"{out.relative_to(ROOT)}  ·  {out.stat().st_size / 1024:.0f} KB  ·  {n_img} figures inlined")
    return out


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "01-ngrams")
