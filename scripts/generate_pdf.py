"""Convert FINAL_REPORT.md to FINAL_REPORT.pdf using xhtml2pdf (pure Python)."""
from __future__ import annotations

from pathlib import Path
import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
MD_PATH  = ROOT / "FINAL_REPORT.md"
PDF_PATH = ROOT / "FINAL_REPORT.pdf"

CSS_STYLE = """
@page {
    size: a4 portrait;
    margin: 2.2cm 2.5cm 2.5cm 2.5cm;
    @frame footer {
        -pdf-frame-content: footer_content;
        bottom: 1cm; left: 2.5cm; right: 2.5cm; height: 0.8cm;
    }
}

body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.6;
    color: #1a1a1a;
}

/* Headings */
h1 {
    font-size: 24pt;
    font-weight: bold;
    color: #0f4c81;
    border-bottom: 3px solid #0f4c81;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 4px;
}
h1 + h3 {
    font-size: 10.5pt;
    font-weight: normal;
    color: #555;
    margin-bottom: 18px;
    border: none;
}
h2 {
    font-size: 15pt;
    font-weight: bold;
    color: #0f4c81;
    border-bottom: 1.5px solid #d0d8e4;
    padding-bottom: 4px;
    margin-top: 28px;
    margin-bottom: 10px;
}
h3 {
    font-size: 12pt;
    font-weight: bold;
    color: #2c3e50;
    margin-top: 18px;
    margin-bottom: 6px;
}
h4 {
    font-size: 10.5pt;
    font-weight: bold;
    color: #1a1a1a;
    margin-top: 12px;
    margin-bottom: 4px;
}

/* Paragraph */
p { margin-bottom: 8px; }

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 9pt;
}
thead { background-color: #0f4c81; color: white; }
thead th { padding: 6px 8px; text-align: left; font-weight: bold; }
tbody tr:nth-child(even) { background-color: #f2f6fb; }
tbody td { padding: 5px 8px; border-bottom: 1px solid #dde3ea; vertical-align: top; }

/* Code */
pre {
    background: #1e2638;
    color: #e8eaf6;
    border-radius: 4px;
    padding: 10px 14px;
    font-family: Courier, monospace;
    font-size: 8pt;
    margin: 10px 0;
}
code {
    font-family: Courier, monospace;
    font-size: 8.5pt;
    background: #eef2f7;
    color: #c0392b;
    padding: 1px 4px;
    border-radius: 2px;
}
pre code { background: transparent; color: inherit; padding: 0; }

/* Blockquote */
blockquote {
    border-left: 4px solid #00d4aa;
    background: #f0fdf9;
    margin: 12px 0;
    padding: 8px 14px;
    color: #444;
    font-style: italic;
}
blockquote p { margin-bottom: 0; }

/* Lists */
ul, ol { margin: 6px 0 8px 20px; }
li { margin-bottom: 3px; }

/* HR */
hr { border: none; border-top: 1.5px solid #d0d8e4; margin: 22px 0; }

/* Links */
a { color: #0f4c81; }

/* Footer */
#footer_content {
    font-size: 8pt;
    color: #888;
    text-align: center;
}
"""


def convert() -> None:
    """Read FINAL_REPORT.md, render to HTML, write PDF."""
    md_text = MD_PATH.read_text(encoding="utf-8")

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "nl2br", "sane_lists"],
    )

    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>{CSS_STYLE}</style>
</head>
<body>
  {html_body}
  <div id="footer_content">
    FairSquare — Technical Report · April 2026 · Page <pdf:pagenumber> / <pdf:pagecount>
  </div>
</body>
</html>"""

    with PDF_PATH.open("wb") as pdf_file:
        result = pisa.CreatePDF(full_html, dest=pdf_file, encoding="utf-8")

    if result.err:
        print(f"❌  PDF generation failed with {result.err} error(s).")
    else:
        size_kb = PDF_PATH.stat().st_size / 1024
        print(f"✅  PDF generated: {PDF_PATH}")
        print(f"    Size: {size_kb:.1f} KB")


if __name__ == "__main__":
    convert()
