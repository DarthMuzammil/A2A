"""Build the static site: src/*.html (head part, <!--/HEAD-->, body part) -> index.html, pricing.html.

Also writes .artifact/ copies for the preview link (page body only, without the html/head/body wrapper).
"""
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "src"
HEAD_COMMON = """<meta name="claude:viewport-fit-sides" content="cover">
<meta name="theme-color" content="#ffffff">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Onest:wght@500;600;700&display=swap">
<link rel="stylesheet" href="assets/site.css">"""

header = (SRC / "header.html").read_text(encoding="utf-8")
footer = (SRC / "footer.html").read_text(encoding="utf-8")
(ROOT / ".artifact").mkdir(exist_ok=True)

for page, key in [("index.html", "product"), ("pricing.html", "pricing")]:
    src = (SRC / page).read_text(encoding="utf-8")
    head, body = src.split("<!--/HEAD-->", 1)
    h = header
    for k in ("product", "pricing"):
        h = h.replace("{{cur-%s}}" % k, ' aria-current="page"' if k == key else "")
    body = body.replace("<!--@header-->", h).replace("<!--@footer-->", footer)
    head = head.strip() + "\n" + HEAD_COMMON
    full = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{head}
</head>
<body>
{body.strip()}
</body>
</html>
"""
    (ROOT / page).write_text(full, encoding="utf-8")
    (ROOT / ".artifact" / page).write_text(head + "\n" + body.strip() + "\n", encoding="utf-8")
    print("wrote", page)
