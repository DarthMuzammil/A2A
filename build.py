"""Build index.html (standalone site) from page.html (page body + head section)."""
from pathlib import Path

src = Path(__file__).with_name("page.html").read_text(encoding="utf-8")
head, body = src.split("<!--/HEAD-->", 1)
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
{head.strip()}
</head>
<body>
{body.strip()}
</body>
</html>
"""
Path(__file__).with_name("index.html").write_text(html, encoding="utf-8")
print("wrote index.html")
