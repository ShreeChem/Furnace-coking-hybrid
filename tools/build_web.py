"""
Builds the web page from web/template.html + web/data.json.
  web/index.html   - the page body (published as the private artifact)
  docs/index.html  - the same page as a full HTML document, for GitHub Pages
Run:  python3 tools/build_web.py
"""
import os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
tpl = open(os.path.join(ROOT, "web", "template.html"), encoding="utf-8").read()
data = open(os.path.join(ROOT, "web", "data.json"), encoding="utf-8").read()
assert "/*__DATA__*/null" in tpl
page = tpl.replace("/*__DATA__*/null", data)
open(os.path.join(ROOT, "web", "index.html"), "w", encoding="utf-8").write(page)
os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
full = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
        '[hidden]{display:none!important}img{max-width:100%}</style>\n'
        + page.split("</style>", 1)[0] + "</style>\n</head>\n<body>\n" + page.split("</style>", 1)[1] + "\n</body>\n</html>\n")
open(os.path.join(ROOT, "docs", "index.html"), "w", encoding="utf-8").write(full)
print(f"web/index.html {len(page)//1024} kB, docs/index.html {len(full)//1024} kB")
