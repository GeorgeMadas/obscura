"""Build the single-file page ttyd serves (ttyd's --index serves one file and nothing else).

Inlines the vendored xterm.js files into index.html and fills in the setup dashboard's public port,
so the page loads with no CDN and the Dashboard quick button can link across ports.
Usage: python3 build.py <source dir> <output file>
"""
import os
import re
import sys

src, out = sys.argv[1], sys.argv[2]
html = open(os.path.join(src, "index.html"), encoding="utf-8").read()


def include(m):
    path = os.path.join(src, m.group(1))
    body = open(path, encoding="utf-8").read()
    if path.endswith(".css"):
        return "<style>\n" + body + "\n</style>"
    return "<script>\n" + body.replace("</script", "<\\/script") + "\n</script>"


html = re.sub(r"<!--#include ([\w./-]+)-->", include, html)
html = html.replace("__DASHBOARD_PORT__", os.environ.get("VAST_TCP_PORT_6006", ""))
tmp = out + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    f.write(html)
os.replace(tmp, out)
print(f"built {out} ({len(html) // 1024} KB)")
