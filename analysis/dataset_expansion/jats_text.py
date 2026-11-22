"""JATS full-text XML -> plain text with section titles, paragraph ids and figure captions."""
import sys, xml.etree.ElementTree as ET
def text(el): return " ".join("".join(el.itertext()).split())
root = ET.parse(sys.argv[1]).getroot()
out = []
front = root.find("front")
if front is not None:
    t = front.find(".//article-title"); out.append("# " + (text(t) if t is not None else ""))
    pd = front.find(".//pub-date"); 
    for ab in front.iter("abstract"): out.append("## Abstract\n" + text(ab))
body = root.find("body")
def walk(el, depth=2):
    for ch in el:
        if ch.tag == "sec":
            t = ch.find("title"); out.append("#" * depth + " " + (text(t) if t is not None else "")); walk(ch, depth + 1)
        elif ch.tag == "p":
            out.append(f"[{ch.get('id','p')}] " + text(ch))
        elif ch.tag in ("fig", "table-wrap", "fig-group"):
            lab = ch.find("label"); cap = ch.find("caption")
            gr = [g.get("{http://www.w3.org/1999/xlink}href") for g in ch.iter("graphic")]
            out.append(f"[{ch.get('id','fig')}] {text(lab) if lab is not None else ''}: {text(cap) if cap is not None else ''} (graphic: {', '.join(g for g in gr if g)})")
if body is not None: walk(body)
open(sys.argv[2], "w").write("\n\n".join(out))
print(sys.argv[2], len(out), "blocks")
