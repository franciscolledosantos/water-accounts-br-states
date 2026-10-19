"""Builds the Ecological Economics submission .docx (double spacing, continuous line numbers, page numbers)
from data/manuscript_en.md + data/manuscript_en_parts.py. Usage: python manuscript_en_docx.py out.docx [gwbench_sentence]"""
import os, re, sys, subprocess, importlib.util
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = os.path.join(W, "manuscrito"); F = os.path.join(W, "figuras")
spec = importlib.util.spec_from_file_location("parts", os.path.join(W, "scripts", "manuscript_parts.py")); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
md = open(os.path.join(D, "manuscript_en.md"), encoding="utf-8").read()
gw = sys.argv[2] if len(sys.argv) > 2 else ""
import sys as _s; _s.path.insert(0, os.path.join(W, "scripts"))
for k, v in P.TAB.items():
    md = md.replace(f"[[{k}]]", v)
for k, (arq, leg) in P.FIG.items():
    md = md.replace(f"[[{k}]]", f"![]({os.path.join(F, arq)}){{width=16cm}}\n\n{leg}")
# rev. 17/10: equations rendered as PNG (matplotlib mathtext) so that LibreOffice's PDF export keeps them;
# the LaTeX source stays in manuscrito/manuscript_en.md
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
_eq = [0]
def _render(m):
    _eq[0] += 1
    tex = m.group(1).strip().replace("\\dfrac", "\\frac").replace("\\qquad", "\\quad\\quad").replace("\\,", " ")
    fig = plt.figure(figsize=(6.3, 0.9)); fig.text(0.5, 0.5, "$" + tex + "$", ha="center", va="center", fontsize=11,
                                               fontfamily="serif", math_fontfamily="stix")
    path = os.path.join(F, f"eq{_eq[0]}.png"); fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
    return f"![]({path}){{width=15cm}}"
md = re.sub(r"```latex\n(.*?)\n```", _render, md, flags=re.S)
tmp = os.path.join(D, "manuscript_en_full.md"); open(tmp, "w", encoding="utf-8").write(md)
out = sys.argv[1]
subprocess.run(["pandoc", tmp, "-f", "markdown+tex_math_dollars+pipe_tables", "-o", out], check=True)
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
doc = Document(out)
for st in doc.styles:
    try:
        if st.type == 1:
            st.font.name = "Times New Roman"; st.font.size = Pt(12) if st.name not in ("Title",) else Pt(14)
            rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
            if rf is None: rf = OxmlElement("w:rFonts"); rpr.append(rf)
            for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"): rf.set(qn(a), "Times New Roman")
            st.paragraph_format.line_spacing = 2.0
            if st.name.startswith("Heading") or st.name == "Title":
                st.font.color.rgb = RGBColor(0, 0, 0); st.font.bold = True
    except Exception:
        pass
for p in doc.paragraphs:
    p.paragraph_format.line_spacing = 2.0
from docx.shared import Cm
for t in doc.tables:
    ncol = len(t.columns)
    first = t.rows[0].cells[0].text.strip() if t.rows else ""
    wide = {"Specification": 6.0, "Pair": 4.5}.get(first)      # rev. 17/10: wider label column in Tables 3 and 4
    t.autofit = False
    if wide:                                                      # fixed layout + grid widths so LibreOffice honours them
        tblPr = t._tbl.tblPr
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        grid = t._tbl.tblGrid
        for j, gc in enumerate(grid.findall(qn("w:gridCol"))):
            gc.set(qn("w:w"), str(int((wide if j == 0 else (16.0 - wide) / (ncol - 1)) * 567)))
    for row in t.rows:
        for j, c in enumerate(row.cells):
            if wide:
                c.width = Cm(wide) if j == 0 else Cm((16.0 - wide) / (ncol - 1))
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.0
                for r in p.runs: r.font.size = Pt(10 if ncol <= 5 else 8.5)
for s in doc.sections:
    sp = s._sectPr
    ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous"); sp.append(ln)
    fp = s.footer.paragraphs[0]
    if "PAGE" not in fp._p.xml:
        fp.alignment = 1
        r = fp.add_run(); f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin"); r._r.append(f1)
        r2 = fp.add_run(); it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = "PAGE"; r2._r.append(it)
        r3 = fp.add_run(); f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end"); r3._r.append(f2)
doc.save(out)
body = md.split("## CRediT")[0]
words = len(re.sub(r"!\[\]\(.*?\)\{.*?\}|\|.*\|", " ", body.split("## 1. Introduction")[1]).split())
print("ok", out, os.path.getsize(out), "words (Introduction→Conclusions, sem tabelas):", words)
