"""Shared figure style, flow diagrams and the Word writer for the manuscripts."""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

NAVY, TEAL, ORANGE, RED, GREY, GOLD = "#16324F", "#1B998B", "#F18F01", "#C73E1D", "#8D99AE", "#E9C46A"
PAL = [NAVY, TEAL, ORANGE, RED, GREY, GOLD]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False, "axes.axisbelow": True})


def box(ax, x, y, w, h, title, body="", color=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.018", fc="white", ec=color, lw=1.4))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.012), w, 0.012, boxstyle="square,pad=0", fc=color, ec=color))
    lines = textwrap.wrap(title, max(10, int(w * 85)))
    ax.text(x + w / 2, y + h - 0.045, "\n".join(lines), ha="center", va="top", fontsize=8.5, fontweight="bold", color=color)
    if body:
        text = "\n".join(l for part in body.split("\n") for l in textwrap.wrap(part, max(12, int(w * 100))))
        ax.text(x + w / 2, y + h - 0.06 - 0.052 * len(lines), text, ha="center", va="top", fontsize=7.4, color="#222222")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=GREY))


def flow(path, steps, title="", height=2.2):
    """steps: list of (title, body); drawn left to right."""
    fig, ax = plt.subplots(figsize=(10, height))
    ax.set_xlim(-0.008, 1.008), ax.set_ylim(0, 1), ax.axis("off")
    n = len(steps)
    gap = 0.025
    w = (1 - gap * (n - 1)) / n
    for i, (t, b) in enumerate(steps):
        x = i * (w + gap)
        box(ax, x, 0.08, w, 0.78, t, b, PAL[i % 4])
        if i:
            arrow(ax, (x - gap + 0.002, 0.47), (x - 0.002, 0.47))
    if title:
        ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)


def shade(cell, hex_color):
    pr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"), s.set(qn("w:color"), "auto"), s.set(qn("w:fill"), hex_color)
    pr.append(s)


def page_field(par):
    run = par.add_run()
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if tag:
            e = OxmlElement("w:fldChar")
            e.set(qn("w:fldCharType"), tag)
        else:
            e = OxmlElement("w:instrText")
            e.text = text
        run._r.append(e)


def write_docx(path, blocks):
    """blocks: ('title'|'h'|'h2'|'p'|'small', text) | ('ul', [..]) | ('table', caption, head, rows) | ('fig', png, caption, spec) | ('ref', [..])"""
    d = Document()
    sec = d.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Inches(1)
    st = d.styles["Normal"]
    st.font.name, st.font.size = "Times New Roman", Pt(11)
    st.paragraph_format.line_spacing, st.paragraph_format.space_after = 1.3, Pt(6)
    for name, size in (("Title", 16), ("Heading 1", 13), ("Heading 2", 11.5)):
        h = d.styles[name]
        h.font.name, h.font.size, h.font.bold, h.font.color.rgb = "Times New Roman", Pt(size), True, RGBColor(0x16, 0x32, 0x4F)
        h.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Times New Roman")
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme"):
            h.element.rPr.rFonts.attrib.pop(qn(a), None)
    foot = sec.footer.paragraphs[0]
    foot.alignment = WD_ALIGN_PARAGRAPH.CENTER
    page_field(foot)
    nf = nt = 0
    for b in blocks:
        kind = b[0]
        if kind == "title":
            d.add_paragraph(b[1], style="Title").alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif kind == "h":
            d.add_heading(b[1], 1)
        elif kind == "h2":
            d.add_heading(b[1], 2)
        elif kind == "p":
            d.add_paragraph(b[1]).alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "small":
            r = d.add_paragraph().add_run(b[1])
            r.font.size, r.italic = Pt(9), True
        elif kind == "ul":
            for x in b[1]:
                d.add_paragraph(x, style="List Bullet")
        elif kind == "ref":
            for i, x in enumerate(b[1], 1):
                q = d.add_paragraph(f"[{i}] {x}")
                q.paragraph_format.left_indent, q.paragraph_format.first_line_indent = Inches(0.35), Inches(-0.35)
        elif kind == "table":
            nt += 1
            r = d.add_paragraph().add_run(f"Table {nt}. {b[1]}")
            r.font.size, r.bold = Pt(9.5), True
            t = d.add_table(rows=1, cols=len(b[2]))
            t.style, t.alignment = "Table Grid", WD_TABLE_ALIGNMENT.CENTER
            for i, row in enumerate([b[2]] + b[3]):
                cells = t.rows[0].cells if i == 0 else t.add_row().cells
                for c, v in zip(cells, row):
                    c.text = str(v)
                    run = c.paragraphs[0].runs[0]
                    run.font.size, run.bold = Pt(9), i == 0
                    c.paragraphs[0].paragraph_format.space_after = Pt(0)
                    if i == 0:
                        shade(c, "DCE6F0")
            d.add_paragraph()
        elif kind == "fig":
            nf += 1
            d.add_picture(str(b[1]), width=Inches(6.2))
            d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = d.add_paragraph().add_run(f"Figure {nf}. {b[2]}")
            r.font.size, r.bold = Pt(9.5), True
            r = d.add_paragraph().add_run(f"Image specification: {b[3]}")
            r.font.size, r.italic = Pt(8), True
    d.save(path)
    words = sum(len(str(x).split()) for b in blocks for x in (b[1] if b[0] in ("ul", "ref") else [b[1]] if b[0] in ("p", "h", "h2", "title") else []))
    return {"figures": nf, "tables": nt, "words": words}
