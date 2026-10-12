"""Export the English manuscript with vector illustrations and typeset math.

uv run --with pypandoc-binary --with typst --with pymupdf python \
    BinaryDiffusionAttention/export_paper_pdf.py
"""
from pathlib import Path
import re

import pypandoc
import typst
import pymupdf

ROOT = Path(__file__).resolve().parent
STYLE = r'''
#set document(title: "Staged Binary-Local Diffusion Attention: Effective Few-Step Diffusion Along the Time Axis", author: "HaoyangMa")
#set page(paper: "a4", margin: (x: 19mm, y: 20mm), numbering: "1", number-align: center)
#set text(font: "Libertinus Serif", size: 10.5pt, fill: rgb("#24364B"))
#set par(justify: true, leading: .65em)
#set heading(numbering: none)
#show heading.where(level: 1): set text(size: 22pt, weight: "bold", fill: rgb("#24364B"))
#show heading.where(level: 2): set text(size: 15pt, weight: "bold", fill: rgb("#008C82"))
#show heading.where(level: 3): set text(size: 12pt, weight: "bold")
#show link: set text(fill: rgb("#4169A1"))
#set image(width: 100%)
#let results-table(widths, headers, rows, numeric: true, size: 8.5pt) = {
  set text(font: "DejaVu Sans", size: size)
  set par(justify: false, leading: .5em)
  table(
    columns: widths,
    inset: (x: 5pt, y: 6pt),
    stroke: none,
    align: (x, y) => if numeric and x > 0 { right + horizon } else { left + horizon },
    fill: (x, y) => if y == 0 { rgb("#E5F1F0") }
      else if calc.even(y) { rgb("#F3F6F9") } else { none },
    table.hline(y: 0, stroke: .9pt + rgb("#008C82")),
    table.header(repeat: true, ..headers.map(h => text(weight: "bold", fill: rgb("#24364B"), h))),
    table.hline(y: 1, stroke: .5pt + rgb("#BACED0")),
    ..rows.flatten(),
    table.hline(stroke: .6pt + rgb("#BACED0")),
  )
}
#let table-title(number, title) = block(above: 9pt, below: 5pt, sticky: true)[
  #text(size: 9.5pt, weight: "bold", fill: rgb("#008C82"))[Table #number. #title]
]
#show raw: set text(size: 7pt)
#show raw.where(block: true): set block(fill: rgb("#F2F5F8"), inset: 7pt, radius: 3pt)
'''


def cell(value):
    """Retain Markdown emphasis/code while avoiding justified table text."""
    return "[" + pypandoc.convert_text(value, "typst", format="markdown",
                                      extra_args=["--wrap=none"]).strip() + "]"


def table_code(widths, headers, rows, numeric=True, size=8.5):
    def sequence(values):
        return "(" + ", ".join(cell(v) for v in values) + ",)"
    return (f"#results-table(({widths}), {sequence(headers)}, "
            "(" + ",\n".join(sequence(row) for row in rows) + ",), "
            f"numeric: {str(numeric).lower()}, size: {size}pt)\n")


def style_tables(source):
    """Use content-aware PDF layouts, leaving the source Markdown unchanged."""
    titles = ["Interaction budget at coverage-sufficient depth",
              "Matched model configurations", "Three-seed topology screening",
              "Hidden-state forward scaling", "Million-position training pilot",
              "Historical depth-sufficient 65K training", "Paired book and code accuracy",
              "Experiment and implementation artifacts"]
    index = 0

    def replace(match):
        nonlocal index
        lines = match.group(0).strip().splitlines()
        parsed = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
        rows = parsed[2:]
        number = index + 1
        title = titles[index]
        code = f'#table-title({number}, "{title}")\n'
        if index == 0:
            code += table_code("1fr, 1fr, 1.4fr", ["Positions", "Layers", "Dense / sparse edges"], rows)
        elif index == 1:
            code += table_code("1.5fr, .6fr, .6fr, .6fr, .8fr, .9fr, 1.2fr, 2fr",
                ["Experiment", "Width", "Layers", "Heads", "FFN", "Vocab.", "Parameters", "Budget"], rows, size=8)
        elif index == 2:
            code += table_code("2.5fr, 1.5fr, 1.5fr, 1.2fr",
                ["Route", "Train CE¹", "Validation CE", "Characters/s"], rows)
            code += '#block(above: 3pt, below: 7pt)[#text(size: 8pt, fill: rgb("#607084"))[¹ Mean of the final ten training updates; ± indicates between-seed variation.]]\n'
        elif index == 3:
            code += table_code("1fr, 1.3fr, 1.3fr, .9fr, 1.6fr",
                ["Context", "Dense chars/s", "Sparse chars/s", "Speedup", "Peak GiB (D / S)"], rows)
        elif index in (4, 5):
            # Keep both objectives visible; separate quality from systems metrics.
            quality = [row[:4] for row in rows]
            costs = [[row[0], *row[4:]] for row in rows]
            headers = (["Model", "Train CE / pass 1", "Train CE / pass 2", "Train CE / pass 3"]
                       if index == 4 else ["Model", "Train CE / pass 1", "Train CE / pass 20", "Valid. CE / pass 20"])
            code += '#block(breakable: false)[\n'
            code += table_code("1.6fr, 1fr, 1fr, 1.1fr", headers, quality)
            code += '#v(6pt)\n'
            code += table_code("1.6fr, 1fr, 1fr, 1.1fr",
                ["Model", "Training time (s)", "Characters/s", "Peak allocated GiB"], costs)
            code += ']\n'
        elif index == 6:
            code += table_code("1.3fr, 1.7fr, 1.3fr, 1.3fr, 1fr",
                ["Dataset", "Model", "Train CE", "Test CE", "Test accuracy"], rows, size=8)
        else:
            code += table_code("1.5fr, 4fr", ["Evidence", "Artifact (relative path)"], rows, numeric=False)
        index += 1
        return "\n```{=typst}\n" + code + "```\n\n"

    result = re.sub(r"(?m)^\|[^\n]*\n\|[-:| ]+\n(?:\|[^\n]*\n)+", replace, source)
    assert index == len(titles), f"Expected {len(titles)} tables, found {index}"
    return result


def main():
    source = (ROOT / "paper_en.md").read_text()
    # Preserve the Markdown original; substitute vector assets only for export.
    source = re.sub(r"\]\(figures/([^)]*)\.png\)", r"](/figures/\1.svg)", source)
    source = style_tables(source)
    body = pypandoc.convert_text(source, "typst",
        format="markdown-implicit_figures+tex_math_single_backslash",
        extra_args=["--wrap=none"])
    pdf = ROOT.parent / "paper_en.pdf"
    typst.compile((STYLE + "\n" + body).encode(), output=str(pdf), root=str(ROOT))
    doc = pymupdf.open(pdf)
    previews = ROOT / "figures" / "pdf_preview"
    previews.mkdir(exist_ok=True)
    chosen = {0}
    for i, page in enumerate(doc):
        text = page.get_text()
        if "Figure 1." in text or "Figure 4." in text or "Table " in text:
            chosen.add(i)
    for i in sorted(chosen):
        doc[i].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(previews / f"page_{i+1:02d}.png")
    for term in ["Figure 1.", "Figure 2.", "Figure 3.", "Figure 4.", "Figure 5."]:
        assert term in "\n".join(p.get_text() for p in doc), f"Missing caption: {term}"
    for i, page in enumerate(doc):
        for block in page.get_text("blocks"):
            assert block[0] >= 0 and block[2] <= page.rect.width + 1, f"Horizontal overflow on page {i+1}"
    print(f"Exported {pdf}: {len(doc)} pages, {pdf.stat().st_size / 1024:.0f} KiB")
    print("Preview pages:", ", ".join(str(i+1) for i in sorted(chosen)))


if __name__ == "__main__":
    main()
