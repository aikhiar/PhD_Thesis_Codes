"""
tables.py -- every table of the reports as .csv (full precision) and as a LaTeX
booktabs .tex file (4 significant digits), ready for the paper.
"""

import csv
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def fmt(v, digits: int = 4) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{digits}g}" if np.isfinite(v) else ""
    return str(v)


def tex_escape(text: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"),
                 ("^", r"\^{}")):
        text = text.replace(a, b)
    return text


def _write_tex(header: List[str], body: List[List[str]], path: str) -> None:
    lines = [r"\begin{tabular}{" + "l" * len(header) + "}", r"\toprule",
             " & ".join(tex_escape(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(tex_escape(c) for c in row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_table(rows: List[Dict[str, Any]], columns: Sequence[Tuple[str, str]], out_dir: str, stem: str) -> None:
    """rows -> <stem>.csv (the keys of `columns`) and <stem>.tex (their headers)."""
    os.makedirs(out_dir, exist_ok=True)
    keys = [k for k, _ in columns]
    with open(os.path.join(out_dir, f"{stem}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for row in rows:
            writer.writerow([fmt(row.get(k), 8) for k in keys])
    _write_tex([h for _, h in columns], [[fmt(row.get(k)) for k in keys] for row in rows],
               os.path.join(out_dir, f"{stem}.tex"))


def csv_to_tex(csv_path: str) -> None:
    """LaTeX twin of an existing CSV table (numbers rounded to 4 significant
    digits), written next to it with the .tex extension."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return

    def cell(text: str) -> str:
        try:
            value = float(text)
        except ValueError:
            return text
        return text if text.strip().lstrip("-").isdigit() else fmt(value)

    header = [h.replace("_", " ") for h in rows[0]]
    _write_tex(header, [[cell(c) for c in r] for r in rows[1:]], os.path.splitext(csv_path)[0] + ".tex")
