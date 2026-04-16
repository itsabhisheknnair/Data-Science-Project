"""
Generate crash_risk_model.ipynb from the current crash_risk_model.py.

This keeps the notebook synchronized with the script implementation. Run:

    python build_notebook.py
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import nbformat as nbf


PY_PATH = Path("crash_risk_model.py")
NB_PATH = Path("crash_risk_model.ipynb")

NOTEBOOK_METADATA = {
    "kernelspec": {"display_name": "base", "language": "python", "name": "python3"},
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    },
}


def split_script_into_cells(source: str) -> list[dict]:
    module_doc = ast.get_docstring(ast.parse(source)) or ""
    body = strip_module_docstring(source)

    cells = [
        nbf.v4.new_markdown_cell(build_intro_markdown(module_doc)),
    ]

    section_pattern = re.compile(r"^#\s+(\d+(?:\.\d+)?)\.\s+(.+?)\s*$")
    current_lines: list[str] = []
    current_title: str | None = "Imports and Setup"

    for line in body.splitlines():
        match = section_pattern.match(line)
        is_decorative = line.startswith("# ") and ("â•" in line or set(line[2:].strip()) <= {"=", "-", "═"})
        if match and not is_decorative:
            append_section_cells(cells, current_title, current_lines)
            number, title = match.groups()
            current_title = f"{number}. {title.strip()}"
            current_lines = [line]
        else:
            current_lines.append(line)

    append_section_cells(cells, current_title, current_lines)
    return [cell for cell in cells if cell.get("source", "").strip()]


def strip_module_docstring(source: str) -> str:
    if not source.startswith('"""'):
        return source
    end = source.find('"""', 3)
    if end == -1:
        return source
    return source[end + 3 :].lstrip("\n")


def build_intro_markdown(module_doc: str) -> str:
    intro = [
        "# ESG Controversy Signals for Equity Crash-Risk Monitoring",
        "",
        "This notebook is synced from `crash_risk_model.py` and includes the latest FDS rubric readiness outputs.",
    ]
    if module_doc:
        intro.extend(["", "```text", module_doc.strip(), "```"])
    return "\n".join(intro)


def append_section_cells(cells: list[dict], title: str | None, lines: list[str]) -> None:
    source = "\n".join(lines).strip()
    if not source:
        return
    if title:
        cells.append(nbf.v4.new_markdown_cell(f"## {title}"))
    cells.append(nbf.v4.new_code_cell(source))


def build_notebook() -> nbf.NotebookNode:
    source = PY_PATH.read_text(encoding="utf-8")
    notebook = nbf.v4.new_notebook()
    notebook.metadata = NOTEBOOK_METADATA
    notebook.cells = split_script_into_cells(source)
    nbf.validate(notebook)
    return notebook


def main() -> None:
    notebook = build_notebook()
    nbf.write(notebook, NB_PATH)
    code_cells = sum(1 for cell in notebook.cells if cell.cell_type == "code")
    markdown_cells = sum(1 for cell in notebook.cells if cell.cell_type == "markdown")
    print(f"Wrote {NB_PATH} with {code_cells} code cells and {markdown_cells} markdown cells.")


if __name__ == "__main__":
    main()
