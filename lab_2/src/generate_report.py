"""Generate a Markdown inference report from saved Ollama responses."""

from __future__ import annotations

import argparse
from typing import Any

from utils import dump_text, load_json_list, resolve_from_root


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for report generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="reports/inference_results.json",
        help="Path to the JSON file with inference results.",
    )
    parser.add_argument(
        "--output",
        default="reports/inference_report.md",
        help="Path to the generated Markdown report.",
    )
    return parser.parse_args()


def sanitize_cell(text: str) -> str:
    """Make text safe for one Markdown table cell."""
    return text.replace("\n", "<br>").replace("|", "\\|").strip()


def build_markdown_report(rows: list[dict[str, Any]]) -> str:
    """Build the final Markdown report with a two-column inference table."""
    lines = [
        "# Отчет инференса LLM",
        "",
        "| Запрос к LLM | Вывод LLM |",
        "|--------------|-----------|",
    ]
    for row in rows:
        prompt = sanitize_cell(str(row.get("prompt", "")))
        response = sanitize_cell(str(row.get("response", "")))
        lines.append(f"| {prompt} | {response} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """Read inference results and write a Markdown report."""
    args = parse_args()
    rows = load_json_list(args.input)
    report_text = build_markdown_report(rows)
    report_path = dump_text(report_text, args.output)
    print(f"report: {resolve_from_root(report_path)}")


if __name__ == "__main__":
    main()
