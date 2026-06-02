from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer()


@app.command()
def main(outputs_dir: str = "outputs") -> None:
    rows = []
    for metrics_path in Path(outputs_dir).rglob("metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append({"experiment": metrics_path.parent.name, **metrics})
    table = pd.DataFrame(rows).sort_values("experiment") if rows else pd.DataFrame()
    output_path = Path(outputs_dir) / "paper_tables.csv"
    table.to_csv(output_path, index=False)


if __name__ == "__main__":
    app()
