from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional


def prepare_csv_for_llm_lab(
    src_path: str,
    dst_path: str,
    expected_col: str,
    encoding: str = "utf-8",
) -> None:
    """
    Normalize a raw CSV into a form that llm-lab can consume.

    - Copies all columns from src_path to dst_path
    - Adds or overwrites a column named 'expected_output'
      using the values from `expected_col`.

    This lets analysts keep their original column naming
    (e.g. 'Ideal_Answer', 'label') but still use llm-lab,
    which expects an 'expected_output' field in each example.

    Parameters
    ----------
    src_path : str
        Path to the input CSV file.
    dst_path : str
        Path to the output CSV file to create.
    expected_col : str
        Name of the column in the source CSV that contains
        the ground-truth / gold answer for evaluation.
    encoding : str, optional
        File encoding, default 'utf-8'.
    """
    src = Path(src_path)
    dst = Path(dst_path)

    if not src.exists():
        raise FileNotFoundError(src)

    with src.open("r", encoding=encoding, newline="") as f_in, \
         dst.open("w", encoding=encoding, newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])

        if expected_col not in fieldnames:
            raise ValueError(
                f"Column '{expected_col}' not found in {src_path}. "
                f"Available columns: {fieldnames}"
            )

        # Ensure 'expected_output' exists in the output schema
        if "expected_output" not in fieldnames:
            fieldnames.append("expected_output")

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # Copy original row and map expected_col -> expected_output
            row = dict(row)
            row["expected_output"] = row[expected_col]
            writer.writerow(row)
