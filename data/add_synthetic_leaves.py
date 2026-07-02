#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


LEAVE_MIN_MAX = {
    "HR": (0, 12),
    "Management": (0, 10),
    "Software": (0, 15),
    "Data Science": (0, 14),
    "Design": (0, 13),
}


def pick_leave_days(row: dict[str, str]) -> int:
    employee_id = row.get("Employee_ID", "")
    department = row.get("Department", "")
    experience = int(row.get("Experience_Years", "0") or 0)

    low, high = LEAVE_MIN_MAX.get(department, (0, 12))

    if experience >= 15:
        low, high = max(0, low - 1), high + 2
    elif experience == 0:
        high = max(3, high - 2)

    seeded_rng = random.Random(f"{employee_id}:{department}:{experience}")
    return seeded_rng.randint(low, high)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add a realistic Leave column to the employee dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/tech_company_employee_data_1000.csv"),
        help="Path to the source CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tech_company_employee_data_1000_with_leave.csv"),
        help="Path for the augmented CSV.",
    )
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames or [])

        if "Leave" not in fieldnames:
            fieldnames.append("Leave")

        rows: list[dict[str, str]] = []
        for row in reader:
            row["Leave"] = str(pick_leave_days(row))
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()