#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


LEAVE_MIN_MAX = {
    "HR": (0, 12),
    "Management": (0, 10),
    "Software": (0, 15),
    "Data Science": (0, 14),
    "Design": (0, 13),
}
OVERTIME_MIN_MAX = {
    "HR": (0, 32),
    "Management": (0, 28),
    "Software": (2, 40),
    "Data Science": (1, 38),
    "Design": (0, 34),
}


def _employee_num(row: dict[str, str]) -> int:
    try:
        return int(row.get("Employee_ID", "") or 0)
    except ValueError:
        return 0


def _experience_years(row: dict[str, str]) -> int:
    try:
        return int(row.get("Experience_Years", "0") or 0)
    except ValueError:
        return 0


def pick_leave_days(row: dict[str, str]) -> int:
    employee_id = row.get("Employee_ID", "")
    department = row.get("Department", "")
    experience = _experience_years(row)

    low, high = LEAVE_MIN_MAX.get(department, (0, 12))
    if experience >= 15:
        low, high = max(0, low - 1), high + 2
    elif experience == 0:
        high = max(3, high - 2)

    seeded_rng = random.Random(f"{employee_id}:{department}:{experience}:leave")
    return seeded_rng.randint(low, high)


def pick_overtime_hours(row: dict[str, str]) -> int:
    employee_id = row.get("Employee_ID", "")
    department = row.get("Department", "")
    experience = _experience_years(row)
    low, high = OVERTIME_MIN_MAX.get(department, (0, 36))

    if experience >= 15:
        low, high = max(0, low - 2), max(high - 4, 24)
    elif experience == 0:
        high = max(24, high - 6)

    if row.get("Remote_Work", "").strip().lower() == "yes":
        high = max(low + 6, high - 4)

    seeded_rng = random.Random(f"{employee_id}:{department}:{experience}:overtime")
    return seeded_rng.randint(low, high)


def pick_training_status(row: dict[str, str]) -> str:
    employee_id = row.get("Employee_ID", "")
    department = row.get("Department", "")
    experience = _experience_years(row)
    seeded_rng = random.Random(f"{employee_id}:{department}:{experience}:training")
    return "Yes" if seeded_rng.random() < 0.98 else "No"


def _cohort_key(row: dict[str, str]) -> tuple[str, int]:
    department = row.get("Department", "Unknown")
    experience = _experience_years(row)
    return department, experience // 5


def _set_leave_abuse(row: dict[str, str]) -> None:
    employee_id = _employee_num(row)
    department = row.get("Department", "")
    experience = _experience_years(row)
    seeded_rng = random.Random(f"{employee_id}:{department}:{experience}:leave-abuse")
    row["Leave"] = str(seeded_rng.randint(18, 30))


def _set_overtime_breach(row: dict[str, str]) -> None:
    employee_id = _employee_num(row)
    department = row.get("Department", "")
    experience = _experience_years(row)
    seeded_rng = random.Random(f"{employee_id}:{department}:{experience}:overtime-breach")
    row["Overtime_Hours"] = str(seeded_rng.randint(52, 72))


def _set_training_missing(row: dict[str, str]) -> None:
    row["Mandatory_Training_Complete"] = "No"


def _set_payroll_outlier(row: dict[str, str], salary: int) -> None:
    row["Salary_USD"] = str(max(1, int(salary)))


def inject_anomalies(rows: list[dict[str, str]]) -> None:
    cohorts: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("Salary_USD"):
            cohorts[_cohort_key(row)].append(index)

    best_by_department: dict[str, tuple[tuple[str, int], list[int]]] = {}
    for cohort_key, members in cohorts.items():
        if len(members) < 8:
            continue
        department, _ = cohort_key
        current_best = best_by_department.get(department)
        if current_best is None or len(members) > len(current_best[1]):
            best_by_department[department] = (cohort_key, members)

    selected_cohorts = sorted(best_by_department.values(), key=lambda item: len(item[1]), reverse=True)

    used_rows: set[int] = set()
    for _, members in selected_cohorts:
        salaries = sorted(
            int(rows[index]["Salary_USD"]) for index in members if rows[index].get("Salary_USD")
        )
        if not salaries:
            continue
        median = salaries[len(salaries) // 2]
        low_idx = next((index for index in members if index not in used_rows), None)
        high_idx = next(
            (index for index in reversed(members) if index not in used_rows and index != low_idx),
            None,
        )
        if low_idx is not None:
            _set_payroll_outlier(rows[low_idx], round(median * 0.35))
            used_rows.add(low_idx)
        if high_idx is not None:
            _set_payroll_outlier(rows[high_idx], round(median * 2.8))
            used_rows.add(high_idx)

    remaining_rows = [index for index in range(len(rows)) if index not in used_rows]
    seeded_rng = random.Random("synthetic-anomaly-plan")
    seeded_rng.shuffle(remaining_rows)

    plans = (
        (_set_leave_abuse, 18),
        (_set_overtime_breach, 18),
        (_set_training_missing, 16),
    )
    cursor = 0
    for mutator, count in plans:
        for _ in range(count):
            if cursor >= len(remaining_rows):
                return
            index = remaining_rows[cursor]
            cursor += 1
            mutator(rows[index])
            used_rows.add(index)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Augment the employee dataset with leave, overtime, training, and synthetic "
            "anomaly coverage."
        )
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

        for field in ["Leave", "Overtime_Hours", "Mandatory_Training_Complete"]:
            if field not in fieldnames:
                fieldnames.append(field)

        rows: list[dict[str, str]] = []
        for row in reader:
            row["Leave"] = str(pick_leave_days(row))
            row["Overtime_Hours"] = str(pick_overtime_hours(row))
            row["Mandatory_Training_Complete"] = pick_training_status(row)
            rows.append(row)

    inject_anomalies(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
