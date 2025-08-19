#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_par_levels.py — Wizard-style Weekly -> 10-Day Par Calculator (Product#-based boxes)

What it does
------------
• Reads Wizard "Daily Consumption" exports with a 2-row header (level_0 groups, level_1 periods/"Product #").
• Auto-detects header rows even if there are title/junk rows above.
• Selects weekly columns by FIRST header row (e.g., "Act. Consumpt. Qty", "Theoretical Qty", etc).
• Weekly -> daily (÷7), averages across chosen weeks (mean/median/trimmed), optional ignore-zero-weeks.
• Computes Par = ceil(avg_daily × coverage_days × (1 + safety)); default coverage_days=10.
• Maps Categories by Product # (optional).
• Maps Pcs in Box by Product # (from BoxQuantities.xlsx) and creates:
    - Par10d_actual_qty_boxes = ceil(Par10d_actual_qty / Pcs in Box)
    - Par10d_theo_qty_boxes   = ceil(Par10d_theo_qty / Pcs in Box)
  (NaN when Pcs in Box is missing/0 or the Par is NaN; no “min 1 box” rule.)

Defaults (edit below or override with CLI)
-----------------------------------------
Input              : ./data/Daily_Consumption.xlsx
Box Quantities     : ./data/BoxQuantities.xlsx        (Product #, Product Name, Pcs in Box)
Product Categories : ./data/product_categories.xlsx   (category, Product #, Product Name)
Category Safety    : ./data/category_safety.xlsx      (category, safety_pct) [optional]
Output             : ./out/par_levels_10d.xlsx
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# ======================
# Defaults
# ======================
DEFAULT_INPUT = "./data/Daily_Consumption.xlsx"
DEFAULT_BOX_PATH = "./data/BoxQuantities.xlsx"          # Product #, Product Name, Pcs in Box
DEFAULT_CAT_PATH = "./data/product_categories.xlsx"     # category, Product #, Product Name
DEFAULT_CAT_SAFETY_PATH = "./data/category_safety.xlsx" # category, safety_pct (optional)
DEFAULT_OUT = "./out/par_levels_10d.xlsx"

DEFAULT_SHEET = 0
DEFAULT_COVERAGE_DAYS = 10     # keep 10 to align with requested column names
DEFAULT_SAFETY = 0.00
DEFAULT_WEEKS = "all"          # "all" or integer like "8" (most recent N weeks by column order)
DEFAULT_METRICS = "actual_qty,theo_qty"  # can add actual_value,theo_value
DEFAULT_METHOD = "mean"        # mean | median | trimmed
DEFAULT_TRIM_PCT = 0.10
DEFAULT_IGNORE_ZERO_WEEKS = False


# ======================
# Config / Aliases
# ======================
ID_L0_NAMES = {
    "location": ["Location Name", "Location"],
    "product_name": ["Product Name", "Item Name"],
    "product_num": [("Metrics", "Product #"), ("Metrics", "Product#"), ("Metrics", "Product Number")],
}

# Weekly families (match FIRST header row / level_0)
METRIC_L0_ALIASES = {
    "actual_qty": [
        "Act. Consumpt. Qty", "Actual Consumpt. Qty", "Actual Consumption Qty",
        "Actual Qty", "Consumpt. Qty"
    ],
    "theo_qty": [
        "Theoretical Qty", "Theo Qty", "Theoretical Consumption Qty"
    ],
    "actual_value": [
        "Act. Consumpt. Value (L)", "Act. Consumpt. Value", "Actual Consumption Value", "Actual Value"
    ],
    "theo_value": [
        "Theoretical Value (L)", "Theoretical Value"
    ],
}


# ======================
# Small utils
# ======================
def log(msg: str) -> None:
    print(f"[INFO] {msg}")

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def to_num(x):
    if isinstance(x, str):
        s = x.strip().replace(",", "")
        if s in {"--", "—", "", "NA", "N/A", "None"}:
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    return x if isinstance(x, (int, float, np.floating)) else np.nan

def avg(values: np.ndarray, method: str = "mean", trim_pct: float = 0.1) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    if method == "median":
        return float(np.median(arr))
    if method == "trimmed":
        if not (0 <= trim_pct < 0.5):
            trim_pct = 0.1
        arr = np.sort(arr)
        n = len(arr); k = int(np.floor(n * trim_pct))
        if k * 2 >= n:
            return float(arr.mean())
        return float(arr[k:n - k].mean())
    return float(arr.mean())

def resolve_required_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    # try ./data/<name> if a bare filename was given
    if not any(sep in raw for sep in ("/", "\\")):
        cand = Path("data") / raw
        if cand.exists():
            return cand
    raise FileNotFoundError(f"File not found: {raw}")

def resolve_optional_path(raw: str | None) -> Optional[Path]:
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    if not any(sep in raw for sep in ("/", "\\")):
        cand = Path("data") / raw
        if cand.exists():
            return cand
    return None


# ======================
# Wizard reader
# ======================
def detect_header_rows(xls_path: Path, sheet) -> Tuple[int, int]:
    """Return (level0_row_idx, level1_row_idx) by scanning first ~15 rows."""
    tmp = pd.read_excel(xls_path, sheet_name=sheet, header=None, nrows=15)
    lvl0 = None
    for i in range(len(tmp)):
        vals = tmp.iloc[i].astype(str).tolist()
        if any(v.strip() in ID_L0_NAMES["location"] for v in vals) and \
           any(v.strip() in ID_L0_NAMES["product_name"] for v in vals):
            lvl0 = i
            break
    if lvl0 is None:
        # fallback: first row that mentions both "Location" and "Product"
        for i in range(len(tmp)):
            vals = tmp.iloc[i].astype(str).tolist()
            if any("Location" in v for v in vals) and any("Product" in v for v in vals):
                lvl0 = i
                break
    if lvl0 is None:
        raise ValueError("Could not detect header rows. Ensure it's a Wizard-style export.")
    return lvl0, lvl0 + 1

def read_wizard(xls_path: Path, sheet) -> pd.DataFrame:
    lvl0, lvl1 = detect_header_rows(xls_path, sheet)
    raw = pd.read_excel(xls_path, sheet_name=sheet, header=None)
    cols_l0 = raw.iloc[lvl0].astype(str).tolist()
    cols_l1 = raw.iloc[lvl1].astype(str).tolist()
    print("[DEBUG] cols_l0:", cols_l0)
    print("[DEBUG] cols_l1:", cols_l1)
    # Check for recursive or malformed objects
    for i, (l0, l1) in enumerate(zip(cols_l0, cols_l1)):
        if isinstance(l0, (list, tuple)) and l0 is l0:
            print(f"[ERROR] Recursive object detected in cols_l0 at index {i}: {l0}")
        if isinstance(l1, (list, tuple)) and l1 is l1:
            print(f"[ERROR] Recursive object detected in cols_l1 at index {i}: {l1}")
    data = raw.iloc[lvl1 + 1:].copy()
    try:
        data.columns = pd.MultiIndex.from_tuples(list(zip(cols_l0, cols_l1)))
    except Exception as e:
        print(f"[ERROR] Failed to create MultiIndex: {e}")
        raise
    data = data.dropna(how="all")
    log(f"Wizard read: {len(data)} rows, {len(data.columns)} columns")
    return data

def find_id_columns(cols: List[Tuple]) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple]]:
    loc_col = name_col = num_col = None
    for c in cols:
        l0, l1 = c
        if str(l0).strip() in ID_L0_NAMES["location"]:
            loc_col = c
        if str(l0).strip() in ID_L0_NAMES["product_name"]:
            name_col = c
    for want0, want1 in ID_L0_NAMES["product_num"]:
        for c in cols:
            if str(c[0]).strip() == want0 and str(c[1]).strip() == want1:
                num_col = c
                break
        if num_col is not None:
            break
    return loc_col, name_col, num_col

def collect_week_cols(cols: List[Tuple], family_l0: List[str]) -> List[Tuple]:
    return [c for c in cols if str(c[0]).strip() in family_l0]

def row_avg_daily(weekly: np.ndarray, ignore_zero: bool, method: str, trim_pct: float) -> float:
    vals = weekly.astype(float)
    if ignore_zero:
        vals = np.where(vals == 0.0, np.nan, vals)
    vals = vals / 7.0
    return avg(vals, method=method, trim_pct=trim_pct)


# ======================
# Core logic
# ======================
def compute_par_levels(
    input_path: Path,
    sheet,
    coverage_days: int,
    safety: float,
    weeks: str,
    metrics: List[str],
    method: str,
    trim_pct: float,
    ignore_zero_weeks: bool,
    box_path: Optional[Path],
    cat_path: Optional[Path],
    cat_safety_path: Optional[Path],
) -> pd.DataFrame:

    df = read_wizard(input_path, sheet)
    cols = list(df.columns)
    loc_col, name_col, num_col = find_id_columns(cols)

    if num_col is not None:
        before = len(df)
        df = df[df[num_col] != "Product #"].copy()
        if len(df) != before:
            log(f"Dropped {before - len(df)} header-ish rows where Product # == 'Product #'.")

    # Metric columns
    fam_to_cols = {}
    for fam in metrics:
        fam_cols = collect_week_cols(cols, METRIC_L0_ALIASES.get(fam, []))
        for c in fam_cols:
            df[c] = df[c].map(to_num)
        if isinstance(weeks, str) and weeks.lower() == "all":
            fam_to_cols[fam] = fam_cols
        else:
            try:
                n = int(weeks)
                fam_to_cols[fam] = fam_cols[:n]
            except Exception:
                fam_to_cols[fam] = fam_cols
        log(f"Metric '{fam}': using {len(fam_to_cols[fam])} week columns.")

    # Base output
    out = pd.DataFrame()
    if loc_col is not None: out["Location"] = df[loc_col].astype(str).values
    if num_col is not None: out["Product #"] = df[num_col].astype(str).str.strip().str.upper().values
    if name_col is not None: out["Product Name"] = df[name_col].astype(str).str.strip().values

    # Category map (by Product #)
    out["Category"] = ""
    if cat_path and cat_path.exists() and "Product #" in out.columns:
        cat_df = pd.read_excel(cat_path)
        if "category" in cat_df.columns:
            cat_df["category"] = cat_df["category"].ffill()
        if "Product #" in cat_df.columns:
            cat_df["Product #"] = cat_df["Product #"].astype(str).str.strip().str.upper()
            cat_map = dict(zip(cat_df["Product #"], cat_df.get("category", "")))
            out["Category"] = out["Product #"].map(cat_map).fillna("")

    # Per-category safety override (optional)
    cat_safety_map: dict[str, float] = {}
    if cat_safety_path and cat_safety_path.exists():
        cs = pd.read_excel(cat_safety_path)
        if {"category", "safety_pct"}.issubset(cs.columns):
            cs["category"] = cs["category"].astype(str).str.strip()  # robustness
            cs["safety_pct"] = pd.to_numeric(cs["safety_pct"], errors="coerce").fillna(0.0)
            cat_safety_map = dict(zip(cs["category"], cs["safety_pct"]))
        else:
            warn("category_safety.xlsx missing columns: category, safety_pct")

    def eff_safety(cat: str) -> float:
        return float(cat_safety_map.get(cat, safety))

    # Compute averages & par per metric family
    created_par_cols = []
    for fam, week_cols in fam_to_cols.items():
        if not week_cols:
            warn(f"No week columns for '{fam}'. Skipping.")
            continue

        weekly = df[week_cols].to_numpy(dtype=float)
        avg_daily = np.apply_along_axis(
            row_avg_daily, 1, weekly,
            ignore_zero=ignore_zero_weeks,
            method=method,
            trim_pct=trim_pct
        )
        out[f"AvgDaily_{fam}"] = avg_daily

        saf = out["Category"].apply(eff_safety) if "Category" in out.columns else pd.Series([safety]*len(out))
        par_units = np.ceil(avg_daily * coverage_days * (1.0 + saf.values.astype(float)))
        par_col = f"Par{coverage_days}d_{fam}"
        out[par_col] = par_units
        created_par_cols.append(par_col)

    # ---------- BOXES by Product # (exact columns requested) ----------
    pcs_in_box_ser = None
    if box_path and box_path.exists() and "Product #" in out.columns:
        bdf = pd.read_excel(box_path)
        required = {"Product #", "Product Name", "Pcs in Box"}
        if required.issubset(bdf.columns):
            bdf["Product #"] = bdf["Product #"].astype(str).str.strip().str.upper()
            bdf["Pcs in Box"] = pd.to_numeric(bdf["Pcs in Box"], errors="coerce")
            pcs_in_box_ser = out["Product #"].map(dict(zip(bdf["Product #"], bdf["Pcs in Box"])))
        else:
            warn("BoxQuantities.xlsx must have columns: Product #, Product Name, Pcs in Box")

    def ceil_div_series(numer: pd.Series, denom: pd.Series) -> pd.Series:
        num = pd.to_numeric(numer, errors="coerce")
        den = pd.to_numeric(denom, errors="coerce")
        out_ser = pd.Series(np.nan, index=num.index, dtype="float")
        # If denominator is >0 and numerator is not NaN, do ceil division
        mask = den.gt(0) & num.notna()
        out_ser.loc[mask] = np.ceil(num.loc[mask] / den.loc[mask])
        # If denominator is zero, set result to zero
        zero_mask = den.eq(0)
        out_ser.loc[zero_mask] = 0.0
        return out_ser

    if pcs_in_box_ser is not None:
        pcs_in_box_ser = pcs_in_box_ser.reindex(out.index)
        # EXACT names requested (coverage_days defaults to 10)
        if "Par10d_actual_qty" in out.columns:
            out["Par10d_actual_qty_boxes"] = ceil_div_series(out["Par10d_actual_qty"], pcs_in_box_ser)
        if "Par10d_theo_qty" in out.columns:
            out["Par10d_theo_qty_boxes"] = ceil_div_series(out["Par10d_theo_qty"], pcs_in_box_ser)

    # Drop rows where all Par columns are zero/NaN
    if created_par_cols:
        keep = ~(out[created_par_cols].fillna(0).eq(0).all(axis=1))
        out = out[keep]
    else:
        warn("No Par columns created — check metric aliases vs your file headers.")

    # Order & sort
    leading = [c for c in ["Location", "Category", "Product #", "Product Name"] if c in out.columns]
    rest = [c for c in out.columns if c not in leading]
    out = out[leading + rest]
    if "Location" in out.columns and "Product Name" in out.columns:
        out = out.sort_values(["Location", "Product Name"], kind="stable")
    elif "Location" in out.columns and "Product #" in out.columns:
        out = out.sort_values(["Location", "Product #"], kind="stable")

    return out.reset_index(drop=True)


# ======================
# IO
# ======================
def save_output(df: pd.DataFrame, out_path: Path, coverage_days: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if str(out_path).lower().endswith(".csv"):
        df.to_csv(out_path, index=False)
    else:
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            df.to_excel(w, sheet_name=f"Par_Levels_{coverage_days}d", index=False)
    log(f"Saved: {out_path.resolve()}")


# ======================
# CLI
# ======================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT, help="Wizard Excel path")
    p.add_argument("--sheet", default=DEFAULT_SHEET, help="Sheet index (int) or name (str)")
    p.add_argument("--coverage_days", type=int, default=DEFAULT_COVERAGE_DAYS)
    p.add_argument("--safety", type=float, default=DEFAULT_SAFETY, help="Global safety (e.g., 0.2 for +20%)")
    p.add_argument("--weeks", default=DEFAULT_WEEKS, help='Most recent N weeks by column order, or "all"')
    p.add_argument("--metrics", default=DEFAULT_METRICS,
                   help="Comma list: actual_qty,theo_qty,actual_value,theo_value")
    p.add_argument("--method", default=DEFAULT_METHOD, choices=["mean", "median", "trimmed"])
    p.add_argument("--trim_pct", type=float, default=DEFAULT_TRIM_PCT)
    p.add_argument("--ignore_zero_weeks", action="store_true", default=DEFAULT_IGNORE_ZERO_WEEKS)
    p.add_argument("--box_path", default=DEFAULT_BOX_PATH,
                   help='Excel with ["Product #","Product Name","Pcs in Box"] (matched by Product #)')
    p.add_argument("--cat_path", default=DEFAULT_CAT_PATH,
                   help='Excel with ["category","Product #","Product Name"] (matched by Product #)')
    p.add_argument("--cat_safety_path", default=DEFAULT_CAT_SAFETY_PATH,
                   help='Excel with ["category","safety_pct"] (optional)')
    p.add_argument("--out", default=DEFAULT_OUT, help="Output path (.xlsx or .csv)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    inp = resolve_required_path(args.input)
    box_p = resolve_optional_path(args.box_path)
    cat_p = resolve_optional_path(args.cat_path)
    cat_safe_p = resolve_optional_path(args.cat_safety_path)

    # Sheet may be int or str
    sheet = args.sheet
    try:
        sheet = int(sheet)
    except Exception:
        pass

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    log(f"Input: {inp}")
    if box_p: log(f"Box file: {box_p}")
    if cat_p: log(f"Category file: {cat_p}")
    if cat_safe_p: log(f"Category-safety file: {cat_safe_p}")
    log(f"Coverage={args.coverage_days}, Safety={args.safety}, Weeks={args.weeks}, Method={args.method}, IgnoreZeros={args.ignore_zero_weeks}")

    df = compute_par_levels(
        input_path=inp,
        sheet=sheet,
        coverage_days=args.coverage_days,
        safety=args.safety,
        weeks=args.weeks,
        metrics=metrics,
        method=args.method,
        trim_pct=args.trim_pct,
        ignore_zero_weeks=args.ignore_zero_weeks,
        box_path=box_p,
        cat_path=cat_p,
        cat_safety_path=cat_safe_p,
    )

    save_output(df, Path(args.out), args.coverage_days)


if __name__ == "__main__":
    main()
