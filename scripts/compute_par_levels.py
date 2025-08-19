#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_par_levels.py — Wizard Weekly -> 10-Day Par (flat headers, robust categories, Product# boxes)

• Flat, sanitized headers (no MultiIndex) to avoid Pandas recursion errors.
• Weekly -> daily (÷7), average across N weeks (mean/median/trimmed), optional ignore-zero-weeks.
• Par = ceil(avg_daily × coverage_days × (1 + safety)).
• Category mapping by Product # with robust, case-insensitive header handling.
• Per-category safety (optional).
• Boxes by Product #:
    Par10d_actual_qty_boxes = ceil(Par10d_actual_qty / Pcs in Box)
    Par10d_theo_qty_boxes   = ceil(Par10d_theo_qty   / Pcs in Box)

Defaults expect files under ./data/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# ======================
# Defaults
# ======================
DEFAULT_INPUT = "./data/Daily_Consumption.xlsx"
DEFAULT_BOX_PATH = "./data/BoxQuantities.xlsx"          # Product #, Product Name, Pcs in Box
DEFAULT_CAT_PATH = "./data/product_categories.xlsx"     # (category|dept|group...), (product #|product#|sku|item #|product number)
DEFAULT_CAT_SAFETY_PATH = "./data/category_safety.xlsx" # category, safety_pct (optional)
DEFAULT_OUT = "./out/par_levels_10d.xlsx"

DEFAULT_SHEET = 0
DEFAULT_COVERAGE_DAYS = 10
DEFAULT_SAFETY = 0.00
DEFAULT_WEEKS = "all"                  # "all" or integer like "8"
DEFAULT_METRICS = "actual_qty,theo_qty"
DEFAULT_METHOD = "mean"                # mean | median | trimmed
DEFAULT_TRIM_PCT = 0.10
DEFAULT_IGNORE_ZERO_WEEKS = False

# ======================
# Aliases / Matching
# ======================
ID_L0_NAMES = {
    "location": ["Location Name", "Location"],
    "product_name": ["Product Name", "Item Name"],
    # Product # appears as the 2nd header row under group "Metrics"
    "product_num_pair": ("Metrics", "Product #"),
    "product_num_pair_alt": [("Metrics", "Product#"), ("Metrics", "Product Number")],
}

METRIC_L0_ALIASES = {
    "actual_qty": ["Act. Consumpt. Qty", "Actual Consumpt. Qty", "Actual Consumption Qty", "Actual Qty", "Consumpt. Qty"],
    "theo_qty":   ["Theoretical Qty", "Theo Qty", "Theoretical Consumption Qty"],
    "actual_value": ["Act. Consumpt. Value (L)", "Act. Consumpt. Value", "Actual Consumption Value", "Actual Value"],
    "theo_value":   ["Theoretical Value (L)", "Theoretical Value"],
}

# accepted headers (case/space-insensitive) for the categories file
CAT_NAME_KEYS = ["category", "categories", "dept", "department", "group"]
CAT_PRODNUM_KEYS = ["product #", "product#", "product number", "sku", "item #", "item#", "item number"]

SEP = "||"  # flatten separator

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

def avg(values: np.ndarray, method: str, trim_pct: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    if method == "median":
        return float(np.median(arr))
    if method == "trimmed":
        trim_pct = 0.1 if not (0 <= trim_pct < 0.5) else trim_pct
        arr = np.sort(arr)
        n = len(arr)
        k = int(np.floor(n * trim_pct))
        if k * 2 >= n:
            return float(arr.mean())
        return float(arr[k:n - k].mean())
    return float(arr.mean())

def resolve_required_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
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
# Wizard header handling (flat)
# ======================
def detect_header_rows(xls_path: Path, sheet) -> Tuple[int, int]:
    """Return (level0_row_idx, level1_row_idx) by scanning first ~20 rows."""
    tmp = pd.read_excel(xls_path, sheet_name=sheet, header=None, nrows=20)
    lvl0 = None
    for i in range(len(tmp)):
        vals = tmp.iloc[i].astype(str).tolist()
        if any(v.strip() in ID_L0_NAMES["location"] for v in vals) and \
           any(v.strip() in ID_L0_NAMES["product_name"] for v in vals):
            lvl0 = i
            break
    if lvl0 is None:
        for i in range(len(tmp)):
            vals = tmp.iloc[i].astype(str).tolist()
            if any("Location" in v for v in vals) and any("Product" in v for v in vals):
                lvl0 = i
                break
    if lvl0 is None:
        raise ValueError("Could not detect header rows. Ensure it's a Wizard-style export.")
    return lvl0, lvl0 + 1

def _coerce_str_list(seq) -> List[str]:
    out = []
    for v in seq:
        if v is None:
            out.append("")
            continue
        try:
            if isinstance(v, float) and np.isnan(v):
                out.append("")
            else:
                out.append(str(v))
        except Exception:
            out.append("")
    return out

def _ffill_empty(values: List[str]) -> List[str]:
    last = ""
    out = []
    for s in values:
        ss = s.strip()
        if ss == "":
            out.append(last)
        else:
            out.append(ss)
            last = ss
    return out

def read_wizard_flat(xls_path: Path, sheet) -> pd.DataFrame:
    """Read 2 header rows, sanitize, and flatten as '<l0>||<l1>' columns."""
    lvl0, lvl1 = detect_header_rows(xls_path, sheet)
    raw = pd.read_excel(xls_path, sheet_name=sheet, header=None)

    l0 = _coerce_str_list(raw.iloc[lvl0].tolist())
    l1 = _coerce_str_list(raw.iloc[lvl1].tolist())

    # trim to same length
    n = min(len(l0), len(l1))
    l0 = l0[:n]
    l1 = l1[:n]

    # forward-fill l0, fill empty l1 with placeholder
    l0 = _ffill_empty(l0)
    l1 = [s if s.strip() != "" else f"col_{i}" for i, s in enumerate(l1)]

    flat_cols = [f"{l0[i]}{SEP}{l1[i]}" for i in range(n)]
    data = raw.iloc[lvl1 + 1:, :n].copy()
    data.columns = flat_cols
    data = data.dropna(how="all")
    log(f"Wizard read: {len(data)} rows, {len(data.columns)} columns")
    return data

# ======================
# Column detection (flat)
# ======================
def find_id_columns_flat(cols: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return column keys for Location, Product Name, Product # in FLAT header form."""
    loc_col = name_col = num_col = None
    for c in cols:
        l0 = c.split(SEP, 1)[0].strip()
        if l0 in ID_L0_NAMES["location"]:
            loc_col = c
        if l0 in ID_L0_NAMES["product_name"]:
            name_col = c

    # product #: look for ("Metrics","Product #") or alternatives
    targets = [ID_L0_NAMES["product_num_pair"]] + ID_L0_NAMES["product_num_pair_alt"]
    for want0, want1 in targets:
        key = f"{want0}{SEP}{want1}"
        if key in cols:
            num_col = key
            break
    return loc_col, name_col, num_col

def collect_week_cols_flat(cols: List[str], family_l0: List[str]) -> List[str]:
    return [c for c in cols if c.split(SEP, 1)[0].strip() in family_l0]

# ======================
# Category helpers
# ======================
def _normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        if isinstance(c, str):
            new_cols[c] = " ".join(c.strip().split()).lower()
        else:
            new_cols[str(c)] = str(c).strip().lower()
    return df.rename(columns=new_cols)

def load_category_map(cat_path: Path) -> dict[str, str]:
    """Loads product_categories.xlsx robustly and returns {PRODUCT#: category}."""
    try:
        cdf = pd.read_excel(cat_path)  # first sheet by default
    except Exception as e:
        warn(f"Could not read categories file: {e}")
        return {}

    cdf = _normalize_colnames(cdf)

    # find category column
    cat_col = None
    for k in CAT_NAME_KEYS:
        if k in cdf.columns:
            cat_col = k
            break
    if cat_col is None:
        warn(f"Could not find a category column among: {CAT_NAME_KEYS}")
        return {}

    # find product number column
    pnum_col = None
    for k in CAT_PRODNUM_KEYS:
        if k in cdf.columns:
            pnum_col = k
            break
    if pnum_col is None:
        warn(f"Could not find a product number column among: {CAT_PRODNUM_KEYS}")
        return {}

    # normalize values
    cdf[cat_col] = cdf[cat_col].astype(str).str.strip()
    cdf[pnum_col] = cdf[pnum_col].astype(str).str.strip().str.upper()

    # drop rows where product # is empty
    cdf = cdf[cdf[pnum_col] != ""].copy()

    # forward-fill categories if provided like section headers
    cdf[cat_col] = cdf[cat_col].replace({"nan": ""})
    cdf[cat_col] = cdf[cat_col].replace("", np.nan).ffill().fillna("")

    # build mapping; if duplicates, keep the last non-empty category
    cdf = cdf.dropna(subset=[pnum_col])
    cdf = cdf[[pnum_col, cat_col]].drop_duplicates(subset=[pnum_col], keep="last")
    mapping = dict(zip(cdf[pnum_col], cdf[cat_col]))
    log(f"Loaded {len(mapping)} category mappings from {cat_path.name}.")
    return mapping

# ======================
# Core computation
# ======================
def row_avg_daily(weekly: np.ndarray, ignore_zero: bool, method: str, trim_pct: float) -> float:
    vals = weekly.astype(float)
    if ignore_zero:
        vals = np.where(vals == 0.0, np.nan, vals)
    vals = vals / 7.0
    return avg(vals, method=method, trim_pct=trim_pct)

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

    df = read_wizard_flat(input_path, sheet)
    cols = list(df.columns)
    loc_col, name_col, num_col = find_id_columns_flat(cols)

    if num_col is not None and num_col in df.columns:
        before = len(df)
        df = df[df[num_col].astype(str) != "Product #"].copy()
        if len(df) != before:
            log(f"Dropped {before - len(df)} header-ish rows where Product # == 'Product #'.")

    # Metrics
    fam_to_cols: dict[str, List[str]] = {}
    for fam in metrics:
        fam_cols = collect_week_cols_flat(cols, METRIC_L0_ALIASES.get(fam, []))
        for c in fam_cols:
            df[c] = df[c].map(to_num)
        if isinstance(weeks, str) and weeks.lower() == "all":
            fam_to_cols[fam] = fam_cols
        else:
            try:
                n = int(weeks)
                fam_to_cols[fam] = fam_cols[:n]  # wizard: left->right most-recent
            except Exception:
                fam_to_cols[fam] = fam_cols
        log(f"Metric '{fam}': using {len(fam_to_cols[fam])} week columns.")

    # Base output
    out = pd.DataFrame()
    if loc_col is not None: out["Location"] = df[loc_col].astype(str).values
    if num_col is not None: out["Product #"] = df[num_col].astype(str).str.strip().str.upper().values
    if name_col is not None: out["Product Name"] = df[name_col].astype(str).str.strip().values

    # Categories by Product #
    out["Category"] = ""
    mapped_count = 0
    if cat_path and cat_path.exists() and "Product #" in out.columns:
        cat_map = load_category_map(cat_path)
        if cat_map:
            out["Category"] = out["Product #"].map(cat_map).fillna("")
            mapped_count = (out["Category"] != "").sum()
    log(f"Category mapped rows: {mapped_count} / {len(out)}")

    # Optional per-category safety
    cat_safety_map: dict[str, float] = {}
    if cat_safety_path and cat_safety_path.exists():
        cs = pd.read_excel(cat_safety_path)
        cs = _normalize_colnames(cs)
        if {"category", "safety_pct"}.issubset(cs.columns):
            cs["category"] = cs["category"].astype(str).str.strip()
            cs["safety_pct"] = pd.to_numeric(cs["safety_pct"], errors="coerce").fillna(0.0)
            cat_safety_map = dict(zip(cs["category"], cs["safety_pct"]))
            log(f"Per-category safety rows: {len(cat_safety_map)}")
        else:
            warn("category_safety.xlsx missing columns: category, safety_pct")

    def eff_safety(cat: str) -> float:
        return float(cat_safety_map.get(cat, safety))

    # AvgDaily + Par per metric family
    created_par_cols: List[str] = []
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

        saf_series = out["Category"].apply(eff_safety) if "Category" in out.columns else pd.Series([safety]*len(out))
        par_units = np.ceil(avg_daily * coverage_days * (1.0 + saf_series.values.astype(float)))
        par_col = f"Par{coverage_days}d_{fam}"
        out[par_col] = par_units
        created_par_cols.append(par_col)

    # Get pcs_in_box from product_categories.xlsx
    pcs_in_box_ser = None
    if cat_path and cat_path.exists() and "Product #" in out.columns:
        cdf = pd.read_excel(cat_path)
        cdf = _normalize_colnames(cdf)
        bx_pnum = None
        for k in CAT_PRODNUM_KEYS + ["product#", "product #", "product number"]:
            if k in cdf.columns:
                bx_pnum = k
                break
        pcs_col = None
        for k in ["pcs in box", "pieces in box", "units per box", "pack size", "case size"]:
            if k in cdf.columns:
                pcs_col = k
                break
        if bx_pnum and pcs_col:
            cdf[bx_pnum] = cdf[bx_pnum].astype(str).str.strip().str.upper()
            cdf[pcs_col] = pd.to_numeric(cdf[pcs_col], errors="coerce")
            pcs_in_box_ser = out["Product #"].map(dict(zip(cdf[bx_pnum], cdf[pcs_col])))
            log(f"Box mapping rows matched: {(pcs_in_box_ser.notna() & pcs_in_box_ser.gt(0)).sum()} / {len(out)}")
        else:
            warn("product_categories.xlsx needs columns for product # and pcs-in-box (any common naming).")

    if pcs_in_box_ser is not None:
        pcs_in_box_ser = pcs_in_box_ser.reindex(out.index)
        # Actual Box Qty
        if f"Par{coverage_days}d_actual_qty" in out.columns:
            out["Actual Box Qty"] = (out[f"Par{coverage_days}d_actual_qty"] / pcs_in_box_ser).round(2)
        # Theo Box Qty
        if f"Par{coverage_days}d_theo_qty" in out.columns:
            out["Theo Box Qty"] = (out[f"Par{coverage_days}d_theo_qty"] / pcs_in_box_ser).round(2)

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

    # Final mapping stats
    if "Category" in out.columns:
        log(f"Final rows with Category: {(out['Category']!='').sum()} / {len(out)}")

    return out.reset_index(drop=True)

# ======================
# Save
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
                   help='Excel with Product # and Pcs in Box (column names are flexible)')
    p.add_argument("--cat_path", default=DEFAULT_CAT_PATH,
                   help='Excel with Category and Product # (column names are flexible)')
    p.add_argument("--cat_safety_path", default=DEFAULT_CAT_SAFETY_PATH,
                   help='Excel with ["category","safety_pct"] (optional)')
    p.add_argument("--out", default=DEFAULT_OUT, help="Output path (.xlsx or .csv)")
    return p.parse_args()


# --- Hardcoded parameters for your workflow ---
from pathlib import Path

DAILY_CONSUMPTION_PATH = Path("data/Daily_Consumption.xlsx")
PRODUCT_CATEGORIES_PATH = Path("data/product_categories.xlsx")
SHEET_NAME = 0  # or use the actual sheet name if needed
COVERAGE_DAYS = 10
SAFETY_FACTOR = 0.2  # 20% safety stock
WEEKS = 5  # Use 5 most recent weeks
METRICS = ["actual_qty", "theo_qty"]
METHOD = "mean"
TRIM_PCT = 0.0
IGNORE_ZERO_WEEKS = True
OUTPUT_PATH = Path("C:/Projects/Par_Level_updates/out/Par_Levels_Output.xlsx")

# --- Run computation ---
df = compute_par_levels(
    input_path=DAILY_CONSUMPTION_PATH,
    sheet=SHEET_NAME,
    coverage_days=COVERAGE_DAYS,
    safety=SAFETY_FACTOR,
    weeks=WEEKS,
    metrics=METRICS,
    method=METHOD,
    trim_pct=TRIM_PCT,
    ignore_zero_weeks=IGNORE_ZERO_WEEKS,
    box_path=None,
    cat_path=PRODUCT_CATEGORIES_PATH,
    cat_safety_path=None,
)

save_output(df, OUTPUT_PATH, COVERAGE_DAYS)
