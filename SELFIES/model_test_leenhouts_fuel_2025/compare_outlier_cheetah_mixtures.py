#!/usr/bin/env python3
"""Match Leenhouts 2025 outliers to nearest Cheetah mixtures and write a paired report."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DEFAULT_OUTLIERS = _HERE / "evaluation_results" / "leenhouts_2025_outliers.dat"
DEFAULT_CN_MIX = _HERE.parent / "perturbation_testing" / "cn_mixture_selfies.dat"
DEFAULT_OUTPUT = _HERE / "evaluation_results" / "leenhouts_2025_outlier_cheetah_comparison.dat"
N_COMPONENTS = 10
VOL_TOL_CLOSE = 0.05


def load_dat(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    lines = [ln.rstrip("\n") for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    header = lines[0].split("\t")
    rows = [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]
    return header, rows


def is_active(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    return bool(s) and s.lower() != "nan"


def fnum(value: str | float | None, default: float = math.nan) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def extract_components(row: dict[str, str]) -> list[tuple[str, float]]:
    vol_by_selfies: dict[str, float] = {}
    for i in range(1, N_COMPONENTS + 1):
        selfies = row.get(f"cpnt_selfies_{i}", "")
        vol = row.get(f"cpnt_vol_{i}", "0")
        if not is_active(selfies):
            continue
        v = fnum(vol, 0.0)
        if v <= 0 or math.isnan(v):
            continue
        sf = selfies.strip()
        vol_by_selfies[sf] = vol_by_selfies.get(sf, 0.0) + v
    total = sum(vol_by_selfies.values()) or 1.0
    comps = [(sf, v / total) for sf, v in vol_by_selfies.items()]
    comps.sort(key=lambda x: x[0])
    return comps


def vol_l1(c1: list[tuple[str, float]], c2: list[tuple[str, float]]) -> float:
    """Sum of |Δvol_fraction| over all components (union of both mixtures)."""
    d1 = dict(c1)
    d2 = dict(c2)
    keys = sorted(set(d1) | set(d2))
    return sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)


def vol_deltas(
    leenhouts: list[tuple[str, float]],
    cheetah: list[tuple[str, float]],
    ordered_selfies: list[str],
) -> list[float]:
    d_ln = dict(leenhouts)
    d_ch = dict(cheetah)
    return [100.0 * (d_ch.get(sf, 0.0) - d_ln.get(sf, 0.0)) for sf in ordered_selfies]


def format_delta(delta_pct: float) -> str:
    if abs(delta_pct) < 0.005:
        return "0.00"
    return f"{delta_pct:+.2f}"


def jaccard_selfies(c1: list[tuple[str, float]], c2: list[tuple[str, float]]) -> float:
    s1 = {sf for sf, _ in c1}
    s2 = {sf for sf, _ in c2}
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def build_selfies_name_map(cn_rows: list[dict[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}

    def register(sf: str, name: str, priority: int) -> None:
        current = mapping.get(sf)
        if current is None:
            mapping[sf] = name
            return
        # Prefer explicit chemical names from pure-component rows.
        if priority == 2 and current in {"PRF 100", "PRF 95", "PRF 90", "PRF 80", "PRF 75", "PRF 70", "PRF 60", "PRF 50", "PRF 40", "PRF 30", "PRF 20", "PRF 10", "PRF 0"}:
            mapping[sf] = name

    for row in cn_rows:
        name = row.get("mixture_name", "").strip()
        if not name or name.lower() == "nan":
            continue
        comps = extract_components(row)
        if row.get("mixture_type", "").strip().lower() == "pure component":
            for sf, _ in comps:
                register(sf, name, priority=2)
            continue
        if len(comps) == 1:
            register(comps[0][0], name, priority=1)
    return mapping


def selfies_to_name(selfies: str, cache: dict[str, str], name_map: dict[str, str]) -> str:
    if selfies in cache:
        return cache[selfies]
    if selfies in name_map:
        cache[selfies] = name_map[selfies]
        return cache[selfies]
    try:
        import selfies as sf
        from rdkit import Chem

        smiles = sf.decoder(selfies)
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is not None:
            name = Chem.MolToSmiles(mol)
            cache[selfies] = name
            return name
    except Exception:
        pass
    short = selfies if len(selfies) <= 48 else selfies[:45] + "..."
    cache[selfies] = short
    return short


def find_best_match(
    ocomps: list[tuple[str, float]],
    cn_records: list[dict],
) -> tuple[dict, float, str]:
    oset = tuple(sf for sf, _ in ocomps)
    same_set = [rec for rec in cn_records if tuple(sf for sf, _ in rec["comps"]) == oset]
    if same_set:
        best = min(same_set, key=lambda rec: vol_l1(ocomps, rec["comps"]))
        l1 = vol_l1(ocomps, best["comps"])
        if l1 <= VOL_TOL_CLOSE:
            match_type = "close"
        else:
            match_type = "same_components"
        return best, l1, match_type

    best_rec = None
    best_score = (-1.0, -999.0)
    for rec in cn_records:
        score = (jaccard_selfies(ocomps, rec["comps"]), -vol_l1(ocomps, rec["comps"]))
        if score > best_score:
            best_score = score
            best_rec = rec
    assert best_rec is not None
    return best_rec, vol_l1(ocomps, best_rec["comps"]), "partial"


def format_vol(vol: float) -> str:
    return f"{100.0 * vol:.2f}"


def write_report(
    outliers: list[dict[str, str]],
    cn_rows: list[dict[str, str]],
    output: Path,
) -> None:
    name_map = build_selfies_name_map(cn_rows)
    name_cache: dict[str, str] = {}

    cn_records = []
    for row in cn_rows:
        cn_records.append(
            {
                "cn_mxtr_selfies_id": row.get("cn_mxtr_selfies_id", ""),
                "mixture_id": row.get("mixture_id", ""),
                "mixture_name": row.get("mixture_name", ""),
                "CN": fnum(row.get("CN", "nan")),
                "comps": extract_components(row),
            }
        )

    max_comp = max(len(extract_components(row)) for row in outliers)
    comp_cols: list[str] = []
    for i in range(1, max_comp + 1):
        comp_cols.extend([
            f"component_{i}_name",
            f"component_{i}_vol_pct_leenhouts",
            f"component_{i}_vol_pct_cheetah",
            f"component_{i}_vol_delta_pct",
        ])

    header = [
        "pair_no",
        "source",
        "record_id",
        "mixture_id",
        "mixture_name",
        "CN",
        "cn_predicted",
        "cn_abs_error",
        "sum_abs_vol_delta_pct",
        "match_type",
        *comp_cols,
    ]

    outlier_pairs: list[tuple[dict, dict, float, str]] = []
    for orow in outliers:
        ocomps = extract_components(orow)
        best, l1, match_type = find_best_match(ocomps, cn_records)
        outlier_pairs.append((orow, best, l1, match_type))

    outlier_pairs.sort(key=lambda x: -fnum(x[0].get("cn_abs_error", "nan")))

    lines: list[str] = [
        "# Leenhouts 2025 outlier vs nearest Cheetah mixture comparison",
        "# One pair = two consecutive rows (Leenhouts, then Cheetah).",
        "# Volume fractions are normalized and shown as vol% (0-100).",
        "# vol_delta_pct = Cheetah vol% minus Leenhouts vol% (shown on Cheetah row only).",
        "# sum_abs_vol_delta_pct = sum of |vol_delta_pct| over components (total composition gap in percentage points).",
        f"# Close match: same component SELFIES set and sum_abs_vol_delta_pct <= {100.0 * VOL_TOL_CLOSE:.1f}.",
        f"# Pairs: {len(outlier_pairs)}",
        "",
        "\t".join(header),
    ]

    for pair_no, (orow, crec, l1, match_type) in enumerate(outlier_pairs, start=1):
        ocomps = extract_components(orow)
        ccomps = crec["comps"]
        selfies_order = [sf for sf, _ in ocomps]
        cheetah_order = selfies_order if match_type != "partial" else [sf for sf, _ in ccomps]
        deltas = vol_deltas(ocomps, ccomps, selfies_order if match_type != "partial" else cheetah_order)
        sum_abs_delta = sum(abs(d) for d in deltas)

        def comp_fields_ln(ordered_selfies: list[str]) -> list[str]:
            comp_dict = dict(ocomps)
            fields: list[str] = []
            for sf in ordered_selfies:
                fields.append(selfies_to_name(sf, name_cache, name_map))
                fields.append(format_vol(comp_dict.get(sf, 0.0)))
                fields.append("")
                fields.append("")
            for _ in range(max_comp - len(ordered_selfies)):
                fields.extend(["", "", "", ""])
            return fields

        def comp_fields_ch(ordered_selfies: list[str], row_deltas: list[float]) -> list[str]:
            comp_dict = dict(ccomps)
            ln_dict = dict(ocomps)
            fields: list[str] = []
            for i, sf in enumerate(ordered_selfies):
                fields.append(selfies_to_name(sf, name_cache, name_map))
                if match_type != "partial":
                    fields.append(format_vol(ln_dict.get(sf, 0.0)))
                else:
                    fields.append("")
                fields.append(format_vol(comp_dict.get(sf, 0.0)))
                fields.append(format_delta(row_deltas[i]) if i < len(row_deltas) else "")
            for _ in range(max_comp - len(ordered_selfies)):
                fields.extend(["", "", "", ""])
            return fields

        leenhouts_row = [
            str(pair_no),
            "Leenhouts",
            orow.get("dcn_trn_selfies_id", ""),
            orow.get("mixture_id", ""),
            orow.get("mixture_name", "") if is_active(orow.get("mixture_name")) else "",
            f"{fnum(orow.get('cn_actual', orow.get('DCN', 'nan'))):.4f}",
            f"{fnum(orow.get('cn_predicted', orow.get('predicted_CN', 'nan'))):.4f}",
            f"{fnum(orow.get('cn_abs_error', 'nan')):.4f}",
            "",
            match_type,
            *comp_fields_ln(selfies_order),
        ]
        cheetah_row = [
            str(pair_no),
            "Cheetah",
            crec["cn_mxtr_selfies_id"],
            crec["mixture_id"],
            crec["mixture_name"],
            f"{crec['CN']:.4f}",
            "",
            "",
            f"{sum_abs_delta:.2f}",
            match_type,
            *comp_fields_ch(cheetah_order, deltas),
        ]
        lines.append("\t".join(leenhouts_row))
        lines.append("\t".join(cheetah_row))
        lines.append("")

    output.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {len(outlier_pairs)} pairs ({len(outlier_pairs) * 2} rows) to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Leenhouts outliers to Cheetah mixtures.")
    parser.add_argument("--outliers", type=Path, default=DEFAULT_OUTLIERS)
    parser.add_argument("--cn-mix", type=Path, default=DEFAULT_CN_MIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    _, outliers = load_dat(args.outliers)
    _, cn_rows = load_dat(args.cn_mix)
    write_report(outliers, cn_rows, args.output)


if __name__ == "__main__":
    main()
