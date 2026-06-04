"""
Convert InChI and/or SMILES strings to SELFIES (SELF-referencing Embedded Strings).

Uses RDKit to parse structures and the `selfies` package to encode them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import selfies as sf
from rdkit import Chem


class MoleculeConversionError(ValueError):
    """Raised when a structure cannot be parsed or encoded as SELFIES."""


def _mol_to_selfies(mol: Chem.Mol) -> str:
    smiles = Chem.MolToSmiles(mol)
    if not smiles:
        raise MoleculeConversionError("RDKit produced an empty SMILES string")
    try:
        return sf.encoder(smiles)
    except Exception as exc:
        raise MoleculeConversionError(f"SELFIES encoding failed: {exc}") from exc


def inchi_to_selfies(inchi: str) -> str:
    """Convert an InChI identifier to a SELFIES string."""
    inchi = inchi.strip()
    if not inchi:
        raise MoleculeConversionError("Empty InChI string")
    if not inchi.startswith("InChI="):
        raise MoleculeConversionError("InChI must start with 'InChI='")

    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        raise MoleculeConversionError("RDKit could not parse InChI")
    return _mol_to_selfies(mol)


def smiles_to_selfies(smiles: str) -> str:
    """Convert a SMILES string to a SELFIES string."""
    smiles = smiles.strip()
    if not smiles:
        raise MoleculeConversionError("Empty SMILES string")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise MoleculeConversionError("RDKit could not parse SMILES")
    return _mol_to_selfies(mol)


def identifier_to_selfies(identifier: str) -> str:
    """Convert InChI or SMILES to SELFIES (auto-detected from the prefix)."""
    identifier = identifier.strip()
    if not identifier or identifier.lower() in {"nan", "none", "null"}:
        raise MoleculeConversionError("Empty or missing identifier")

    if identifier.startswith("InChI="):
        return inchi_to_selfies(identifier)
    return smiles_to_selfies(identifier)


def _is_blank(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    text = str(value).strip()
    return not text or text.lower() in {"nan", "none", "null"}


def convert_series(
    values: Iterable,
    *,
    skip_errors: bool = False,
) -> list[str | None]:
    """Convert a sequence of InChI/SMILES values to SELFIES."""
    results: list[str | None] = []
    for value in values:
        if _is_blank(value):
            results.append(None)
            continue
        try:
            results.append(identifier_to_selfies(str(value)))
        except MoleculeConversionError:
            if skip_errors:
                results.append(None)
            else:
                raise
    return results


def _detect_separator(path: Path) -> str:
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    return "\t" if "\t" in first_line else ","


def _inchi_columns(columns: Iterable[str]) -> list[str]:
    return [col for col in columns if "inchi" in col.lower()]


def convert_table(
    df: pd.DataFrame,
    columns: list[str],
    *,
    suffix: str = "_selfies",
    skip_errors: bool = False,
) -> pd.DataFrame:
    """Add SELFIES columns alongside the selected InChI/SMILES columns."""
    out = df.copy()
    for col in columns:
        if col not in df.columns:
            raise MoleculeConversionError(f"Column not found: {col}")
        out_col = f"{col}{suffix}" if suffix else col
        out[out_col] = convert_series(df[col], skip_errors=skip_errors)
    return out


def convert_text_file(
    path: Path,
    *,
    skip_errors: bool = False,
) -> list[str]:
    """Read one InChI/SMILES per line and return SELFIES strings."""
    lines = path.read_text(encoding="utf-8").splitlines()
    selfies_list: list[str] = []
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            selfies_list.append(identifier_to_selfies(line))
        except MoleculeConversionError as exc:
            if skip_errors:
                continue
            raise MoleculeConversionError(f"Line {line_no}: {exc}") from exc
    return selfies_list


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert InChI and/or SMILES to SELFIES.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--inchi", type=str, help="Single InChI string")
    group.add_argument("--smiles", type=str, help="Single SMILES string")
    group.add_argument(
        "--identifier",
        type=str,
        help="Single InChI or SMILES (auto-detected)",
    )

    parser.add_argument(
        "--input",
        type=Path,
        help="Input file: TSV/CSV table or plain text (one structure per line)",
    )
    parser.add_argument(
        "--column",
        action="append",
        dest="columns",
        metavar="NAME",
        help="Table column to convert (repeatable)",
    )
    parser.add_argument(
        "--all-inchi-columns",
        action="store_true",
        help="Convert every column whose name contains 'inchi'",
    )
    parser.add_argument(
        "--suffix",
        default="_selfies",
        help="Suffix for new output columns in table mode (default: _selfies)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write results to this file (default: stdout)",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="Field separator for table input/output (default: tab if present)",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip invalid structures in batch mode (writes empty cells)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.inchi:
            result = inchi_to_selfies(args.inchi)
            _write_output(result, args.output)
            return 0

        if args.smiles:
            result = smiles_to_selfies(args.smiles)
            _write_output(result, args.output)
            return 0

        if args.identifier:
            result = identifier_to_selfies(args.identifier)
            _write_output(result, args.output)
            return 0

        if args.input:
            return _run_batch(args)

        parser.error(
            "Provide --inchi, --smiles, --identifier, or --input. "
            "Use -h for help."
        )
    except MoleculeConversionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _write_output(text: str, path: Path | None) -> None:
    if path:
        path.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _run_batch(args: argparse.Namespace) -> int:
    path: Path = args.input
    if not path.exists():
        raise MoleculeConversionError(f"Input file not found: {path}")

    sep = args.sep or _detect_separator(path)

    try:
        df = pd.read_csv(path, sep=sep)
    except Exception as exc:
        raise MoleculeConversionError(f"Could not read table: {exc}") from exc

    columns = list(args.columns or [])
    if args.all_inchi_columns:
        columns = _inchi_columns(df.columns)
    elif not columns and len(df.columns) > 1:
        columns = _inchi_columns(df.columns)

    if columns:
        out = convert_table(
            df,
            columns,
            suffix=args.suffix,
            skip_errors=args.skip_errors,
        )
        if args.output:
            out.to_csv(args.output, sep=sep, index=False)
        else:
            out.to_csv(sys.stdout, sep=sep, index=False)
        return 0

    selfies_list = convert_text_file(path, skip_errors=args.skip_errors)
    text = "\n".join(selfies_list)
    _write_output(text, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
