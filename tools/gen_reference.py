"""Generate PySCF reference data for cases declared in testdata/reference/cases.json.

For each case we:
  1. Load the basis from data/basis JSON files so PySCF and ZInt use the
     same contraction order and function-type metadata.
  2. Optionally preoptimize the geometry at HF / case basis (skipped for atoms)
     so the integrals live at a chemically sane stationary point.
  3. Compute 1e integrals (S, T, V_nuc) and optionally the 2e ERI tensor
     and an RHF energy + density matrix on the optimized geometry.
  4. Dump everything to testdata/reference/cases/<name>.json.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf
from pyscf.data.nist import BOHR
from pyscf.geomopt.geometric_solver import optimize as geometric_optimize


REPO_ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = REPO_ROOT / "testdata" / "reference" / "cases.json"
OUTPUT_DIR = REPO_ROOT / "testdata" / "reference" / "cases"
LOCAL_BASIS_DIR = REPO_ROOT / "data" / "basis"
ZERO_TOLERANCE = 5e-13
SIGNIFICANT_DIGITS = 10
SCF_SIGNIFICANT_DIGITS = 8

_BSE_ELEMENT_Z = {
    sym: z for z, sym in enumerate(
        [
            "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
            "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
            "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr",
        ]
    )
}


def _preoptimize(mol: gto.Mole) -> gto.Mole:
    """HF geometry optimization at the case's own basis. Single atoms are
    returned unchanged. Output is silenced; on failure the exception
    propagates."""
    if mol.natm < 2:
        return mol
    mf = scf.RHF(mol)
    mf.verbose = 0
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return geometric_optimize(mf, maxsteps=200)


def _basis_filename(name: str) -> str:
    out = name.lower()
    for src, dst in [
        ("*", "_st_"),
        ("/", "_sl_"),
        ("(", "_"),
        (")", "_"),
        (",", "_"),
        ("+", "_p_"),
        (" ", "_"),
    ]:
        out = out.replace(src, dst)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") + ".json"


def pyscf_basis_shell(shell: dict) -> list[list]:
    """Convert one local BSE JSON electron_shell to PySCF internal basis rows.

    For generalized contractions, keeping coefficient columns in JSON order is
    important. BSE's NWChem formatter may reorder columns for readability, while
    ZInt consumes the JSON order directly.
    """
    angular_momenta = [int(value) for value in shell["angular_momentum"]]
    exponents = [float(value) for value in shell["exponents"]]
    coefficients = [[float(value) for value in row] for row in shell["coefficients"]]

    if len(angular_momenta) == 1:
        return [
            [angular_momenta[0]]
            + [
                [exponent] + [column[primitive_index] for column in coefficients]
                for primitive_index, exponent in enumerate(exponents)
            ]
        ]

    out: list[list] = []
    for coefficient_index, angular_momentum in enumerate(angular_momenta):
        out.append(
            [angular_momentum]
            + [
                [exponent, coefficients[coefficient_index][primitive_index]]
                for primitive_index, exponent in enumerate(exponents)
            ]
        )
    return out


def basis_dict_for(case_basis: str, symbols: list[str]) -> dict[str, list]:
    """Return a PySCF basis map built from the local JSON files ZInt reads."""
    path = LOCAL_BASIS_DIR / _basis_filename(case_basis)
    if not path.exists():
        raise FileNotFoundError(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, list] = {}
    for sym in sorted(set(symbols)):
        z = str(_BSE_ELEMENT_Z[sym])
        element = data.get("elements", {}).get(z)
        if element is None:
            raise KeyError(f"{case_basis} has no basis data for {sym}")

        shells: list[list] = []
        for shell in element.get("electron_shells", []):
            shells.extend(pyscf_basis_shell(shell))
        out[sym] = shells
    return out


def use_cartesian_ao(case_basis: str, symbols: list[str]) -> bool:
    """Mirror the local JSON basis convention when PySCF has a global
    cartesian/spherical AO choice. s and p shells are equivalent for this
    choice; high-angular shells decide it."""
    path = LOCAL_BASIS_DIR / _basis_filename(case_basis)
    if not path.exists():
        return False

    data = json.loads(path.read_text(encoding="utf-8"))
    zs = {_BSE_ELEMENT_Z[sym] for sym in symbols}
    for z in zs:
        element = data.get("elements", {}).get(str(z), {})
        for shell in element.get("electron_shells", []):
            if max(shell.get("angular_momentum", [0])) <= 1:
                continue
            function_type = shell.get("function_type", "gto")
            if function_type == "gto_cartesian":
                return True
            if function_type == "gto_spherical":
                return False
    return False


def ao_labels_for(mol: gto.Mole) -> list[dict]:
    """Return one label per AO in the same order used by PySCF integral arrays."""
    labels = []
    for i, (atom_index, atom_symbol, shell, component) in enumerate(mol.ao_labels(fmt=False)):
        labels.append(
            {
                "index": i,
                "atom_index": int(atom_index),
                "atom_symbol": atom_symbol,
                "shell": shell,
                "component": component,
                "name": f"{shell}{component}",
            }
        )
    return labels


def cartesian_normalization_scales(mol: gto.Mole, overlap: np.ndarray) -> np.ndarray:
    """PySCF's cartesian high-angular AOs use its own cartesian convention.
    ZInt stores normalized Cartesian components, so cartesian reference
    arrays are diagonally transformed to unit AO self-overlap."""
    scales = np.ones(mol.nao_nr())
    if not mol.cart:
        return scales
    ao_loc = mol.ao_loc_nr(cart=True)
    overlap_diagonal = np.diag(overlap)
    for shell_index in range(mol.nbas):
        if mol.bas_angular(shell_index) <= 1:
            continue
        start = int(ao_loc[shell_index])
        end = int(ao_loc[shell_index + 1])
        scales[start:end] = 1.0 / np.sqrt(overlap_diagonal[start:end])
    return scales


def scale_one_electron(matrix: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return scales[:, None] * matrix * scales[None, :]


def scale_eri(eri: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (
        scales[:, None, None, None]
        * scales[None, :, None, None]
        * scales[None, None, :, None]
        * scales[None, None, None, :]
        * eri
    )


def canonicalize_json(value, digits: int = SIGNIFICANT_DIGITS):
    """Round floating reference data so repeated generation is byte-stable."""
    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        if abs(as_float) < ZERO_TOLERANCE:
            return 0.0
        return float(f"{as_float:.{digits}g}")
    if isinstance(value, list):
        return [canonicalize_json(item, digits) for item in value]
    if isinstance(value, dict):
        return {
            key: canonicalize_json(
                item,
                SCF_SIGNIFICANT_DIGITS if key == "scf" else digits,
            )
            for key, item in value.items()
        }
    return value


def run_case(case: dict) -> dict:
    name = case["name"]
    print(
        f"  [{case.get('cost', 'unknown')}] {name}: "
        f"basis={case['basis']} atoms={len(case['atoms'])}",
        flush=True,
    )

    symbols = [a[0] for a in case["atoms"]]
    atom_lines = [
        f"{sym}  {xyz[0]:.10f}  {xyz[1]:.10f}  {xyz[2]:.10f}"
        for sym, xyz in case["atoms"]
    ]

    basis_map = basis_dict_for(case["basis"], symbols)
    cartesian_ao = use_cartesian_ao(case["basis"], symbols)

    mol = gto.M(
        atom="\n".join(atom_lines),
        basis=basis_map,
        unit=case["unit"],
        charge=int(case["charge"]),
        spin=int(case["spin"]),
        cart=cartesian_ao,
        verbose=0,
    )

    if case.get("optimize_geometry", True):
        mol = _preoptimize(mol)

    # pyscf stores atom coords internally in bohr; emit them in the case's unit.
    coords_bohr = mol.atom_coords()
    scale = 1.0 if case["unit"] == "bohr" else BOHR
    used_atoms = [
        [mol.atom_pure_symbol(i), [float(c * scale) for c in coords_bohr[i]]]
        for i in range(mol.natm)
    ]

    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V_nuc = mol.intor("int1e_nuc")
    scales = cartesian_normalization_scales(mol, S)
    S = scale_one_electron(S, scales)
    T = scale_one_electron(T, scales)
    V_nuc = scale_one_electron(V_nuc, scales)

    out: dict = {
        "case": name,
        "molecule": {
            "atoms": used_atoms,
            "unit": case["unit"],
            "charge": int(case["charge"]),
            "spin": int(case["spin"]),
        },
        "basis": case["basis"],
        "nao": int(mol.nao_nr()),
        "natm": int(mol.natm),
        "nelectron": int(mol.nelectron),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "pyscf_version": __import__("pyscf").__version__,
        "ao_labels": ao_labels_for(mol),
        "S": S.tolist(),
        "T": T.tolist(),
        "V_nuc": V_nuc.tolist(),
    }

    if case.get("compute_eri", False):
        eri = mol.intor("int2e")
        eri = scale_eri(eri, scales)
        out["eri"] = eri.tolist()

    if case.get("compute_scf", False):
        mf = scf.RHF(mol)
        mf.verbose = 0
        e_tot = mf.kernel()
        density = mf.make_rdm1()
        density = density / (scales[:, None] * scales[None, :])
        out["scf"] = {
            "method": "RHF",
            "converged": bool(mf.converged),
            "energy": float(e_tot),
            "density": density.tolist(),
            "mo_energies": mf.mo_energy.tolist(),
        }

    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    cases = config.get("cases", [])
    if not cases:
        print(f"No cases found in {CASES_FILE}", file=sys.stderr)
        return 1

    seen_names: set[str] = set()
    expected_paths: set[Path] = set()
    for case in cases:
        name = str(case.get("name", ""))
        if not name:
            print("Every case needs a non-empty name", file=sys.stderr)
            return 1
        if name in seen_names:
            print(f"Duplicate case name in {CASES_FILE}: {name}", file=sys.stderr)
            return 1
        seen_names.add(name)
        if case.get("cost") not in {"fast", "slow"}:
            print(f"{name}: cost must be 'fast' or 'slow'", file=sys.stderr)
            return 1
        expected_paths.add(OUTPUT_DIR / f"{name}.json")

    start = time.perf_counter()
    failures: list[tuple[str, str]] = []

    for case in cases:
        try:
            payload = canonicalize_json(run_case(case))
        except Exception as exc:
            failures.append((case.get("name", "?"), repr(exc)))
            continue

        out_path = OUTPUT_DIR / f"{case['name']}.json"
        out_path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )

    if not failures:
        for path in OUTPUT_DIR.glob("*.json"):
            if path not in expected_paths:
                path.unlink()

    elapsed = time.perf_counter() - start
    print(f"Wrote {len(cases) - len(failures)} cases in {elapsed:.1f}s to {OUTPUT_DIR}")
    if failures:
        print(f"WARNING: {len(failures)} case(s) failed:", file=sys.stderr)
        for name, err in failures:
            print(f"  {name}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
