#!/usr/bin/env python3
"""Korak 5 — + neuronska mreža → loto7_korak5.txt"""

from __future__ import annotations

import argparse
from pathlib import Path

from loto7_engine import Loto7Engine
from loto7_koraci_io import upisi_predikciju
from lottery_config import DEFAULT_CSV, LOTO7
from ucitaj_podatke import poslednje_kolo, ucitaj_kola


def main() -> None:
    p = argparse.ArgumentParser(description="Korak 5 — neuronska mreža (može biti sporo)")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--pokusaja", type=int, default=8000)
    p.add_argument("--seed", type=int, default=LOTO7.seed)
    args = p.parse_args()
    kola = ucitaj_kola(args.csv)
    eng = Loto7Engine(kola, korak=5, seed=args.seed)
    combo, sc = eng.najbolja(preth=poslednje_kolo(kola), max_pokusaja=args.pokusaja)
    out = upisi_predikciju(5, kola, combo, sc, args.csv)
    print(f"Korak 5 → {out}")


if __name__ == "__main__":
    main()


"""
Loto 7/39 — KORAK 5
Korak 4 + neuronska mreža (PyTorch, može biti sporo)
Generisano: 23.06.2026 11:08
CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4636_k49.csv
Kola u CSV: 4636
Seed: 39

Kombinacija: 07  09  19  22  24  32  38
Skor: 0.7275
"""
