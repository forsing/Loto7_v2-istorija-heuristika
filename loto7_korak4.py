#!/usr/bin/env python3
"""Korak 4 — + Bajes → loto7_korak4.txt"""

from __future__ import annotations

import argparse
from pathlib import Path

from loto7_engine import Loto7Engine
from loto7_koraci_io import upisi_predikciju
from lottery_config import DEFAULT_CSV, LOTO7
from ucitaj_podatke import poslednje_kolo, ucitaj_kola


def main() -> None:
    p = argparse.ArgumentParser(description="Korak 4")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--pokusaja", type=int, default=8000)
    p.add_argument("--seed", type=int, default=LOTO7.seed)
    args = p.parse_args()
    kola = ucitaj_kola(args.csv)
    eng = Loto7Engine(kola, korak=4, seed=args.seed)
    combo, sc = eng.najbolja(preth=poslednje_kolo(kola), max_pokusaja=args.pokusaja)
    out = upisi_predikciju(4, kola, combo, sc, args.csv)
    print(f"Korak 4 → {out}")


if __name__ == "__main__":
    main()




"""
Loto 7/39 — KORAK 4
Korak 3 + Bajesovski sloj
Generisano: 23.06.2026 11:03
CSV: /data/loto7hh_4636_k49.csv
Kola u CSV: 4636
Seed: 39

Kombinacija: 07  x  19  y  24  z  38
Skor: 0.7275
"""
