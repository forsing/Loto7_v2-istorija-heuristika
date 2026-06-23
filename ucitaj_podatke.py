"""
Učitavanje Loto 7/39 istorije iz CSV (Num1..Num7).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from lottery_config import LOTO7, DEFAULT_CSV


def ucitaj_kola(putanja: str | Path | None = None) -> List[List[int]]:
    """Vrati listu kola — svako kolo je sortirana lista od 7 brojeva."""
    putanja = Path(putanja or DEFAULT_CSV)
    kola: List[List[int]] = []
    with putanja.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        kolone = [f"Num{i}" for i in range(1, LOTO7.brojeva_po_kombinaciji + 1)]
        for red in reader:
            brojevi = sorted(int(red[c]) for c in kolone)
            if len(brojevi) != LOTO7.brojeva_po_kombinaciji:
                raise ValueError(f"Kolo nema {LOTO7.brojeva_po_kombinaciji} brojeva: {brojevi}")
            if not all(LOTO7.min_broj <= b <= LOTO7.max_broj for b in brojevi):
                raise ValueError(f"Broj van opsega 1–39: {brojevi}")
            kola.append(brojevi)
    return kola


def poslednje_kolo(kola: List[List[int]]) -> List[int]:
    return kola[-1] if kola else []


def kao_dlt_format(kola: List[List[int]]) -> List[tuple[List[int], List[int]]]:
    """Kompatibilnost sa starim modulima: (7 brojeva, prazna dodatna zona)."""
    return [(k, []) for k in kola]


def kola_u_dataframe(kola: List[List[int]]):
    """DataFrame za strukturalne skills module (Glavna zona1..7)."""
    import pandas as pd

    cols = [f"Glavna zona{i}" for i in range(1, LOTO7.brojeva_po_kombinaciji + 1)]
    return pd.DataFrame([dict(zip(cols, k)) for k in kola])
