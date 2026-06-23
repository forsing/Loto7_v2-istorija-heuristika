"""
Putanje izlaza za 6 koraka.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

from lottery_config import DEFAULT_CSV, LOTO7

DIR = Path(__file__).resolve().parent

KORAK_IZLAZ = {
    1: DIR / "loto7_korak1.txt",
    2: DIR / "loto7_korak2.txt",
    3: DIR / "loto7_korak3.txt",
    4: DIR / "loto7_korak4.txt",
    5: DIR / "loto7_korak5.txt",
    6: DIR / "loto7_korak6.txt",
}

KORAK_OPIS = {
    1: "Retrospektivni test (walk-forward, ceo CSV)",
    2: "6 bazena + obrasci + struktura (razmake, gravitacija, matrica)",
    3: "Korak 2 + ograničenje fuzije (5 strategija)",
    4: "Korak 3 + Bajesovski sloj",
    5: "Korak 4 + neuronska mreža (PyTorch, može biti sporo)",
    6: "Pun pipeline — svi slojevi",
}


def upisi_predikciju(
    korak: int,
    kola: List[List[int]],
    combo: List[int],
    score: float,
    csv_putanja: Path = DEFAULT_CSV,
) -> Path:
    putanja = KORAK_IZLAZ[korak]
    sada = datetime.now().strftime("%d.%m.%Y %H:%M")
    tekst = "  ".join(f"{b:02d}" for b in combo)
    linije = [
        f"Loto 7/39 — KORAK {korak}",
        KORAK_OPIS[korak],
        f"Generisano: {sada}",
        f"CSV: {csv_putanja}",
        f"Kola u CSV: {len(kola)}",
        f"Seed: {LOTO7.seed}",
        "",
        f"Kombinacija: {tekst}",
        f"Skor: {score:.4f}",
        "",
    ]
    putanja.write_text("\n".join(linije), encoding="utf-8")
    return putanja
