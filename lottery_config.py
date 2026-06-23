"""
Centralna konfiguracija — Loto 7/39 (Srbija), seed 39.
Opsezi zbira/raspnona iz loto7hh_4636_k49.csv (4636 kola).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_CSV = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4636_k49.csv")


@dataclass(frozen=True)
class LotteryConfig:
    min_broj: int = 1
    max_broj: int = 39
    brojeva_po_kombinaciji: int = 7
    seed: int = 39
    # zone: 1–13 / 14–26 / 27–39
    zona1_do: int = 13
    zona2_do: int = 26
    # statistika iz CSV (p10–p90)
    zbir_opseg: tuple[int, int] = (105, 176)
    zbir_ideal: tuple[int, int] = (130, 152)
    raspon_opseg: tuple[int, int] = (23, 36)
    prozor_vruc_hladno: int = 30


TEZINE_FUZIJE = {
    "pattern": 0.35,
    "struktura": 0.20,
    "bajes": 0.20,
    "constraint": 0.15,
    "neural": 0.10,
}


LOTO7 = LotteryConfig()

# Skills moduli (polazni API) — Loto 7/39 jedna zona
FRONT_N = LOTO7.max_broj
FRONT_SELECT = LOTO7.brojeva_po_kombinaciji
BACK_N = 0
BACK_SELECT = 0
if FRONT_N != 39:
    raise RuntimeError(f"Loto 7/39: FRONT_N mora biti 39, jeste {FRONT_N}")
FRONT_RANGE = (LOTO7.min_broj, LOTO7.max_broj)
BACK_RANGE = (0, 0)
# zone 1–13 / 14–26 / 27–39
ZONE1_RANGE = range(LOTO7.min_broj, LOTO7.zona1_do + 1)
ZONE2_RANGE = range(LOTO7.zona1_do + 1, LOTO7.zona2_do + 1)
ZONE3_RANGE = range(LOTO7.zona2_do + 1, LOTO7.max_broj + 1)
NUM_RANGE = range(LOTO7.min_broj, LOTO7.max_broj + 1)


def proveri_niz_verovatnoce(niz, ocekivano: int, ime: str):
    """Niz verovatnoća mora imati tačno ocekivano elemenata (indeks 0 = broj 1)."""
    import numpy as np
    a = np.asarray(niz, dtype=float).reshape(-1)
    if a.size != ocekivano:
        raise ValueError(f"{ime}: očekivano {ocekivano}, dobijeno {a.size}")
    return a

PROSTI_BROJEVI = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

POOL_TEZINE = {
    "vruci": 0.30,
    "hladni": 0.15,
    "balans": 0.20,
    "trend": 0.20,
    "igra": 0.10,
    "genetika": 0.05,
}

OBRAZAC_TEZINE = {
    "raspon": 0.12,
    "uzastopni": 0.10,
    "zbir": 0.12,
    "parno_neparno": 0.10,
    "ponavljanje": 0.15,
    "krajnja_cifra": 0.10,
    "zona": 0.12,
    "prosti": 0.07,
    "ac_vrednost": 0.12,
}


def zona(broj: int) -> int:
    if broj <= LOTO7.zona1_do:
        return 1
    if broj <= LOTO7.zona2_do:
        return 2
    return 3
