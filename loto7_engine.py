"""
Loto 7/39 — motor sa nivoom koraka (2–6).
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from loto7_filter import prolazi_filter
from loto7_fusion import Loto7Fusion
from loto7_sampler import Loto7Sampler
from lottery_config import LOTO7


class Loto7Engine:
    def __init__(
        self,
        kola: List[List[int]],
        korak: int = 6,
        seed: int = LOTO7.seed,
        tiho: bool = True,
    ):
        if korak < 2 or korak > 6:
            raise ValueError("korak mora biti 2–6 za predikciju")
        self.kola = kola
        self.korak = korak
        self.rng = random.Random(seed)
        self.sampler = Loto7Sampler(kola, rng=self.rng, tiho=tiho)
        neural_fn = None
        if korak >= 5:
            from loto7_neural import napravi_neural_skor
            neural_fn = napravi_neural_skor(kola, tiho=tiho)
        self.fusion = Loto7Fusion(kola, neural_skor_fn=neural_fn if korak >= 5 else None)

    def _skor(self, brojevi: List[int], preth: Optional[List[int]]) -> float:
        p = self.fusion.pattern.oceni_kombinaciju(brojevi, preth)
        s = self.fusion.struktura.kombinovano(brojevi)
        c = self.fusion.constraint.fusion_skor(brojevi)
        b = self.fusion.bajes.skor_kombinacije(brojevi)

        if self.korak == 2:
            return 0.55 * p + 0.45 * s
        if self.korak == 3:
            osnova = 0.55 * p + 0.45 * s
            return 0.70 * osnova + 0.30 * c
        if self.korak == 4:
            osnova = 0.55 * p + 0.45 * s
            sred = 0.65 * osnova + 0.35 * c
            return 0.75 * sred + 0.25 * b
        if self.korak == 5:
            osnova = 0.55 * p + 0.45 * s
            sred = 0.65 * osnova + 0.35 * c
            s4 = 0.75 * sred + 0.25 * b
            if self.fusion.neural_skor_fn:
                n = float(self.fusion.neural_skor_fn(brojevi))
                return 0.85 * s4 + 0.15 * n
            return s4
        return self.fusion.ukupan_skor(brojevi, preth)

    def generisi_kandidate(
        self,
        preth: Optional[List[int]] = None,
        max_pokusaja: int = 8000,
        cilj: Optional[int] = None,
    ) -> List[Tuple[List[int], float]]:
        preth = preth or (self.kola[-1] if self.kola else [])
        vidjeno: set[tuple[int, ...]] = set()
        out: List[Tuple[List[int], float]] = []
        for _ in range(max_pokusaja):
            if cilj is not None and len(out) >= cilj:
                break
            combo = self.sampler.nasumicna_kombinacija()
            key = tuple(combo)
            if key in vidjeno:
                continue
            vidjeno.add(key)
            if not prolazi_filter(combo):
                continue
            out.append((combo, self._skor(combo, preth)))
        out.sort(key=lambda x: -x[1])
        return out

    def najbolja(
        self,
        preth: Optional[List[int]] = None,
        max_pokusaja: int = 8000,
    ) -> Tuple[List[int], float]:
        k = self.generisi_kandidate(preth=preth, max_pokusaja=max_pokusaja)
        if not k:
            combo = sorted(
                self.rng.sample(
                    range(LOTO7.min_broj, LOTO7.max_broj + 1),
                    LOTO7.brojeva_po_kombinaciji,
                )
            )
            return combo, 0.0
        return k[0]
