"""
Loto 7/39 — sampler preko polaznog pool_sampler.py.
"""

from __future__ import annotations

import random
from typing import List

from lottery_config import LOTO7
from pool_sampler import MultiPoolSampler
from ucitaj_podatke import kao_dlt_format


class Loto7Sampler:
  def __init__(self, kola: List[List[int]], rng: random.Random | None = None, tiho: bool = True):
    self.kola = kola
    self.rng = rng or random.Random(LOTO7.seed)
    self._pool = MultiPoolSampler(kao_dlt_format(kola), tiho=tiho)

  def nasumicna_kombinacija(self) -> List[int]:
    state = random.getstate()
    random.setstate(self.rng.getstate())
    try:
      combos = self._pool.stratified_sample(1, zone="front")
      if combos and len(combos[0]) == LOTO7.brojeva_po_kombinaciji:
        return sorted(combos[0])
      # dopuna ako stratified vrati kraće
      combo = list(combos[0]) if combos else []
      opseg = [n for n in range(LOTO7.min_broj, LOTO7.max_broj + 1) if n not in combo]
      while len(combo) < LOTO7.brojeva_po_kombinaciji and opseg:
        combo.append(self.rng.choice(opseg))
        opseg = [n for n in opseg if n not in combo]
      return sorted(combo[: LOTO7.brojeva_po_kombinaciji])
    finally:
      self.rng.setstate(random.getstate())
      random.setstate(state)

