"""
Loto 7/39 — filter kombinacija (zbir, raspon, parno, zone).
"""

from __future__ import annotations

from typing import List

from lottery_config import LOTO7, zona


def prolazi_filter(brojevi: List[int]) -> bool:
  if len(brojevi) != LOTO7.brojeva_po_kombinaciji:
    return False
  if len(set(brojevi)) != LOTO7.brojeva_po_kombinaciji:
    return False
  if not all(LOTO7.min_broj <= b <= LOTO7.max_broj for b in brojevi):
    return False
  zbir = sum(brojevi)
  if not (LOTO7.zbir_opseg[0] <= zbir <= LOTO7.zbir_opseg[1]):
    return False
  raspon = max(brojevi) - min(brojevi)
  if not (LOTO7.raspon_opseg[0] <= raspon <= LOTO7.raspon_opseg[1]):
    return False
  parnih = sum(1 for b in brojevi if b % 2 == 0)
  if parnih == 0 or parnih == LOTO7.brojeva_po_kombinaciji:
    return False
  z1 = sum(1 for b in brojevi if zona(b) == 1)
  z2 = sum(1 for b in brojevi if zona(b) == 2)
  z3 = sum(1 for b in brojevi if zona(b) == 3)
  if min(z1, z2, z3) == 0:
    return False
  s = sorted(brojevi)
  run = 1
  for i in range(1, len(s)):
    run = run + 1 if s[i] == s[i - 1] + 1 else 1
    if run >= 5:
      return False
  return True
