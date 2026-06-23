"""
Loto 7/39 — fuzija preko polaznih skills modula (pattern, constraint, bajes, struktura).
"""

from __future__ import annotations

from typing import List, Optional

from bayesian import BetaBinomialMissing
from constraint_engine import DLTConstraintEngine
from difference_sequence import DLTDifferenceSequence
from lottery_config import LOTO7, TEZINE_FUZIJE
from matrix_displacement import DLTMatrixDisplacement
from number_gravity import DLTNumberGravity
from pattern_recognizer import DLTPatternRecognizer
from ucitaj_podatke import kao_dlt_format, kola_u_dataframe

TEZINE = TEZINE_FUZIJE


class Loto7Fusion:
  def __init__(
    self,
    kola: List[List[int]],
    neural_skor_fn=None,
  ):
    draws = kao_dlt_format(kola)
    df = kola_u_dataframe(kola)
    self.kola = kola
    self._pattern = DLTPatternRecognizer(draws)
    self._pattern.build_distributions()
    self._constraint = DLTConstraintEngine()
    self._bajes = BetaBinomialMissing(method="jeffreys")
    self._bajes.update_batch(kola)
    self._diff = DLTDifferenceSequence(df)
    self._gravity = DLTNumberGravity(df)
    self._matrix = DLTMatrixDisplacement(df)
    self.neural_skor_fn = neural_skor_fn

  def _struktura(self, brojevi: List[int]) -> float:
    d = self._diff.get_diff_score(brojevi)
    g = self._gravity.calculate_gravity_score(brojevi)
    m = (
      self._matrix.calculate_dispersion_score(brojevi)
      + self._matrix.calculate_center_gravity(brojevi)
    ) / 2.0
    return (d + g + m) / 3.0

  def ukupan_skor(self, brojevi: List[int], preth: Optional[List[int]] = None) -> float:
    preth = preth or (self.kola[-1] if self.kola else [])
    p = self._pattern.score_combo(brojevi, preth)["total_score"]
    s = self._struktura(brojevi)
    b = sum(self._bajes.get_expected_prob(n) for n in brojevi) / len(brojevi)
    c = self._constraint_skor(brojevi)
    total = (
      TEZINE["pattern"] * p
      + TEZINE["struktura"] * s
      + TEZINE["bajes"] * b
      + TEZINE["constraint"] * c
    )
    if self.neural_skor_fn is not None:
      total += TEZINE["neural"] * float(self.neural_skor_fn(brojevi))
    return total

  def _constraint_skor(self, brojevi: List[int]) -> float:
    ok_scores = []
    for sid in self._constraint.strategy_configs:
      ok, _ = self._constraint.validate_strategy(brojevi, [], sid)
      if ok:
        ok_scores.append(self._constraint.score_soft(brojevi, [], sid))
    return sum(ok_scores) / len(ok_scores) if ok_scores else 0.0

  # kompatibilnost sa loto7_engine._skor
  @property
  def pattern(self):
    return self

  @property
  def struktura(self):
    return self

  @property
  def constraint(self):
    return self

  @property
  def bajes(self):
    return self

  def oceni_kombinaciju(self, brojevi: List[int], preth: Optional[List[int]] = None) -> float:
    preth = preth or (self.kola[-1] if self.kola else [])
    return self._pattern.score_combo(brojevi, preth)["total_score"]

  def kombinovano(self, brojevi: List[int]) -> float:
    return self._struktura(brojevi)

  def fusion_skor(self, brojevi: List[int]) -> float:
    return self._constraint_skor(brojevi)

  def skor_kombinacije(self, brojevi: List[int]) -> float:
    return sum(self._bajes.get_expected_prob(n) for n in brojevi) / len(brojevi)
