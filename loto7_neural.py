"""
Loto 7/39 — opcioni neuralni sloj (PyTorch).
"""

from __future__ import annotations

from typing import Callable, List, Optional

try:
  from neural_models import NeuralEnsemble, TORCH_AVAILABLE
except ImportError:
  TORCH_AVAILABLE = False
  NeuralEnsemble = None  # type: ignore


def napravi_neural_skor(kola: List[List[int]], tiho: bool = True) -> Optional[Callable[[List[int]], float]]:
  if not TORCH_AVAILABLE or NeuralEnsemble is None:
    return None
  try:
    draws = [(k, []) for k in kola]
    ens = NeuralEnsemble(draws, train_epochs=8, auto_train=True, verbose=not tiho)

    def _skor(brojevi: List[int]) -> float:
      probs = ens.predict_probabilities()
      front = probs.get("front", {})
      if not front:
        return 0.5
      return sum(front.get(n, 0.0) for n in brojevi) / len(brojevi)

    return _skor
  except Exception:
    return None

