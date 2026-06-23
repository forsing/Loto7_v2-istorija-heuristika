#!/usr/bin/env python3
"""
Motor ograničenja — tvrda, strategijska, meka.
"""


from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from lottery_config import LOTO7, PROSTI_BROJEVI, zona


# ============================================================
# Konstante — Loto 7/39
# ============================================================

FRONT_RANGE = (LOTO7.min_broj, LOTO7.max_broj)
BACK_RANGE = (0, 0)
FRONT_COUNT = LOTO7.brojeva_po_kombinaciji
BACK_COUNT = 0

PRIMES_FRONT = PROSTI_BROJEVI
PRIMES_BACK: set[int] = set()


# ============================================================
# Konfiguracija strategija (prebačeno iz dlt_strategy_fusion_v2.py)
# ============================================================

STRATEGY_CONFIGS: Dict[int, Dict] = {
    1: {
        "name": "Strategija balansirane konfiguracije",
        "hard_sum_range": LOTO7.zbir_ideal,
        "hard_odd_even": (3, 4),
        "hard_zone_required": True,
        "hard_span_min": 28,
        "hard_prime_count": (1, 4),
        "soft_ac_range": (6, 12),
        "soft_consecutive_min": 0,
        "soft_consecutive_max": 2,
        "soft_sum_ideal": LOTO7.zbir_ideal,
    },
    2: {
        "name": "Strategija širokog opsega",
        "hard_sum_range": LOTO7.zbir_opseg,
        "hard_zone_required": True,
        "hard_span_min": 23,
        "soft_ac_range": (5, 12),
        "soft_consecutive_min": 0,
        "soft_consecutive_max": 3,
        "soft_sum_ideal": LOTO7.zbir_opseg,
    },
    3: {
        "name": "Strategija agresivne visoke sume",
        "hard_sum_range": (150, LOTO7.zbir_opseg[1]),
        "hard_zone_required": True,
        "hard_span_min": 30,
        "soft_ac_range": (6, 12),
        "soft_consecutive_min": 0,
        "soft_consecutive_max": 2,
        "soft_sum_ideal": (150, LOTO7.zbir_opseg[1]),
    },
    4: {
        "name": "Strategija konzervativne sume",
        "hard_sum_range": (LOTO7.zbir_opseg[0], 130),
        "hard_zone_required": True,
        "hard_span_min": 23,
        "soft_ac_range": (5, 11),
        "soft_consecutive_min": 0,
        "soft_consecutive_max": 3,
        "soft_sum_ideal": (LOTO7.zbir_opseg[0], 130),
    },
    5: {
        "name": "Strategija praćenja trenda",
        "hard_sum_range": LOTO7.zbir_opseg,
        "hard_zone_required": False,
        "hard_span_min": 20,
        "soft_ac_range": (5, 12),
        "soft_consecutive_min": 0,
        "soft_consecutive_max": 3,
        "soft_sum_ideal": (120, 160),
    },
    6: {
        "name": "Strategija niš ravnoteže (uključujući sve-par/sve-nepar)",
        "hard_sum_range": LOTO7.zbir_opseg,
        "hard_zone_required": False,
        "hard_span_min": 20,
        "soft_ac_range": (4, 12),
        "soft_consecutive_min": 0,
        "soft_consecutive_max": 3,
        "soft_sum_ideal": LOTO7.zbir_opseg,
        "allow_all_odd": True,
        "allow_all_even": True,
    },
}


# ============================================================
# Glavna klasa motora
# ============================================================

class DLTConstraintEngine:
    """
    Jedinstveni motor za validaciju ograničenja

    Pruža tri vrste validacije ograničenja:
    - validate_hard: 7 brojeva ∈ [1,39], međusobno različiti (Loto 7/39)
    - validate_strategy: suma, par/nepar, tri zone, raspon, prosti brojevi itd.
    - score_soft: meki skor (AC, uzastopni, idealan zbir)
    """

    def __init__(self, strategy_configs: Optional[Dict[int, Dict]] = None):
        """
        Args:
            strategy_configs: opciono, zamenjuje podrazumevanu konfiguraciju strategija
        """
        self.strategy_configs = strategy_configs or STRATEGY_CONFIGS

    # ----------------------------------------------------------
    # Validacija tvrdog ograničenja
    # ----------------------------------------------------------

    def validate_hard(self, front: List[int], back: List[int]) -> Tuple[bool, str]:
        """
        Validacija tvrdog ograničenja: osnovni format + provera opsega

        Mora biti ispunjeno:
        - 7 brojeva ∈ [1, 39], međusobno različiti
        - nema dodatne zone (back mora biti prazna lista)

        Returns:
            (da li je prošlo, opis razloga)
        """
        # Tvrdo ograničenje Glavne zone
        if len(front) != FRONT_COUNT:
            return False, f"Broj pogrešan: {len(front)}, očekivano {FRONT_COUNT}"

        if len(set(front)) != len(front):
            return False, f"Kombinacija sadrži duplikate: {front}"

        if not all(FRONT_RANGE[0] <= n <= FRONT_RANGE[1] for n in front):
            bad = [n for n in front if not (FRONT_RANGE[0] <= n <= FRONT_RANGE[1])]
            return False, f"Brojevi van opsega [1,39]: {bad}"

        if BACK_COUNT == 0:
            if back:
                return False, f"Loto 7/39 nema dodatnu zonu: {back}"
            return True, "OK"

        # DLT dodatna zona (ako je BACK_COUNT > 0)
        if len(back) != BACK_COUNT:
            return False, f"Dodatna zona broj pogrešan: {len(back)}, očekivano {BACK_COUNT}"

        if len(set(back)) != len(back):
            return False, f"Dodatna zona sadrži duplikate: {back}"

        if not all(BACK_RANGE[0] <= n <= BACK_RANGE[1] for n in back):
            bad = [n for n in back if not (BACK_RANGE[0] <= n <= BACK_RANGE[1])]
            return False, f"Dodatna zona brojevi van opsega [1,12]: {bad}"

        return True, "OK"

    def validate_strategy(
        self,
        front: List[int],
        back: List[int],
        strategy_type: int
    ) -> Tuple[bool, str]:
        """
        Validacija strategijskog ograničenja

        Primena odgovarajuće konfiguracije strategije prema strategy_type.

        Args:
            front: lista brojeva Glavne zone
            back: lista brojeva Dodatne zone
            strategy_type: tip strategije (1-5)

        Returns:
            (da li je prošlo, opis razloga)
        """
        if strategy_type not in self.strategy_configs:
            return False, f"Nepoznat tip strategije: {strategy_type}"

        cfg = self.strategy_configs[strategy_type]
        front_sorted = sorted(front)

        # Opseg sume
        if "hard_sum_range" in cfg:
            lo, hi = cfg["hard_sum_range"]
            s = sum(front)
            if not (lo <= s <= hi):
                return False, f"Glavna zona suma {s} van opsega strategije [{lo},{hi}]"

        # Potpunost tri zone
        if cfg.get("hard_zone_required", False):
            z1 = sum(1 for n in front if zona(n) == 1)
            z2 = sum(1 for n in front if zona(n) == 2)
            z3 = sum(1 for n in front if zona(n) == 3)
            if z1 == 0 or z2 == 0 or z3 == 0:
                return False, f"Tri zone [{z1},{z2},{z3}] nepotpune (strategija zahteva sve tri zone)"

        # Odnos par/nepar
        if "hard_odd_even" in cfg:
            odd = sum(1 for n in front if n % 2 == 1)
            even = len(front) - odd
            target = cfg["hard_odd_even"]
            # Provera da li su dozvoljeni svi-nepar/svi-par (niš putanja strategije 6)
            allow_all_odd = cfg.get("allow_all_odd", False)
            allow_all_even = cfg.get("allow_all_even", False)
            n = len(front)
            if even == n and allow_all_even:
                pass
            elif odd == n and allow_all_odd:
                pass
            elif (odd, even) != target and (odd, even) != (target[1], target[0]):
                return False, f"Odnos par/nepar ({odd}:{even}) ne odgovara zahtevima strategije"

        # Raspon
        if "hard_span_min" in cfg:
            span = max(front) - min(front)
            if span < cfg["hard_span_min"]:
                return False, f"Raspon {span} manji od minimalnog zahteva {cfg['hard_span_min']}"

        # Isključivanje zaokruženih desetica
        if cfg.get("hard_no_round", False):
            if any(n % 10 == 0 for n in front):
                return False, "Strategija zabranjuje zaokružene desetice"

        # Broj prostih brojeva
        if "hard_prime_count" in cfg:
            prime_cnt = sum(1 for n in front if n in PRIMES_FRONT)
            lo, hi = cfg["hard_prime_count"]
            if not (lo <= prime_cnt <= hi):
                return False, f"Broj prostih brojeva {prime_cnt} van opsega strategije [{lo},{hi}]"

        return True, "OK"

    # ----------------------------------------------------------
    # Meki skor
    # ----------------------------------------------------------

    def score_soft(
        self,
        front: List[int],
        back: List[int],
        strategy_type: Optional[int] = None
    ) -> float:
        """
        Meki skor

        Računa skor zadovoljenja (0.0 ~ 1.0) prema konfiguraciji strategije, više je bolje.

        Dimenzije ocenjivanja:
        - Da li je AC vrednost u idealnom opsegu
        - Da li je broj uzastopnih brojeva odgovarajući
        - Da li je suma u idealnom intervalu

        Args:
            front: lista brojeva Glavne zone
            back: lista brojeva Dodatne zone
            strategy_type: opciono, tip strategije za odgovarajuće kriterijume ocenjivanja

        Returns:
            skor zadovoljenja (0.0 ~ 1.0)
        """
        score = 1.0

        # Konfiguracija specifična za strategiju
        cfg = {}
        if strategy_type and strategy_type in self.strategy_configs:
            cfg = self.strategy_configs[strategy_type]

        # Ocenjivanje AC vrednosti
        if "soft_ac_range" in cfg:
            ac = self._compute_ac(front)
            lo, hi = cfg["soft_ac_range"]
            if not (lo <= ac <= hi):
                score *= 0.5

        # Ocenjivanje uzastopnih brojeva
        if "soft_consecutive_min" in cfg or "soft_consecutive_max" in cfg:
            consecutive_count = self._count_consecutive(front)
            lo = cfg.get("soft_consecutive_min", 0)
            hi = cfg.get("soft_consecutive_max", 999)
            if not (lo <= consecutive_count <= hi):
                score *= 0.7

        # Ocenjivanje idealnog intervala sume
        if "soft_sum_ideal" in cfg:
            s = sum(front)
            lo, hi = cfg["soft_sum_ideal"]
            if not (lo <= s <= hi):
                score *= 0.7

        return max(0.0, score)

    # ----------------------------------------------------------
    # Kombinovana validacija (Tvrdo ograničenje + Strategijsko ograničenje)
    # ----------------------------------------------------------

    def validate_full(
        self,
        front: List[int],
        back: List[int],
        strategy_type: int
    ) -> Tuple[bool, str]:
        """
        Potpuna validacija: Tvrdo ograničenje + Strategijsko ograničenje

        Ekvivalentno uzastopnom pozivanju validate_hard i validate_strategy.
        """
        # Korak 1: Tvrdo ograničenje
        ok, msg = self.validate_hard(front, back)
        if not ok:
            return False, f"[Tvrdo ograničenje] {msg}"

        # Korak 2: Strategijsko ograničenje
        ok, msg = self.validate_strategy(front, back, strategy_type)
        if not ok:
            return False, f"[Strategijsko ograničenje] {msg}"

        return True, "OK"

    # ----------------------------------------------------------
    # Pomoćne metode za izračunavanje
    # ----------------------------------------------------------

    @staticmethod
    def _compute_ac(numbers: List[int]) -> int:
        """Računa AC vrednost"""
        diffs = set()
        for i in range(len(numbers)):
            for j in range(i + 1, len(numbers)):
                diffs.add(abs(numbers[j] - numbers[i]))
        return len(diffs) - (len(numbers) - 1)

    @staticmethod
    def _count_consecutive(numbers: List[int]) -> int:
        """Računa broj uzastopnih brojeva (parova sa razlikom 1)"""
        sorted_nums = sorted(numbers)
        count = 0
        i = 0
        while i < len(sorted_nums) - 1:
            if sorted_nums[i + 1] - sorted_nums[i] == 1:
                count += 1
                i += 2
            else:
                i += 1
        return count

    @staticmethod
    def _zone_distribution(front: List[int]) -> Tuple[int, int, int]:
        """Raspodela po zonama 1–13 / 14–26 / 27–39."""
        z1 = sum(1 for n in front if zona(n) == 1)
        z2 = sum(1 for n in front if zona(n) == 2)
        z3 = sum(1 for n in front if zona(n) == 3)
        return z1, z2, z3

    @staticmethod
    def _odd_even_ratio(front: List[int]) -> Tuple[int, int]:
        """Računa odnos par/nepar"""
        odd = sum(1 for n in front if n % 2 == 1)
        even = len(front) - odd
        return odd, even


# ============================================================
# Samostalni funkcijski interfejs (kompatibilnost sa starim kodom)
# ============================================================

# Globalna podrazumevana instanca motora
_default_engine: Optional[DLTConstraintEngine] = None


def get_engine() -> DLTConstraintEngine:
    """Dobavlja globalnu instancu motora ograničenja (lenjo kreiranje)"""
    global _default_engine
    if _default_engine is None:
        _default_engine = DLTConstraintEngine()
    return _default_engine


def validate_hard(front: List[int], back: List[int]) -> Tuple[bool, str]:
    """Samostalna funkcija: validacija tvrdog ograničenja"""
    return get_engine().validate_hard(front, back)


def validate_strategy(
    front: List[int],
    back: List[int],
    strategy_type: int
) -> Tuple[bool, str]:
    """Samostalna funkcija: validacija strategijskog ograničenja"""
    return get_engine().validate_strategy(front, back, strategy_type)


def score_soft(
    front: List[int],
    back: List[int],
    strategy_type: Optional[int] = None
) -> float:
    """Samostalna funkcija: računanje mekog skora"""
    return get_engine().score_soft(front, back, strategy_type)


# ============================================================
# Brzi test
# ============================================================

if __name__ == "__main__":
    engine = DLTConstraintEngine()

    # Test slučajevi
    test_cases = [
        ([5, 10, 15, 20, 25, 30, 35], [], 1, True, "Balans — normalno"),
        ([1, 1, 3, 5, 7, 9, 11], [], 1, False, "Duplikat"),
        ([1, 5, 12, 20, 28, 35, 40], [], 1, False, "Van opsega 1–39"),
        ([5, 12, 18, 24, 28, 31, 35], [3, 9], 1, False, "Dodatna zona nije dozvoljena"),
    ]

    print("Loto7ConstraintEngine provera")
    print("=" * 60)

    all_pass = True
    for front, back, stype, expected_ok, desc in test_cases:
        ok_h, msg_h = engine.validate_hard(front, back)
        if not ok_h:
            ok_s, msg_s = False, "Prethodno tvrdo ograničenje nije prošlo"
        else:
            ok_s, msg_s = engine.validate_strategy(front, back, stype)

        ok_full, _ = engine.validate_full(front, back, stype)
        sc = engine.score_soft(front, back, stype)
        actual_ok = ok_full

        status = "✅" if actual_ok == expected_ok else "❌"
        if actual_ok != expected_ok:
            all_pass = False
        print(f"{status} {desc}")
        print(f"   Tvrdo ograničenje: {ok_h} {msg_h}")
        print(f"   Strategijsko ograničenje: {ok_s} {msg_s}")
        print(f"   Meki skor: {sc:.2f}")

    print("=" * 60)
    print(f"Provera: {'✅ Sve prošlo' if all_pass else '❌ Ima grešaka'}")


"""
kor: Loto7ConstraintEngine provera
============================================================
✅ Balans — normalno
   Tvrdo ograničenje: True OK
   Strategijsko ograničenje: True OK
   Meki skor: 0.50
✅ Duplikat
   Tvrdo ograničenje: False Kombinacija sadrži duplikate: [1, 1, 3, 5, 7, 9, 11]
   Strategijsko ograničenje: False Prethodno tvrdo ograničenje nije prošlo
   Meki skor: 0.35
✅ Van opsega 1–39
   Tvrdo ograničenje: False Brojevi van opsega [1,39]: [40]
   Strategijsko ograničenje: False Prethodno tvrdo ograničenje nije prošlo
   Meki skor: 1.00
✅ Dodatna zona nije dozvoljena
   Tvrdo ograničenje: False Loto 7/39 nema dodatnu zonu: [3, 9]
   Strategijsko ograničenje: False Prethodno tvrdo ograničenje nije prošlo
   Meki skor: 0.70
============================================================
Provera: ✅ Sve prošlo
"""
