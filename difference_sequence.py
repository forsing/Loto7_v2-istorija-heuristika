#!/usr/bin/env python3
"""
Analiza sekvence razlika između brojeva.
"""


from typing import List, Dict, Any
import numpy as np

from lottery_config import LOTO7, FRONT_SELECT, FRONT_N


class DLTDifferenceSequence:
    """
    Analiza sekvence razlika:
    1. Prvi red razlika: razlika susednih brojeva
    2. Drugi red razlika: razlika razlika (ubrzanje)
    3. Entropija razlika: stepen haosa u raspodeli razlika
    
    Napomena: svi parametri imaju fizičku gornju granicu, bez prekoračenja
    """
    
    # Zlatna raspodela razlika (iz Istorije)
    DIFF_HIGH_FREQ = [1, 2, 3, 4, 5, 6]  # visokofrekventne razlike (ograničeno)
    DIFF_VERY_RARE = [15, 16, 17, 18]   # veoma retke razlike (ograničeno)
    
    def __init__(self, historical_data=None):
        """
        Inicijalizacija analizatora sekvence razlika
        
        Args:
            historical_data: IstorijaDataFrame (sadrži kolone Glavna zona1-5)
        """
        self.historical_data = historical_data
        self.diff_distribution = None  # verovatnoća raspodele razlika
        if historical_data is not None:
            self._build_diff_distribution()
    
    def _build_diff_distribution(self):
        """Gradnja raspodele razlika iz Istorije"""
        df = self.historical_data
        cols = [f"Glavna zona{i}" for i in range(1, FRONT_SELECT + 1)]
        front_numbers = df[cols].values
        
        diff_counts = np.zeros(FRONT_N)  # razlike 1..(N-1)
        
        for row in front_numbers:
            sorted_row = sorted([int(n) for n in row])
            for i in range(len(sorted_row) - 1):
                diff = sorted_row[i + 1] - sorted_row[i]
                if 1 <= diff <= 34:
                    diff_counts[diff - 1] += 1
        
        # normalizacija u verovatnoću (ograničeno: 0-1)
        total = diff_counts.sum()
        if total > 0:
            self.diff_distribution = diff_counts / total
        else:
            self.diff_distribution = None
    
    def calculate_diff_sequence(self, combo: List[int]) -> np.ndarray:
        """
        Računanje sekvence razlika kombinacije (prvi red razlika)
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            np.ndarray: niz razlika (4 razlike)
        """
        sorted_combo = sorted(combo)
        diffs = np.array([
            sorted_combo[i + 1] - sorted_combo[i] 
            for i in range(len(sorted_combo) - 1)
        ])
        return diffs
    
    def calculate_diff_entropy(self, combo: List[int]) -> float:
        """
        Računanje entropije razlika (niža = jača regularnost)
        
        Entropija: H = -Σp*log(p)
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            float: entropija razlika (0-1, ograničeno)
        """
        diffs = self.calculate_diff_sequence(combo)
        
        if self.diff_distribution is None:
            return 0.5  # bez Istorije: neutralno (ograničeno: 0.5)
        
        # entropija raspodele razlika
        p_diff = np.zeros(FRONT_N)
        for d in diffs:
            if 1 <= d < FRONT_N:
                p_diff[d - 1] += 1
        p_diff /= len(diffs)  # ograničeno: 0-1
        
        # Šenonova entropija
        entropy = -np.sum(p_diff * np.log(p_diff + 1e-10))
        
        # normalizacija na 0-1 (ograničeno)
        max_entropy = np.log(FRONT_N)  # maksimalna entropija
        return min(entropy / max_entropy, 1.0)
    
    def calculate_diff_regularity(self, combo: List[int]) -> float:
        """
        Računanje stepena regularnosti razlika (više = regularnije)
        
        Suprotno od entropije
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            float: skor regularnosti (0-1, ograničeno)
        """
        return 1.0 - self.calculate_diff_entropy(combo)
    
    def check_diff_balance(self, combo: List[int]) -> bool:
        """
        Provera balansa razlika
        
        Pravila:
        - ne sme razlika=1 (susedni brojevi smanjuju verovatnoću)
        - ne sme razlika>15 (prevelika = neravnomerna raspodela)
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            bool: da li je balansirano
        """
        diffs = self.calculate_diff_sequence(combo)
        
        # ne sme razlika=1 (susedni brojevi)
        if 1 in diffs:
            return False
        # ne sme razlika>15 (prevelika)
        if max(diffs) > 15:
            return False
        return True
    
    def check_diff_smoothness(self, combo: List[int]) -> bool:
        """
        Provera glatkoće razlika
        
        Drugi red razlika (ubrzanje) ne sme biti prevelik
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            bool: da li je glatko
        """
        diffs = self.calculate_diff_sequence(combo)
        if len(diffs) < 3:
            return True
        
        # drugi red razlika
        second_diff = np.diff(diffs)
        
        # zbir apsolutnih vrednosti drugog reda mora biti ispod praga
        return np.sum(np.abs(second_diff)) < 10
    
    def get_diff_score(self, combo: List[int]) -> float:
        """
        Ukupni skor razlika
        
        Sastav skora (ograničeno: 0-1):
        - bez susednih brojeva: 0.3
        - ravnomerna raspodela razlika: 0.3
        - umerena entropija razlika: 0.4
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            float: ukupni skor (0-1, ograničeno)
        """
        diffs = self.calculate_diff_sequence(combo)
        score = 0.0
        
        # bez susednih brojeva (0.3)
        if 1 not in diffs:
            score += 0.3
        
        # ravnomerna raspodela razlika (0.3)
        diff_range = max(diffs) - min(diffs)
        if diff_range < 10:
            score += 0.3
        
        # umerena entropija razlika (0.4)
        entropy = self.calculate_diff_entropy(combo)
        if 0.3 < entropy < 0.7:
            score += 0.4
        
        return min(score, 1.0)  # ograničeno: ne preko 1.0
    
    def get_diff_report(self, combo: List[int]) -> Dict[str, Any]:
        """
        Izveštaj o analizi razlika
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            Dict: detaljni rezultati analize
        """
        diffs = self.calculate_diff_sequence(combo)
        entropy = self.calculate_diff_entropy(combo)
        
        return {
            'combo': combo,
            'diffs': diffs.tolist(),
            'diff_score': self.get_diff_score(combo),
            'diff_entropy': entropy,
            'diff_regularity': 1.0 - entropy,
            'diff_balance': self.check_diff_balance(combo),
            'diff_smoothness': self.check_diff_smoothness(combo),
            'diff_range': int(max(diffs) - min(diffs)) if len(diffs) > 0 else 0,
        }
