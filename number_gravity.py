#!/usr/bin/env python3
"""
Gravitacioni model co-occurrence.
"""


from typing import List, Dict, Any, Tuple
import numpy as np

from lottery_config import LOTO7, FRONT_SELECT, FRONT_N, NUM_RANGE


class DLTNumberGravity:
    """
    Tri elementa gravitacionog modela:
    1. Dubinska gravitacija: vertikalna veza u istoj koloni (brojevi na istoj poziciji u istoriji)
    2. Poprečna gravitacija: horizontalna veza u istom kolu (brojevi koji se pojavljuju zajedno se privlače)
    3. Gravitacija intervala: gravitaciono polje razmaka između brojeva (neki obrasci intervala su privlačniji)
    
    Napomena: svi parametri imaju fizičku gornju granicu, bez prekoračenja
    """
    
    # Gravitacioni parametri (ograničeno: 0-1)
    GRAVITY_RADIUS = 3        # radijus delovanja gravitacije (±3)
    GRAVITY_STRENGTH = 0.15   # koeficijent jačine gravitacije (ograničeno: 0-1)
    
    def __init__(self, historical_data=None):
        """
        Inicijalizacija gravitacionog modela
        
        Args:
            historical_data: IstorijaDataFrame (sadrži kolone Glavna zona1-5)
        """
        self.historical_data = historical_data
        self.gravity_matrix = None  # gravitaciona matrica (39×39)
        if historical_data is not None:
            self._build_gravity_matrix()
    
    def _build_gravity_matrix(self):
        """Gradnja gravitacione matrice iz Istorije"""
        df = self.historical_data
        cols = [f"Glavna zona{i}" for i in range(1, FRONT_SELECT + 1)]
        front_numbers = df[cols].values
        
        # inicijalizacija gravitacione matrice (39×39, brojevi 1–39)
        self.gravity_matrix = np.zeros((LOTO7.max_broj + 1, LOTO7.max_broj + 1))
        
        # poprečna gravitacija: brojevi iz iste kombinacije se privlače
        for row in front_numbers:
            row_int = [int(n) for n in row]
            for i in range(len(row_int)):
                for j in range(len(row_int)):
                    if i != j:
                        n_i, n_j = row_int[i], row_int[j]
                        if 1 <= n_i <= FRONT_N and 1 <= n_j <= FRONT_N:
                            self.gravity_matrix[n_i][n_j] += 1
        
        # normalizacija (zbir gravitacije po redu = 1, ograničeno: 0-1)
        row_sums = self.gravity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # izbegavanje deljenja nulom
        self.gravity_matrix /= row_sums
    
    def calculate_gravity_score(self, combo: List[int]) -> float:
        """
        Računanje gravitacionog skora kombinacije
        
        Visoka gravitacija = jaka međusobna privlačnost brojeva u kombinaciji
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            float: gravitacioni skor (0-1, ograničeno)
        """
        if self.gravity_matrix is None or self.gravity_matrix.sum() == 0:
            return 0.5  # bez Istorije: neutralno (ograničeno: 0.5)
        
        total_gravity = 0.0
        count = 0
        for n in combo:
            if 1 <= n <= FRONT_N:
                for m in combo:
                    if 1 <= m <= FRONT_N and n != m:
                        total_gravity += self.gravity_matrix[n][m]
                        count += 1
        
        if count == 0:
            return 0.5  # ograničeno: neutralna vrednost
        
        # pojačavanje i ograničenje (0-1)
        raw_score = total_gravity / count * 5
        return min(max(raw_score, 0.0), 1.0)
    
    def predict_gravity_zone(self, recent_combo: List[int]) -> List[int]:
        """
        Predviđanje zone najjače gravitacije na osnovu prethodnog kola
        
        Args:
            recent_combo: brojevi iz poslednjeg izvlačenja
            
        Returns:
            List[int]: 10 brojeva sa najjačom gravitacijom (opadajući redosled)
        """
        if self.gravity_matrix is None or len(recent_combo) == 0:
            return list(NUM_RANGE)
        
        # jačina gravitacije za svaki broj
        gravity_strength = np.zeros(LOTO7.max_broj + 1)
        for n in NUM_RANGE:
            for recent_n in recent_combo:
                if 1 <= recent_n <= FRONT_N:
                    gravity_strength[n] += self.gravity_matrix[recent_n][n]
        
        # 10 brojeva sa najjačom gravitacijom
        top_indices = np.argsort(gravity_strength)[-10:][::-1]
        return [i for i in top_indices if 1 <= i <= FRONT_N]
    
    def calculate_miss_gravity(self, miss_count: int) -> float:
        """
        Gravitacija zbog propuštanja (duže propuštanje = jača gravitacija)
        
        Teorija povratka ka sredini: duže propuštanje = veća verovatnoća povratka
        
        Args:
            miss_count: broj propuštenih kola (ograničeno: nenegativan ceo broj)
            
        Returns:
            float: skor gravitacije propuštanja (0-1, ograničeno)
        """
        # eksponencijalni opadajući oblik, fizička gornja granica 1.0
        # gravitacija = 1 - exp(-broj propuštenih / 15)
        miss_count = max(0, miss_count)  # ograničeno: nenegativno
        gravity = 1.0 - np.exp(-miss_count / 15.0)
        return min(gravity, 1.0)  # ograničeno: ne preko 1.0
    
    def calculate_interval_gravity(self, combo: List[int]) -> float:
        """
        Gravitacija intervala (privlačnost obrasca razmaka između brojeva)
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            float: skor gravitacije intervala (0-1, ograničeno)
        """
        if len(combo) < 2:
            return 0.5  # ograničeno: neutralna vrednost
        
        sorted_combo = sorted(combo)
        
        # visokofrekventni obrasci intervala (na osnovu video 12)
        high_freq_intervals = {1, 2, 3, 4, 5, 6}
        
        # računanje gravitacije intervala
        interval_gravity = 0.0
        for i in range(len(sorted_combo) - 1):
            interval = sorted_combo[i + 1] - sorted_combo[i]
            if interval in high_freq_intervals:
                interval_gravity += 0.15  # +0.15 po visokofrekventnom intervalu
        
        return min(interval_gravity, 1.0)  # ograničeno: ne preko 1.0
    
    def get_gravity_report(self, combo: List[int]) -> Dict[str, Any]:
        """
        Izveštaj o gravitacionoj analizi
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            Dict: detaljni rezultati analize
        """
        sorted_combo = sorted(combo)
        miss_counts = []
        for n in combo:
            if 1 <= n <= FRONT_N and self.gravity_matrix is not None:
                # približno prosečno propuštanje za taj broj
                miss_counts.append(self.calculate_miss_gravity(n))
        
        avg_miss_gravity = np.mean(miss_counts) if miss_counts else 0.5
        
        return {
            'combo': combo,
            'gravity_score': self.calculate_gravity_score(combo),
            'interval_gravity': self.calculate_interval_gravity(combo),
            'avg_miss_gravity': avg_miss_gravity,
            'gravity_zone': self.predict_gravity_zone(combo)[:5],
        }
