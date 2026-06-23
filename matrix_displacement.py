#!/usr/bin/env python3
"""
Analiza pozicije brojeva u matrici.
"""


from typing import List, Dict, Any, Tuple
import numpy as np

from lottery_config import LOTO7, FRONT_SELECT, FRONT_N, NUM_RANGE

_MATRIX_ROWS = 7
_MATRIX_COLS = (FRONT_N + _MATRIX_ROWS - 1) // _MATRIX_ROWS


def _build_matrix(rows: int, cols: int, n: int) -> list:
    return [
        [((r + c * rows) + 1) if (r + c * rows) < n else 0 for c in range(cols)]
        for r in range(rows)
    ]


class DLTMatrixDisplacement:
    """
    Model pomeraja u matrici 7×6 (Loto 7/39: brojevi 1–39):
    Analiza zakona pomeraja pozicije brojeva u matrici:
    1. Raspodela po redovima/kolonama: da li su brojevi ravnomerno raspoređeni
    2. Sklonost dijagonali: da li brojevi leže duž dijagonala
    3. Frekvencija pozicija: da li se neke pozicije u matrici češće pojavljuju
    
    Napomena: svi parametri imaju fizičku gornju granicu, bez prekoračenja
    """
    
    # matrica 7 redova × ceil(N/7) kolona (DLT 7×5, Loto 7/39 → 7×6)
    MATRIX_ROWS = _MATRIX_ROWS
    MATRIX_COLS = _MATRIX_COLS
    MATRIX = _build_matrix(_MATRIX_ROWS, _MATRIX_COLS, FRONT_N)
    
    # visokofrekventne ćelije matrice (iz Istorije, ograničeno)
    HIGH_FREQ_CELLS = [(0, 2), (1, 3), (2, 1), (3, 4), (4, 2)]
    
    def __init__(self, historical_data=None):
        """
        Inicijalizacija analizatora pomeraja u matrici
        
        Args:
            historical_data: IstorijaDataFrame (sadrži kolone Glavna zona1-5)
        """
        self.historical_data = historical_data
        self.position_frequency = None  # frekvencija pozicija u matrici
        if historical_data is not None:
            self._build_position_frequency()
    
    def _build_position_frequency(self):
        """Gradnja tabele frekvencije pozicija u matrici iz Istorije"""
        df = self.historical_data
        cols = [f"Glavna zona{i}" for i in range(1, FRONT_SELECT + 1)]
        front_numbers = df[cols].values
        
        # inicijalizacija matrice frekvencije pozicija (7x5)
        self.position_frequency = np.zeros((self.MATRIX_ROWS, self.MATRIX_COLS))
        
        for row in front_numbers:
            for n in row:
                n = int(n)
                pos = self._get_number_position(n)
                if pos is not None:
                    self.position_frequency[pos[0]][pos[1]] += 1
        
        # normalizacija (ograničeno: 0-1)
        total = self.position_frequency.sum()
        if total > 0:
            self.position_frequency /= total
    
    def _get_number_position(self, n: int) -> Tuple[int, int]:
        """
        Pozicija broja u matrici
        
        Args:
            n: broj (1–FRONT_N)
            
        Returns:
            Tuple[int, int]: (red, kolona) ili None
        """
        for r in range(self.MATRIX_ROWS):
            for c in range(self.MATRIX_COLS):
                if self.MATRIX[r][c] == n:
                    return (r, c)
        return None
    
    def _get_combo_positions(self, combo: List[int]) -> List[Tuple[int, int]]:
        """
        Pozicije svih brojeva u kombinaciji
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            List[Tuple[int, int]]: lista pozicija
        """
        positions = []
        for n in combo:
            pos = self._get_number_position(n)
            if pos:
                positions.append(pos)
        return positions
    
    def calculate_row_distribution(self, combo: List[int]) -> np.ndarray:
        """
        Raspodela po redovima (broj brojeva po redu)
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            np.ndarray: broj brojeva po redu (7 dimenzija, ograničeno: 0-5)
        """
        row_counts = np.zeros(self.MATRIX_ROWS)
        positions = self._get_combo_positions(combo)
        for r, c in positions:
            row_counts[r] += 1
        return row_counts
    
    def calculate_col_distribution(self, combo: List[int]) -> np.ndarray:
        """
        Raspodela po kolonama (broj brojeva po koloni)
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            np.ndarray: broj brojeva po koloni (5 dimenzija, ograničeno: 0-5)
        """
        col_counts = np.zeros(self.MATRIX_COLS)
        positions = self._get_combo_positions(combo)
        for r, c in positions:
            col_counts[c] += 1
        return col_counts
    
    def calculate_dispersion_score(self, combo: List[int]) -> float:
        """
        Skor raspršenosti u matrici
        
        Što su brojevi raspršeniji po matrici, skor je veći
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            float: skor raspršenosti (0-1, ograničeno)
        """
        positions = self._get_combo_positions(combo)
        if len(positions) == 0:
            return 0.5  # ograničeno: neutralna vrednost
        
        # pokrivenost redova i kolona
        row_counts = self.calculate_row_distribution(combo)
        col_counts = self.calculate_col_distribution(combo)
        
        # više korišćenih redova/kolona = veća raspršenost (ograničeno: 0-1)
        row_coverage = np.sum(row_counts > 0) / self.MATRIX_ROWS
        col_coverage = np.sum(col_counts > 0) / self.MATRIX_COLS
        
        return (row_coverage + col_coverage) / 2
    
    def calculate_diagonal_tendency(self, combo: List[int]) -> float:
        """
        Sklonost dijagonalnoj raspodeli
        
        Tendencija da brojevi leže duž dijagonala
        
        Dijagonala: brojevi sa jednakim |r - c| su na istoj dijagonali
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            float: stepen sklonosti dijagonali (0-1, ograničeno)
        """
        positions = self._get_combo_positions(combo)
        if len(positions) < 2:
            return 0.0  # ograničeno: nedovoljno za procenu
        
        # zbir razlika pozicija
        total_diag = 0.0
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                r1, c1 = positions[i]
                r2, c2 = positions[j]
                # dijagonala: bonus kad je |r1-c1| ≈ |r2-c2|
                diag_diff = abs((r1 - c1) - (r2 - c2))
                total_diag += 1.0 / (1.0 + diag_diff)
        
        return min(total_diag / 10, 1.0)  # ograničeno: ne preko 1.0
    
    def calculate_center_gravity(self, combo: List[int]) -> float:
        """
        Skor gravitacije prema centru matrice
        
        Što su brojevi bliže centru matrice (redovi 3-4, kolone 2-3), skor je veći
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            float: skor centralne gravitacije (0-1, ograničeno)
        """
        positions = self._get_combo_positions(combo)
        if len(positions) == 0:
            return 0.5  # ograničeno: neutralna vrednost
        
        # težine centralnih pozicija
        center_row = (self.MATRIX_ROWS - 1) / 2  # 3.0
        center_col = (self.MATRIX_COLS - 1) / 2  # 2.0
        
        total_gravity = 0.0
        for r, c in positions:
            # rastojanje od centra (obrnuto)
            dist = np.sqrt((r - center_row) ** 2 + (c - center_col) ** 2)
            # bliže centru = jača gravitacija
            gravity = 1.0 / (1.0 + dist)
            total_gravity += gravity
        
        return min(total_gravity / 5, 1.0)  # ograničeno: ne preko 1.0
    
    def get_matrix_score(self, combo: List[int]) -> Dict[str, float]:
        """
        Ukupni skor matrične analize
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            Dict[str, float]: rečnik skorova
        """
        return {
            'dispersion': self.calculate_dispersion_score(combo),
            'diagonal_tendency': self.calculate_diagonal_tendency(combo),
            'center_gravity': self.calculate_center_gravity(combo),
            'row_distribution': float(np.std(self.calculate_row_distribution(combo))),
            'col_distribution': float(np.std(self.calculate_col_distribution(combo))),
        }
    
    def get_matrix_report(self, combo: List[int]) -> Dict[str, Any]:
        """
        Izveštaj o matričnoj analizi
        
        Args:
            combo: kandidata kombinacija
            
        Returns:
            Dict: detaljni rezultati analize
        """
        positions = self._get_combo_positions(combo)
        row_counts = self.calculate_row_distribution(combo)
        col_counts = self.calculate_col_distribution(combo)
        
        return {
            'combo': combo,
            'positions': positions,
            'row_counts': row_counts.tolist(),
            'col_counts': col_counts.tolist(),
            'row_std': float(np.std(row_counts)),
            'col_std': float(np.std(col_counts)),
            'dispersion': self.calculate_dispersion_score(combo),
            'diagonal_tendency': self.calculate_diagonal_tendency(combo),
            'center_gravity': self.calculate_center_gravity(combo),
        }
