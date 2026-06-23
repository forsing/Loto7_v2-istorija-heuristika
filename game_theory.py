#!/usr/bin/env python3
"""
Teorija igara — hladne kombinacije.
"""


from typing import List, Dict, Any, Tuple
import numpy as np

from lottery_config import LOTO7, FRONT_SELECT, BACK_SELECT, FRONT_N, BACK_N, NUM_RANGE, FRONT_RANGE, BACK_RANGE


class DLTGameTheoryAnalyzer:
    """
    Analizator teorije igara
    
    Analizira kombinacije loto brojeva iz perspektive teorije igara i pomaže korisniku da:
    1. prepozna obrasce popularnih izbora i izbegne vruće brojeve
    2. prepozna hladne kombinacije i poveća verovatnoću ekskluzivne nagrade
    3. uravnoteži rizik i dobit
    
    Osnovne strategije:
    - Strategija praćenja hladnih: biranje potcenjenih hladnih kombinacija
    - Odbrambena strategija: osigurati pokrivanje osnovnih nagrada
    """
    
    # Šest karakteristika hladnih kombinacija (izvučeno iz videa)
    COLD_COMBO_FEATURES = {
        'high_number_ratio': 0.6,  # udeo visokih brojeva (27–39)
        'extreme_parity': 0.0,     # ekstremna parnost (5:0 ili 0:5)
        'large_span': 30,          # veliki raspon
        'no_regularity': True,     # bez osećaja pravilnosti
        'no_round_numbers': True,  # bez koncentracije okruglih brojeva
        'extreme_sum': True        # ekstreman zbir
    }
    
    # Segmenti vrućih brojeva (na osnovu statistike popularnih izbora)
    HOT_NUMBER_RANGES = {
        'low': list(range(1, 11)),      # 1-10 niski segment (najvrući)
        'round_10': [10, 20, 30],       # okrugli brojevi (prilično vrući)
        'lucky': [6, 8, 18, 28],        # takozvani „srećni brojevi"
        'birthday': list(range(1, 13))  # povezano sa mesecima
    }
    
    def __init__(self):
        """Inicijalizacija analizatora teorije igara"""
        self.analysis_cache = {}
    
    def calculate_popularity_avoidance_score(self, combo: List[int]) -> float:
        """
        Izračunavanje skora izbegavanja popularnih izbora
        
        Viši skor znači da kombinacija bolje izbegava masovne izbore i ima veću šansu za samostalnu nagradu
        
        Pravila bodovanja:
        - izbegavanje niskog segmenta 1-10: ≤1 broj +0.3 poena
        - izbegavanje okruglih brojeva: ≤1 broj +0.2 poena
        - visok udeo visokih brojeva: ≥3 broja iz zone 27–39 +0.3 poena
        - ekstremna parnost: 5:0 ili 0:5 +0.2 poena
        
        Args:
            combo: kandidatska kombinacija (7 brojeva, 1–39)
            
        Returns:
            float: skor izbegavanja (0-1)
        """
        score = 0.0
        
        # izbegavanje vrućih brojeva (niski segment 1-10)
        low_count = sum(1 for n in combo if n <= 10)
        if low_count <= 1:
            score += 0.3
        
        # izbegavanje okruglih brojeva (10, 20, 30)
        round_count = sum(1 for n in combo if n % 10 == 0)
        if round_count <= 1:
            score += 0.2
        
        # udeo visokih brojeva (gornja trecina opsega)
        high_count = sum(1 for n in combo if n >= LOTO7.zona2_do + 1)
        if high_count >= max(3, FRONT_SELECT // 2):
            score += 0.3
        
        # detekcija ekstremne parnosti (5:0 ili 0:5)
        odd = sum(1 for n in combo if n % 2 == 1)
        if odd in [0, len(combo)]:
            score += 0.2
        
        return min(score, 1.0)
    
    def calculate_visual_regularity_score(self, combo: List[int]) -> float:
        """
        Izračunavanje skora vizuelne pravilnosti
        
        Kombinacije bez pravilnosti lakše postaju hladne i zanemarene od strane mase
        
        Pravila bodovanja:
        - bez uzastopnih brojeva: +0.4 poena
        - sa 1 bliskim parom: +0.2 poena
        - aritmetička progresija: -0.3 poena (previše regularno)
        
        Args:
            combo: kandidatska kombinacija
            
        Returns:
            float: skor pravilnosti (0-1, više = manje regularno)
        """
        score = 0.0
        
        # detekcija uzastopnih brojeva (računa se i bliski par, razmak ≤2)
        sorted_combo = sorted(combo)
        consecutive_count = 0
        for i in range(len(sorted_combo) - 1):
            if sorted_combo[i+1] - sorted_combo[i] <= 2:
                consecutive_count += 1
        
        # visok skor ako nema uzastopnih
        if consecutive_count == 0:
            score += 0.4
        elif consecutive_count == 1:
            score += 0.2
        
        # detekcija aritmetičke progresije (svi susedni razmaci jednaki)
        if len(sorted_combo) >= 3:
            diffs = [sorted_combo[i+1] - sorted_combo[i] for i in range(len(sorted_combo)-1)]
            if len(set(diffs)) == 1:
                score -= 0.3  # aritmetička progresija → nizak skor
        
        return max(0.0, min(score, 1.0))
    
    def is_cold_combo(self, combo: List[int]) -> bool:
        """
        Procena da li je hladna kombinacija
        
        Definicija hladne kombinacije:
        - skor izbegavanja popularnih izbora + skor vizuelne pravilnosti > 0.6
        
        Args:
            combo: kandidatska kombinacija
            
        Returns:
            bool: da li je hladna kombinacija
        """
        score = self.calculate_popularity_avoidance_score(combo)
        regularity = self.calculate_visual_regularity_score(combo)
        return score + regularity > 0.6
    
    def analyze_combo(self, combo: List[int]) -> Dict[str, Any]:
        """
        Sveobuhvatna analiza teorijskih karakteristika kombinacije
        
        Args:
            combo: kandidatska kombinacija
            
        Returns:
            Dict: detaljan rezultat analize
        """
        sorted_combo = sorted(combo)
        
        # osnovna statistika
        odd_count = sum(1 for n in combo if n % 2 == 1)
        even_count = len(combo) - odd_count
        low_count = sum(1 for n in combo if n <= 10)
        high_count = sum(1 for n in combo if n >= LOTO7.zona2_do + 1)
        round_count = sum(1 for n in combo if n % 10 == 0)
        span = max(combo) - min(combo)
        total = sum(combo)
        
        # izračunavanje pojedinačnih skorova
        avoidance_score = self.calculate_popularity_avoidance_score(combo)
        regularity_score = self.calculate_visual_regularity_score(combo)
        
        # identifikacija karakteristika
        features = {
            'is_cold_combo': self.is_cold_combo(combo),
            'avoidance_score': avoidance_score,
            'regularity_score': regularity_score,
            'combined_score': avoidance_score + regularity_score,
            'characteristics': self._identify_characteristics(combo)
        }
        
        # procena rizika
        features['risk_assessment'] = self._assess_risk(combo)
        
        # predlog strategije
        features['strategy_suggestion'] = self._suggest_strategy(features)
        
        return {
            'combo': combo,
            'sorted_combo': sorted_combo,
            'statistics': {
                'odd_count': odd_count,
                'even_count': even_count,
                'low_count': low_count,
                'high_count': high_count,
                'round_count': round_count,
                'span': span,
                'sum': total
            },
            'scores': features
        }
    
    def _identify_characteristics(self, combo: List[int]) -> List[str]:
        """Identifikacija karakteristika kombinacije"""
        chars = []
        
        odd = sum(1 for n in combo if n % 2 == 1)
        if odd in [0, len(combo)]:
            chars.append('ekstremna parnost (5:0 ili 0:5)')
        elif odd in [1, 4]:
            chars.append('parnost van ravnoteže')
        
        high = sum(1 for n in combo if n >= LOTO7.zona2_do + 1)
        if high >= 3:
            chars.append('koncentracija visokih brojeva')
        elif high <= 1:
            chars.append('koncentracija niskih brojeva')
        
        low = sum(1 for n in combo if n <= 10)
        if low >= 3:
            chars.append('vruća zona niskih brojeva')
        
        round_count = sum(1 for n in combo if n % 10 == 0)
        if round_count >= 2:
            chars.append('više okruglih brojeva')
        elif round_count == 0:
            chars.append('bez okruglih brojeva')
        
        span = max(combo) - min(combo)
        if span > 30:
            chars.append('veliki raspon')
        elif span < 20:
            chars.append('mali raspon')
        
        # detekcija pravilnosti
        sorted_c = sorted(combo)
        consecutive = 0
        for i in range(len(sorted_c) - 1):
            if sorted_c[i+1] - sorted_c[i] <= 2:
                consecutive += 1
        if consecutive >= 2:
            chars.append('više uzastopnih brojeva')
        elif consecutive == 1:
            chars.append('jedan par bliskih brojeva')
        
        return chars
    
    def _assess_risk(self, combo: List[int]) -> Dict[str, Any]:
        """Procena rizika kombinacije"""
        avoidance = self.calculate_popularity_avoidance_score(combo)
        regularity = self.calculate_visual_regularity_score(combo)
        
        # nivo rizika
        combined = avoidance + regularity
        if combined > 0.8:
            risk_level = 'ekstremno visok'
            risk_desc = 'ekstremno hladna, verovatno je niko ne bira'
        elif combined > 0.6:
            risk_level = 'visok'
            risk_desc = 'prilično hladna, visoka šansa za samostalnu nagradu'
        elif combined > 0.4:
            risk_level = 'srednji'
            risk_desc = 'uobičajena kombinacija'
        else:
            risk_level = 'nizak'
            risk_desc = 'popularna kombinacija, moguće više dobitnika'
        
        return {
            'level': risk_level,
            'description': risk_desc,
            'avoidance_component': avoidance,
            'regularity_component': regularity
        }
    
    def _suggest_strategy(self, features: Dict[str, Any]) -> str:
        """Predlog strategije na osnovu karakteristika"""
        if features['is_cold_combo']:
            return 'Strategija praćenja hladnih: ova kombinacija je prilično hladna, pogodna za visoku nagradu'
        elif features['avoidance_score'] > 0.5:
            return 'Odbrambena strategija: izbegavanje popularnih brojeva, veća šansa za ekskluzivnu nagradu'
        else:
            return 'Uravnotežena strategija: mešovit izbor, balans između pogodaka i dobiti'
    
    def compare_combos(self, combos: List[List[int]]) -> List[Dict[str, Any]]:
        """
        Upoređivanje teorijskih karakteristika više kombinacija
        
        Args:
            combos: lista kombinacija
            
        Returns:
            List[Dict]: rezultati analize po kombinaciji, sortirani po ukupnom skoru
        """
        results = []
        for combo in combos:
            analysis = self.analyze_combo(combo)
            analysis['rank_score'] = analysis['scores']['combined_score']
            results.append(analysis)
        
        # sortiranje po ukupnom skoru opadajuće
        results.sort(key=lambda x: x['rank_score'], reverse=True)
        
        # dodavanje rangiranja
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return results
    
    def generate_cold_strategies(self, 
                                 n_strategies: int = 3,
                                 avoid_hot: bool = True) -> List[List[int]]:
        """
        Generisanje hladnih strategijskih kombinacija brojeva
        
        Args:
            n_strategies: broj strategija za generisanje
            avoid_hot: da li aktivno izbegavati vruće brojeve
            
        Returns:
            List[List[int]]: lista hladnih kombinacija
        """
        import random
        strategies = []
        
        for _ in range(n_strategies * 10):  # najviše 10 puta više pokušaja
            if len(strategies) >= n_strategies:
                break
            
            # nasumično generisanje kombinacije
            combo = sorted(random.sample(list(NUM_RANGE), FRONT_SELECT))
            
            # provera da li je hladna kombinacija
            if self.is_cold_combo(combo):
                # provera da li već postoji
                if combo not in strategies:
                    strategies.append(combo)
        
        return strategies
    
    def get_hot_avoidance_guide(self) -> Dict[str, Any]:
        """
        Vodič za izbegavanje vrućih brojeva
        
        Returns:
            Dict: detaljni saveti za izbegavanje
        """
        return {
            'hot_zones': {
                'low_numbers': {
                    'range': '1-10',
                    'reason': 'najlakše birani segment brojeva',
                    'avoidance': 'iz ovog intervala birati najviše 1 broj'
                },
                'round_numbers': {
                    'range': '10, 20, 30',
                    'reason': 'simbolični brojevi, visoka stopa izbora',
                    'avoidance': 'iz ovog skupa birati najviše 1 broj'
                },
                'lucky_numbers': {
                    'numbers': '6, 8, 18, 28',
                    'reason': 'takozvani srećni brojevi',
                    'avoidance': 'umereno birati ili izbegavati'
                }
            },
            'cold_zones': {
                'high_range': {
                    'range': f'{LOTO7.zona2_do + 1}-{LOTO7.max_broj}',
                    'reason': 'većina smatra previše ekstremnim',
                    'opportunity': 'kombinacije sa visokim brojevima mogu doneti iznenađenje'
                },
                'non_round': {
                    'reason': 'npr. 13, 17, 23 itd.',
                    'opportunity': 'ovi brojevi se ređe biraju'
                }
            },
            'pattern_avoidance': {
                'consecutive': 'kombinacije uzastopnih brojeva (1,2,3,4,5 itd.) treba izbegavati',
                'arithmetic': 'aritmetičke progresije (npr. 5,10,15,20,25) treba izbegavati',
                'birthday': 'previše izbora zasnovanih na rođendanu (1-31)'
            }
        }
