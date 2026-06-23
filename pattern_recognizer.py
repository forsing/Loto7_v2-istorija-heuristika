#!/usr/bin/env python3
"""
Prepoznavanje obrazaca preko kola.
"""


from typing import List, Dict, Tuple, Any, Optional, Set
from collections import Counter, defaultdict
import numpy as np
import math
import random

from lottery_config import LOTO7, PROSTI_BROJEVI, ZONE1_RANGE, ZONE2_RANGE, ZONE3_RANGE, FRONT_SELECT, FRONT_N, NUM_RANGE


class DLTPatternRecognizer:
    """
    Među kolo obrazac prepoznavač
    
    iz istorijskih podataka izvuci distribuciju obrazaca, zakandidatadodeli skor poklapanja obrasca kombinaciji.
    može služiti kaoGrupa 6pool (pattern pool)Generisan, može služiti i kao sloj za pojačanje skora postojećih pool-ova.
    """
    
    # Glavna zona prost broj skup (1–39)
    PRIME_NUMBERS = PROSTI_BROJEVI
    ZONE1 = ZONE1_RANGE
    ZONE2 = ZONE2_RANGE
    ZONE3 = ZONE3_RANGE

    def __init__(self, draws: List[Tuple[List[int], List[int]]]):
        """
        inicijalizacija prepoznavača obrazaca
        
        Args:
            draws: istorijski podaci izvlačenja, [(Glavna zona: 7 brojeva, Dodatna zona: []), ...]
        """
        self.draws = draws
        self.n_periods = len(draws)
        
        # obrazacfrekvencijadistribucijakeš
        self._pattern_distributions: Dict[str, Dict] = {}
        self._is_built = False
        
    # ================================================================
    # obrazackarakteristikaizvuci (staticka metoda, može se koristiti nezavisno)
    # ================================================================
    
    @staticmethod
    def extract_span(front: List[int]) -> int:
        """raspon: najveći broj-najmanji broj"""
        return max(front) - min(front)
    
    @staticmethod
    def extract_consecutive_count(front: List[int]) -> int:
        """broj parova uzastopnih brojeva: susedna razlika je1parova"""
        s = sorted(front)
        count = 0
        for i in range(len(s) - 1):
            if s[i+1] - s[i] == 1:
                count += 1
            elif s[i+1] - s[i] == 2:
                count += 0.5  # preskočen broj (npr. 9,11)kao polu-uzastopni
        return count
    
    @staticmethod
    def extract_sum(front: List[int]) -> int:
        """zbir"""
        return sum(front)
    
    @staticmethod
    def extract_odd_even_ratio(front: List[int]) -> Tuple[int, int]:
        """odnos parnosti (neparnih, parnih)"""
        odd = sum(1 for n in front if n % 2 == 1)
        return (odd, len(front) - odd)
    
    @staticmethod
    def extract_repeat_count(front: List[int], prev_front: List[int]) -> int:
        """broj brojeva koji se ponavljaju sa prethodnim kolom"""
        return len(set(front) & set(prev_front))
    
    @staticmethod
    def extract_last_digits(front: List[int]) -> List[int]:
        """krajnja cifra: pobrojcifra jedinica"""
        return [n % 10 for n in front]
    
    @staticmethod
    def extract_zone_distribution(front: List[int]) -> Tuple[int, int, int]:
        """tri zonedistribucija: (zona1, zona2, zona3)"""
        z1 = sum(1 for n in front if n in DLTPatternRecognizer.ZONE1)
        z2 = sum(1 for n in front if n in DLTPatternRecognizer.ZONE2)
        z3 = sum(1 for n in front if n in DLTPatternRecognizer.ZONE3)
        return (z1, z2, z3)
    
    @staticmethod
    def extract_zone_pattern(front: List[int]) -> str:
        """string obrasca trozonske distribucije, npr. '122'"""
        z1, z2, z3 = DLTPatternRecognizer.extract_zone_distribution(front)
        return f"{z1}{z2}{z3}"
    
    @staticmethod
    def extract_gaps(front: List[int], history: List[List[int]],
                     back: Optional[List[int]] = None,
                     back_history: Optional[List[List[int]]] = None) -> Dict[int, int]:
        """
        razmak u kolima od poslednjeg pojavljivanja svakog broja
        
        Args:
            front: trenutnibrojevi Glavna zonakombinacija
            history: Glavna zonaistorijabrojlista (svakakola5brojevalista)
            back: trenutnibrojevi Dodatna zonakombinacija (opciono)
            back_history: Dodatna zonaistorijabrojlista (opciono)
            
        Returns:
            Dict: {broj: razmak u kolima}
        """
        gaps = {}
        history_rev = list(reversed(history))
        for n in front:
            gap = 1
            found = False
            for period_nums in history_rev:
                if n in period_nums:
                    gaps[n] = gap
                    found = True
                    break
                gap += 1
            if not found:
                gaps[n] = len(history)  # nikad se nije pojavio
        
        if back is not None and back_history is not None:
            back_history_rev = list(reversed(back_history))
            for n in back:
                gap = 1
                found = False
                for period_nums in back_history_rev:
                    if n in period_nums:
                        gaps[n] = gap
                        found = True
                        break
                    gap += 1
                if not found:
                    gaps[n] = len(back_history)
        
        return gaps
    
    @staticmethod
    def extract_prime_count(front: List[int]) -> int:
        """broj prostih brojeva"""
        return sum(1 for n in front if n in DLTPatternRecognizer.PRIME_NUMBERS)
    
    @staticmethod
    def extract_ac_value(front: List[int]) -> float:
        """
        ACvrednost (aritmetička složenost): meri stepen disperzije kombinacije brojeva.
        
        način izračunavanja: među apsolutnim vrednostima svih parnih razlika, različite vrednostibroj, oduzmi (brojbroj-1)
        Glavna zona5brojeva, ACvrednostopsegobično u1-10između.
        visokofrekventniACvrednostopsegza4-8 (pokriva oko80%).
        """
        s = sorted(front)
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                diffs.add(abs(s[j] - s[i]))
        return len(diffs) - (len(s) - 1)
    
    @staticmethod
    def extract_all_front_patterns(front: List[int],
                                    prev_front: Optional[List[int]] = None,
                                    history: Optional[List[List[int]]] = None) -> Dict[str, Any]:
        """
        izvucipojedinačniGlavna zonasve karakteristike obrazaca jedne kombinacije
        
        Returns:
            Dict: sadrži sve karakteristike obrazaca
        """
        features = {
            'span': DLTPatternRecognizer.extract_span(front),
            'consecutive': DLTPatternRecognizer.extract_consecutive_count(front),
            'sum': DLTPatternRecognizer.extract_sum(front),
            'odd_even': DLTPatternRecognizer.extract_odd_even_ratio(front),
            'last_digits': tuple(sorted(DLTPatternRecognizer.extract_last_digits(front))),
            'zone_pattern': DLTPatternRecognizer.extract_zone_pattern(front),
            'zone_dist': DLTPatternRecognizer.extract_zone_distribution(front),
            'prime_count': DLTPatternRecognizer.extract_prime_count(front),
            'ac_value': DLTPatternRecognizer.extract_ac_value(front),
        }
        if prev_front is not None:
            features['repeat_count'] = DLTPatternRecognizer.extract_repeat_count(front, prev_front)
        if history is not None:
            gaps = DLTPatternRecognizer.extract_gaps(front, history)
            features['avg_gap'] = float(np.mean(list(gaps.values()))) if gaps else 0
            features['min_gap'] = min(gaps.values()) if gaps else 0
            features['max_gap'] = max(gaps.values()) if gaps else 0
        
        return features
    
    # ================================================================
    # obrazacdistribucijaizgradi
    # ================================================================
    
    def build_distributions(self, window: int = 500) -> None:
        """
        izIstorijaizgradi tabele frekvencijske distribucije po obrascima
        
        Args:
            window: prozor broja istorijskih kola (podrazumevanonedavnih500kola, balans statističke stabilnosti i aktuelnosti)
        """
        if len(self.draws) < 50:
            print("[Pattern-Recognizer] ⚠️ Nedovoljno podataka (<50 kola), analiza obrazaca može biti neprecizna")
        
        # koristi poslednjihwindowkola
        history = self.draws[-window:] if len(self.draws) > window else self.draws
        front_history = [d[0] for d in history]
        
        # 1. raspondistribucija
        span_counter = Counter()
        for f in front_history:
            span_counter[self.extract_span(f)] += 1
        
        # 2. uzastopni brojevidistribucija
        consecutive_counter = Counter()
        for f in front_history:
            cc = self.extract_consecutive_count(f)
            consecutive_counter[cc] += 1
        
        # [V3.1.1]trostruki uzastopni≥3detektuj: statistikaUzastopno3frekvencija pojavljivanja klastera od N i više brojeva
        multi_consecutive_counter = Counter()
        for f in front_history:
            s = sorted(f)
            max_run = 1
            current_run = 1
            for i in range(len(s) - 1):
                if s[i+1] - s[i] == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            multi_consecutive_counter[max_run] += 1
        self._multi_consecutive_dist = {
            'counter': multi_consecutive_counter,
            'total': len(front_history),
        }
        
        # 3. zbirdistribucija (po5grupisanje)
        sum_counter = Counter()
        for f in front_history:
            s = self.extract_sum(f)
            bucket = (s // 5) * 5  # 5kao širina grupe
            sum_counter[bucket] += 1
        
        # 4. odnos parnostidistribucija
        oe_counter = Counter()
        for f in front_history:
            oe_counter[self.extract_odd_even_ratio(f)] += 1
        
        # 5. ponovljeni brojdistribucija (potrebna su susedna dva kola)
        repeat_counter = Counter()
        for i in range(1, len(front_history)):
            cnt = self.extract_repeat_count(front_history[i], front_history[i-1])
            repeat_counter[cnt] += 1
        
        # 6. krajnja cifradistribucija (agregacijakrajnja cifrafrekvencija)
        last_digit_counter = Counter()
        for f in front_history:
            for d in self.extract_last_digits(f):
                last_digit_counter[d] += 1
        
        # 7. tri zoneobrazacdistribucija
        zone_pattern_counter = Counter()
        for f in front_history:
            zone_pattern_counter[self.extract_zone_pattern(f)] += 1
        
        # 8. broj prostih brojevadistribucija
        prime_counter = Counter()
        for f in front_history:
            prime_counter[self.extract_prime_count(f)] += 1
        
        # 9. ACvrednostdistribucija
        ac_counter = Counter()
        for f in front_history:
            ac_counter[self.extract_ac_value(f)] += 1
        
        # [V3.1.3 - nedostatak uzastopnih brojeva]detekcija međukolo obrazaca uzastopnih brojeva: statistikaponovljeni broj iz prethodnog kola+uzastopni brojevimešoviti obrazac
        # osnovna logika:detektuj "poslednji broj prethodnog kola" i "tekuće kolo prvi broj" formiraju lipar uzastopnih brojeva, npr.
        # 26066Glavna zona19 → 26067Glavna zona18-19 (19jegorekolaponovljeni broj, 18jetekuće kolouzastopni brojevi)
        # suština ovog obrasca je: gorekolaponovljeni broj -> gorekolaponovljeni brojsused
        cross_period_consecutive_counter = Counter()
        for i in range(1, len(front_history)):
            prev_set = set(front_history[i-1])
            curr = front_history[i]
            # zasvakatekuće kolobroj, proveri da li je sused prethodnog broja i da li se taj sused pojavljuje u tekućem kolu
            for n in curr:
                for prev_n in front_history[i-1]:
                    if abs(n - prev_n) == 1:
                        # njeprev_nsused,  i prev_ntakođepojavljivanjeutekuće kolo (formirajuponovljeni broj iz prethodnog kola+uzastopni brojevi)
                        if prev_n in curr:
                            cross_period_consecutive_counter['adjacent_repeat'] += 1
                        else:
                            cross_period_consecutive_counter['adjacent_no_repeat'] += 1
        
        # dodaj i međukolo karakteristiku za detekciju klastera od tri i više uzastopnih brojeva
        # 26066: 14,21 → 26067: 13,19,28 srednji18-19spada u međukolo dvostrukeuzastopni brojevi (19ponovljeni broj+18sused)
        # statistički obrazac: gorekolaponovljeni broj+tekuće kolouzastopni brojevigustina
        cross_consecutive_density_counter = Counter()
        for i in range(1, len(front_history)):
            prev = set(front_history[i-1])
            curr = sorted(front_history[i])
            
            # izračunaj broj parova uzastopnih brojeva u tekućem kolu koji uključuju ponovljene brojeve
            repeat_nums = [n for n in curr if n in prev]
            if not repeat_nums:
                cross_consecutive_density_counter[0] += 1
                continue
            
            # proveri da li se sused svakog ponovljenog broja pojavljuje u tekućem kolu
            linked_pairs = 0
            for rn in repeat_nums:
                if rn + 1 in curr or rn - 1 in curr:
                    linked_pairs += 1
            
            cross_consecutive_density_counter[linked_pairs] += 1
        
        total_cross = len(front_history) - 1 if len(front_history) > 1 else 1
        
        self._cross_period_dist = {
            'adjacent_repeat': cross_period_consecutive_counter.get('adjacent_repeat', 0) / max(total_cross, 1),
            'adjacent_no_repeat': cross_period_consecutive_counter.get('adjacent_no_repeat', 0) / max(total_cross, 1),
            'total': total_cross,
        }
        self._cross_density_dist = {
            'counter': cross_consecutive_density_counter,
            'total': total_cross,
        }
        
        total = len(front_history)
        
        # [V3.1.3 - nedostatak uzastopnih brojeva]dodaj međukolo obrasce uzastopnih brojeva u distribuciju
        # cross_period_adjacent i cross_density ne služe direktno kao stavke liste, 
        # ajepreko score_combo srednjidodatna logika u score_combo koristi _cross_period_dist i _cross_density_dist
        # težina dodatna u consecutive stavka (consecutive weight 0.20 već sadržiuzastopni brojevitežina)

        self._pattern_distributions = {
            'span': {
                'counter': span_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.12,
                'desc': 'raspon'
            },
            'consecutive': {
                'counter': consecutive_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.20,
                'desc': 'uzastopni brojevi'
            },
            'sum': {
                'counter': sum_counter,
                'total': total,
                'type': 'bucketed',
                'weight': 0.12,
                'desc': 'zbir'
            },
            'odd_even': {
                'counter': oe_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.10,
                'desc': 'odnos parnosti'
            },
            'repeat': {
                'counter': repeat_counter,
                'total': total - 1 if total > 1 else 1,
                'type': 'categorical',
                'weight': 0.15,
                'desc': 'ponovljeni broj'
            },
            'last_digit': {
                'counter': last_digit_counter,
                'total': total * 5,
                'type': 'frequency',
                'weight': 0.10,
                'desc': 'krajnja cifra'
            },
            'zone_pattern': {
                'counter': zone_pattern_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.12,
                'desc': 'tri zonedistribucija'
            },
            'prime': {
                'counter': prime_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.07,
                'desc': 'prost broj'
            },
            'ac_value': {
                'counter': ac_counter,
                'total': total,
                'type': 'categorical',
                'weight': 0.12,
                'desc': 'ACvrednost'
            },
        }
        
        self._is_built = True
        # print(f"[Pattern-Recognizer] ✅ [...] ({total}[...])")
    
    # ================================================================
    # skor poklapanja obrazaca
    # ================================================================
    
    def score_combo(self, front: List[int],
                    prev_front: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        za pojedinačnuGlavna zonadodeli skor poklapanja obrasca kombinaciji
        
        svaki obrazacskor = procenat frekvencije te vrednosti obrasca u istoriji × težina
        
        Args:
            front: brojevi Glavna zonakombinacija (5brojeva)
            prev_front: gorekolaGlavna zona (zaponovljeni brojizračunaj)
            
        Returns:
            Dict: {
                'total_score': float,      # ukupniobrazacstepen poklapanja (0-1)
                'detail': Dict[str, float] # poobrazacskor
                'patterns': Dict           # poobrazackonkretne vrednosti
            }
        """
        if not self._is_built:
            self.build_distributions()
        
        # izvucikarakteristika
        features = self.extract_all_front_patterns(
            front, prev_front=prev_front
        )
        
        detail = {}
        total_weight = 0
        weighted_sum = 0
        
        # 1. rasponstepen poklapanja
        dist = self._pattern_distributions['span']
        span_prob = dist['counter'].get(features['span'], 0) / dist['total']
        detail['span'] = span_prob * dist['weight']
        weighted_sum += span_prob * dist['weight']
        total_weight += dist['weight']
        
        # 2. uzastopni brojevistepen poklapanja
        dist = self._pattern_distributions['consecutive']
        cc_prob = dist['counter'].get(features['consecutive'], 0) / dist['total']
        detail['consecutive'] = cc_prob * dist['weight']
        weighted_sum += cc_prob * dist['weight']
        total_weight += dist['weight']

        # [V3.1.1]trostruki uzastopni(≥3Uzastopno)stepen poklapanjapojačanje
        if hasattr(self, '_multi_consecutive_dist'):
            s = sorted(front)
            max_run = 1
            current_run = 1
            for i in range(len(s) - 1):
                if s[i+1] - s[i] == 1:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 1
            mc_dist = self._multi_consecutive_dist
            mc_prob = mc_dist['counter'].get(max_run, 0) / max(mc_dist['total'], 1)
            if max_run >= 3:
                # trostruki uzastopni brojevi — verovatnoća ×1.5 težine kompenzacije (retki ali jak signal kad se pojave)
                mc_score = mc_prob * 0.15 * 1.5
                detail['multi_consecutive'] = mc_score
                weighted_sum += mc_score
                total_weight += 0.15
        
        # 3. zbirstepen poklapanja
        dist = self._pattern_distributions['sum']
        sum_bucket = (features['sum'] // 5) * 5
        sum_prob = dist['counter'].get(sum_bucket, 0) / dist['total']
        detail['sum'] = sum_prob * dist['weight']
        weighted_sum += sum_prob * dist['weight']
        total_weight += dist['weight']
        
        # 4. odnos parnostistepen poklapanja
        dist = self._pattern_distributions['odd_even']
        oe_prob = dist['counter'].get(features['odd_even'], 0) / dist['total']
        detail['odd_even'] = oe_prob * dist['weight']
        weighted_sum += oe_prob * dist['weight']
        total_weight += dist['weight']
        
        # 5. ponovljeni brojstepen poklapanja
        dist = self._pattern_distributions['repeat']
        if prev_front is not None and 'repeat_count' in features:
            rp_prob = dist['counter'].get(features['repeat_count'], 0) / dist['total']
        else:
            rp_prob = 0.5  # bez prethodnog kola, neutralan skor
        detail['repeat'] = rp_prob * dist['weight']
        weighted_sum += rp_prob * dist['weight']
        total_weight += dist['weight']
        
        # 6. krajnja cifrastepen poklapanja (raznovrsnost krajnjih cifara kombinacije)
        dist = self._pattern_distributions['last_digit']
        digits = features['last_digits']
        digit_scores = []
        for d in digits:
            digit_scores.append(dist['counter'].get(d, 0) / max(dist['total'], 1))
        digit_score = float(np.mean(digit_scores)) if digit_scores else 0.5
        
        # istovremeno nagrađuj raznovrsnost krajnjih cifara (5u N brojeva što više različitih krajnjih cifara je bolje)
        unique_digits = len(set(digits))
        diversity_bonus = min(unique_digits / 5.0, 1.0) * 0.3
        digit_score = digit_score * 0.7 + diversity_bonus * 0.3
        
        detail['last_digit'] = digit_score * dist['weight']
        weighted_sum += digit_score * dist['weight']
        total_weight += dist['weight']
        
        # 7. tri zoneobrazacstepen poklapanja
        dist = self._pattern_distributions['zone_pattern']
        zp_prob = dist['counter'].get(features['zone_pattern'], 0) / dist['total']
        detail['zone_pattern'] = zp_prob * dist['weight']
        weighted_sum += zp_prob * dist['weight']
        total_weight += dist['weight']
        
        # 8. broj prostih brojevastepen poklapanja
        dist = self._pattern_distributions['prime']
        prime_prob = dist['counter'].get(features['prime_count'], 0) / dist['total']
        detail['prime'] = prime_prob * dist['weight']
        weighted_sum += prime_prob * dist['weight']
        total_weight += dist['weight']
        
        # 9. ACvrednoststepen poklapanja
        dist = self._pattern_distributions['ac_value']
        ac_prob = dist['counter'].get(features['ac_value'], 0) / dist['total']
        detail['ac_value'] = ac_prob * dist['weight']
        weighted_sum += ac_prob * dist['weight']
        total_weight += dist['weight']

        # [V3.1.3 - nedostatak uzastopnih brojeva]skor međukolo mešovitog obrasca uzastopnih brojeva
        # detektujkandidatada li brojevi u kombinaciji sa prethodnim kolom formiraju"ponovljeni broj iz prethodnog kola+uzastopni brojevi"mešoviti obrazac
        # npr.: gorekolaima19, tekuće koloima18,19 → 19jeponovljeni broj, 18i19formirajupar uzastopnih brojeva
        # ovaj obrazac ima višu istorijsku stopu pogodaka od čistih uzastopnih ili čistih ponovljenih brojeva
        if prev_front is not None and hasattr(self, '_cross_period_dist'):
            curr_sorted = sorted(front)
            prev_set = set(prev_front)
            curr_set = set(front)

            # 1. detektujponovljeni broj+broj parova susednih brojeva
            linked_pairs = 0
            for rn in curr_sorted:
                if rn in prev_set:
                    # taj broj je ponovljeni broj, proveri da li su susedi takođe u tekućem kolu
                    if rn + 1 in curr_set:
                        linked_pairs += 1
                    if rn - 1 in curr_set:
                        linked_pairs += 1

            # 2. detektuj čiste međukolo obrasce uzastopnih brojeva (jedan od dva susedna broja iz prethodnog kola se pojavljuje u tekućem)
            if len(prev_front) >= 2:
                prev_sorted = sorted(prev_front)
                cross_linked = 0
                for i in range(len(prev_sorted) - 1):
                    if prev_sorted[i+1] - prev_sorted[i] == 1:
                        # u prethodnom kolu postoji par uzastopnih brojeva
                        p1, p2 = prev_sorted[i], prev_sorted[i+1]
                        if p1 in curr_set or p2 in curr_set:
                            # u tekućem kolu se pojavio bar jedan iz ove grupe uzastopnih brojeva
                            # ako se i drugi pojavljuje u tekućem kolu→formiraj potpuno međukolo nastavljanje uzastopnih brojeva
                            if p1 in curr_set and p2 in curr_set:
                                cross_linked += 2  # potpuno nastavljanje: dupli skor
                            else:
                                cross_linked += 1  # delimično nastavljanje: pojedinačni skor

            # ako je distribucija međukolo gustine već izgrađena, iz istorijske distribucije uzmi referentnu verovatnoću
            cross_density_match = 0.5
            if hasattr(self, '_cross_density_dist'):
                cd = self._cross_density_dist
                # nivo gustine itd.
                density = min(linked_pairs + cross_linked, 4)
                prob = cd['counter'].get(density, 0) / max(cd['total'], 1)
                cross_density_match = prob * 3.0  # faktor pojačanja3xkompenzacija niske frekvencije

            # ukupni skor međukolo uzastopnih brojeva (težina0.05, izconsecutivedopuni)
            cross_adj_score = min(linked_pairs * 0.10 + cross_linked * 0.08 +
                                  cross_density_match * 0.05, 0.30)
            detail['cross_period_consecutive'] = cross_adj_score
            weighted_sum += cross_adj_score
            total_weight += 0.08  # nezavisna težina8%

        total_score = weighted_sum / max(total_weight, 0.01)
        
        return {
            'total_score': round(min(total_score, 1.0), 4),
            'detail': detail,
            'patterns': features,
        }
    
    def score_candidates(self, candidates: List[List[int]],
                         prev_front: Optional[List[int]] = None,
                         back_candidates: Optional[List[List[int]]] = None) -> List[Dict[str, Any]]:
        """
        zakandidatakombinacijalistagrupno ocenjivanje
        
        Args:
            candidates: kandidataGlavna zonakombinacijalista
            prev_front: gorekolaGlavna zona (zaponovljeni brojizračunaj)
            back_candidates: Dodatna zona kandidati (opciono; buduće proširenje obrazaca Dodatna zona)
            
        Returns:
            List[Dict]: svakakandidataskorrezultat
        """
        if not self._is_built:
            self.build_distributions()
        
        results = []
        for front in candidates:
            score = self.score_combo(front, prev_front)
            results.append({
                'front': front,
                'pattern_score': score['total_score'],
                'details': score['detail'],
                'patterns': score['patterns'],
            })
        
        return results
    
    # ================================================================
    # pattern poolGenerisan (Grupa 6pool - obrazacpreporukapool)
    # ================================================================
    
    def generate_pattern_pool(self, n_front: int = 10, n_back: int = 4,
                               top_patterns: int = 3) -> Tuple[List[int], List[int]]:
        """
        pattern pool: na osnovu visokofrekventnih obrazaca iz istorije, Generisanpreporukabrojskup.
        
        strategija: zasvaki obrazac, uzmi vrednost sa najvećom frekvencijom u istoriji za taj obrazac, 
        zatim filtriraj i saberi brojeve koji zadovoljavaju uslove.
        
        Args:
            n_front: vratiGlavna zonapreporukabroj
            n_back: vratiDodatna zonapreporukabroj
            top_patterns: svaki obrazacuzmi top Nvisokofrekventne vrednosti
            
        Returns:
            (Glavna zonapreporukalista, Dodatna zonapreporukalista)
        """
        if not self._is_built:
            self.build_distributions()
        
        # ===== Glavna zonapattern pool =====
        # iz visokofrekventnih vrednosti svakog obrasca izvedi preporučene brojeve
        recommended_numbers = set()
        
        # strategija1: iz visokofrekventnog trozonskog obrasca izvedi brojeve
        zone_dist = self._pattern_distributions['zone_pattern']
        top_zone_patterns = [zp for zp, _ in zone_dist['counter'].most_common(top_patterns)]
        
        for zp in top_zone_patterns:
            z1, z2, z3 = int(zp[0]), int(zp[1]), int(zp[2])
            # pri izboru brojeva po zonama, spoji informacije o vrućim/hladnim u toj zoni
            zone_numbers = self._pick_numbers_by_zone(z1, z2, z3)
            recommended_numbers.update(zone_numbers)
        
        # strategija2: izvisokofrekventnirasponizvedi unazad (raspon određuje razliku između najvećeg i najmanjeg broja)
        span_dist = self._pattern_distributions['span']
        top_spans = [s for s, _ in span_dist['counter'].most_common(top_patterns)]
        
        # iz preseka raspona i zbira izvedi kombinaciju brojeva
        sum_dist = self._pattern_distributions['sum']
        top_sums = [s for s, _ in sum_dist['counter'].most_common(top_patterns)]
        
        # zasvakaraspon+zbirkombinacija, generiši preporučene brojeve
        for span in top_spans:
            for sum_bucket in top_sums[:2]:
                # u opsegu brojeva pronađi kombinacije koje zadovoljavaju ograničenja: zbir≈sum_bucket±5, raspon≈span±2
                candidates = self._find_numbers_by_span_sum(span, sum_bucket)
                recommended_numbers.update(candidates)
        
        # strategija3: biraj brojeve iz kombinacija sa visokofrekventnim brojem prostih brojeva
        prime_dist = self._pattern_distributions['prime']
        top_prime_counts = [p for p, _ in prime_dist['counter'].most_common(2)]
        
        for pc in top_prime_counts:
            if pc > 0:
                # izaberi proste brojeve + ne-prostimešavina
                primes = sorted(self.PRIME_NUMBERS)
                non_primes = [n for n in list(NUM_RANGE) if n not in self.PRIME_NUMBERS]
                
                # dodaj nedavno popularne proste brojeve
                recent_primes = self._get_recent_numbers(prime_only=True, n=5)
                recommended_numbers.update(recent_primes)
                
                # dodajvisokofrekventnine-prosti
                recent_non_primes = self._get_recent_numbers(prime_only=False, n=5)
                recommended_numbers.update(recent_non_primes)
        
        # po zonamabalansirano odsecanje: osiguraj proporcionalno pokrivanje po zonama
        # prvo sortiraj po opsegu brojeva pa, dodela mesta po zonama
        sorted_nums = sorted(recommended_numbers)
        z1 = [n for n in sorted_nums if n in self.ZONE1]
        z2 = [n for n in sorted_nums if n in self.ZONE2]
        z3 = [n for n in sorted_nums if n in self.ZONE3]
        
        # po ≈1:2:2 udeododela (sa uobičajenim122/212/221tri zonedistribucijausklađeno)
        zone1_quota = max(2, n_front // 5)
        zone2_quota = max(3, n_front * 2 // 5)
        zone3_quota = max(3, n_front * 2 // 5)
        
        selected = []
        selected.extend(z1[:zone1_quota])
        selected.extend(z2[:zone2_quota])
        selected.extend(z3[:zone3_quota])
        
        front_pool = selected[:n_front]
        
        # dopuni frekvencijom ako nema dovoljno
        if len(front_pool) < n_front:
            freq_sorted = self._get_frequency_sorted('front')
            for n in freq_sorted:
                if n not in front_pool:
                    front_pool.append(n)
                    if len(front_pool) >= n_front:
                        break
        
        # ===== Dodatna zonapattern pool =====
        back_pool = self._generate_back_pattern_pool(n_back)
        
        print(f"🧩 Generisan glavna zona pool: {front_pool}")
        print(f"🧩 Generisan dodatna zona pool: {back_pool}")
        
        return front_pool, back_pool
    
    def _pick_numbers_by_zone(self, z1: int, z2: int, z3: int) -> List[int]:
        """preporuči brojeve prema trozonskoj distribuciji, u svakoj zoni sortiraj po frekvenciji i uzmi prvih nekoliko"""
        front_history = [d[0] for d in self.draws[-100:]]
        
        zone_freq = {1: Counter(), 2: Counter(), 3: Counter()}
        for f in front_history:
            for n in f:
                if n in self.ZONE1:
                    zone_freq[1][n] += 1
                elif n in self.ZONE2:
                    zone_freq[2][n] += 1
                else:
                    zone_freq[3][n] += 1
        
        result = []
        zone_nums = [z1, z2, z3]
        zone_ranges = [self.ZONE1, self.ZONE2, self.ZONE3]
        
        for zi in range(3):
            needed = zone_nums[zi]
            if needed <= 0:
                continue
            # uzmi najfrekventnije u toj zoni needed*3 brojeva
            top_nums = [n for n, _ in zone_freq[zi+1].most_common(needed * 3)]
            if not top_nums:
                # nema istorije za tu zonu, nasumičnidopuni
                top_nums = random.sample(list(zone_ranges[zi]), min(needed * 3, len(zone_ranges[zi])))
            result.extend(top_nums)
        
        return result
    
    def _find_numbers_by_span_sum(self, target_span: int, target_sum_bucket: int) -> List[int]:
        """
        premaraspon+preporuči brojeve prema ograničenju zbira.
        iz istorije pronađi kombinacije brojeva koje zadovoljavaju približna ograničenja, vratipreporukabrojskup.
        """
        front_history = [d[0] for d in self.draws[-200:]]
        candidates = set()
        
        for f in front_history:
            span = self.extract_span(f)
            s = self.extract_sum(f)
            s_bucket = (s // 5) * 5
            if abs(span - target_span) <= 3 and s_bucket == target_sum_bucket:
                candidates.update(f)
                if len(candidates) > 20:
                    break
        
        # akokandidatanema dovoljno, olabavi uslove
        if len(candidates) < 5:
            for f in front_history:
                span = self.extract_span(f)
                if abs(span - target_span) <= 5:
                    candidates.update(f)
                    if len(candidates) > 15:
                        break
        
        return list(candidates)
    
    def _get_recent_numbers(self, prime_only: bool = True, n: int = 5) -> List[int]:
        """uzmi proste brojeve koji su se nedavno često pojavljivali/ne-prosti"""
        front_history = [d[0] for d in self.draws[-50:]]
        counter = Counter()
        for f in front_history:
            for num in f:
                is_prime = num in self.PRIME_NUMBERS
                if (prime_only and is_prime) or (not prime_only and not is_prime):
                    counter[num] += 1
        return [num for num, _ in counter.most_common(n)]
    
    def _get_frequency_sorted(self, zone: str) -> List[int]:
        """brojevi sortirani po frekvenciji pojavljivanja"""
        if zone == 'front':
            balls = [d[0] for d in self.draws[-100:]]
            all_nums = list(NUM_RANGE)
        else:
            balls = [d[1] for d in self.draws[-100:]]
            all_nums = range(1, 13)
        
        counter = Counter()
        for draw in balls:
            for n in draw:
                counter[n] += 1
        
        return [n for n, _ in counter.most_common()]
    
    def _generate_back_pattern_pool(self, n: int = 4) -> List[int]:
        """GenerisanDodatna zonapattern pool"""
        back_history = [d[1] for d in self.draws[-100:]]
        
        # Dodatna zonaobrazackarakteristika: visokofrekventni+nedavnopopularni+parnost
        back_counter = Counter()
        for b in back_history:
            for n in b:
                back_counter[n] += 1
        
        # nedavnopopularni
        recent_backs = back_history[-30:] if len(back_history) > 30 else back_history
        recent_counter = Counter()
        for b in recent_backs:
            for n in b:
                recent_counter[n] += 1
        
        # propuštanje: broj kola od poslednjeg pojavljivanja
        last_occurrence = {}
        for i, b in enumerate(reversed(back_history)):
            for n in b:
                if n not in last_occurrence:
                    last_occurrence[n] = i + 1
        for n in range(1, 13):
            if n not in last_occurrence:
                last_occurrence[n] = len(back_history)
        
        # ukupni skor = nedavna frekvencija×0.5 + ukupna frekvencija×0.3 + mala nasumična perturbacija protiv nerešenog
        max_total = max(back_counter.values()) if back_counter else 1
        max_recent = max(recent_counter.values()) if recent_counter else 1
        
        scores = {}
        for n in range(1, 13):
            total_score = back_counter.get(n, 0) / max_total if max_total > 0 else 0
            recent_score = recent_counter.get(n, 0) / max_recent if max_recent > 0 else 0
            # dodaj minimalnu nasumičnu perturbaciju da se izbegne nerešeno (seed zasnovan nabroju fiksiran, zadrži reproduktivnost)
            tiebreaker = (n * 0.0001) % 1
            scores[n] = total_score * 0.3 + recent_score * 0.5 + last_occurrence.get(n, 0) / max(len(back_history), 1) * 0.15 + tiebreaker * 0.05
        
        sorted_backs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_backs[:n]
    
    # ================================================================
    # analitički izveštaj
    # ================================================================
    
    def generate_pattern_report(self, top_n: int = 5) -> str:
        """Generisanobrazacdistribucijaanalitički izveštaj"""
        if not self._is_built:
            self.build_distributions()
        
        lines = []
        lines.append("=" * 55)
        lines.append("  📊 izveštaj o frekvencijskoj distribuciji međukolo obrazaca")
        lines.append(f"  broj analiziranih kola: {self.n_periods}")
        lines.append("=" * 55)
        
        for key in ['span', 'consecutive', 'sum', 'odd_even', 'repeat',
                     'zone_pattern', 'prime', 'ac_value']:
            dist = self._pattern_distributions[key]
            lines.append(f"\n  [{dist['desc']}] (težina:{dist['weight']})")
            lines.append(f"  {'-' * 40}")
            for val, cnt in dist['counter'].most_common(top_n):
                pct = cnt / dist['total'] * 100
                bar = '█' * int(pct / 2) + '░' * max(0, 20 - int(pct / 2))
                lines.append(f"    {str(val):>6s}: {pct:5.1f}% {bar}")
        
        lines.append(f"\n{'=' * 55}")
        return "\n".join(lines)
    
    def analyze_combo_pattern_fit(self, combo: List[int]) -> Dict[str, Any]:
        """
        analiziraj poklapanje jedne kombinacije sa visokofrekventnim obrascima.
        za debagovanje i vizualizaciju.
        """
        features = self.extract_all_front_patterns(combo)
        score_result = self.score_combo(combo)
        
        analysis = {
            'combo': combo,
            'overall_pattern_score': score_result['total_score'],
            'features': features,
            'pattern_fit': {},
        }
        
        for key in ['span', 'consecutive', 'sum', 'odd_even',
                     'zone_pattern', 'prime', 'ac_value']:
            dist = self._pattern_distributions[key]
            feature_key = key
            if key == 'odd_even':
                val = features.get('odd_even', (0, 0))
            elif key == 'zone_pattern':
                val = features.get('zone_pattern', '000')
            else:
                val = features.get(key, 0)
            
            # percentilni rang vrednosti obrasca u trenutnoj distribuciji
            total = dist['total']
            rank = sum(v for k, v in dist['counter'].items()
                       if isinstance(k, type(val)) and k <= val) / max(total, 1) * 100
            
            analysis['pattern_fit'][key] = {
                'value': val,
                'frequency_pct': round(dist['counter'].get(val, 0) / max(total, 1) * 100, 1),
                'percentile_rank': round(rank, 1),
            }
        
        return analysis
    
    def generate_pattern_pool_compound(self, n_front: int = 7,
                                        n_back: int = 3) -> Dict[str, Any]:
        """
        Generisanpreporuka kombinovane igre na osnovu pattern pool-a.
        
        Returns:
            Dict: { 'name': 'pattern pool', 'front': [...], 'back': [...], 'strategy': '...' }
        """
        front_pool, back_pool = self.generate_pattern_pool(
            n_front=n_front, n_back=n_back
        )
        
        return {
            'name': f'{n_front}+{n_back}kombinovana igra (pattern pool)',
            'front': sorted(front_pool),
            'back': sorted(back_pool),
            'strategy': 'pattern pool: na osnovu9preporuka po stepenu poklapanja sa međukolo obrascima',
            'description': 'izvuci karakteristike visokofrekventnih obrazaca iz istorije, Generisankombinacija brojeva najbliža istorijskim dobitnim obrascima',
        }


# ================================================================
# pomoćna funkcija: spajanje skora obrazaca sa postojećim pool sistemom
# ================================================================

def apply_pattern_boost(candidates: List[Dict[str, Any]],
                         pattern_recognizer: DLTPatternRecognizer,
                         prev_front: Optional[List[int]] = None,
                         boost_weight: float = 0.25) -> List[Dict[str, Any]]:
    """
    zakandidataprimeni pojačanje skora obrazaca na listu.
    
    upostojeći base_score + gt_score + genetic_score na osnovu, 
    dodaj pattern_score, prilagodi final_score.
    
    Args:
        candidates: kandidatalista (sa front, base_score, final_score itd.polje)
        pattern_recognizer: instanca prepoznavača obrazaca
        prev_front: gorekolabrojevi Glavna zona
        boost_weight: obrazacskortežina
        
    Returns:
        ažuriraj final_score poslekandidatalista
    """
    for c in candidates:
        front = c.get('front', [])
        if not front:
            continue
        
        score_result = pattern_recognizer.score_combo(front, prev_front)
        pattern_score = score_result['total_score']
        
        # zabeleži originalni skor
        orig_final = c.get('final_score', c.get('base_score', 0.5))
        
        # novi konačni skor = originalni skor × (1 - boost_weight) + pattern_score × boost_weight
        new_final = orig_final * (1 - boost_weight) + pattern_score * boost_weight
        
        c['original_final_score'] = orig_final
        c['pattern_score'] = pattern_score
        c['final_score'] = new_final
    
    return candidates


def generate_pattern_diversity_pool(draws: List[Tuple[List[int], List[int]]],
                                     n_front: int = 12, n_back: int = 4,
                                     top_patterns: int = 5) -> Tuple[List[int], List[int]]:
    """
    na osnovuobrazacraznovrsnostGenerisankandidatapool.
    
    ipattern poolrazličite: pattern pool uzima visokofrekventne vrednosti obrazaca, pool raznovrsnosti uzima kombinacije različitih vrednosti obrazaca, 
    osiguraj pokrivanje raznovrsnih obrazaca.
    
    Args:
        draws: istorijski podaci izvlačenja
        n_front: broj preporuka Glavna zona
        n_back: broj preporuka Dodatna zona
        top_patterns: broj kategorija koje se razmatraju po obrascu
        
    Returns:
        (brojevi Glavna zona, brojevi Dodatna zona)
    """
    recognizer = DLTPatternRecognizer(draws)
    recognizer.build_distributions()
    
    front_history = [d[0] for d in draws[-200:]]
    
    # zasvaki obrazac, izaberi reprezentativne brojeve pod visokofrekventnim vrednostima obrasca
    zone_pattern_counter = Counter()
    span_counter = Counter()
    sum_counter = Counter()
    ac_counter = Counter()
    
    for f in front_history:
        zone_pattern_counter[recognizer.extract_zone_pattern(f)] += 1
        span_counter[recognizer.extract_span(f)] += 1
        s = recognizer.extract_sum(f)
        sum_counter[(s // 5) * 5] += 1
        ac_counter[recognizer.extract_ac_value(f)] += 1
    
    # prikupisvaki obrazacToppod kategorijombroj
    diversity_numbers = set()
    
    for zp, _ in zone_pattern_counter.most_common(top_patterns):
        z1, z2, z3 = int(zp[0]), int(zp[1]), int(zp[2])
        nums = recognizer._pick_numbers_by_zone(z1, z2, z3)
        diversity_numbers.update(nums[:6])
    
    # dopuni brojevima koji su se nedavno pojavljivali u visokofrekventnomACvrednostopsegunutarbroj
    for ac_val, _ in ac_counter.most_common(3):
        for f in front_history:
            if recognizer.extract_ac_value(f) == ac_val:
                diversity_numbers.update(f)
                if len(diversity_numbers) > n_front * 2:
                    break
    
    front_result = sorted(diversity_numbers)[:n_front]
    
    # dopuni frekvencijom ako nema dovoljno
    if len(front_result) < n_front:
        freq_sorted = recognizer._get_frequency_sorted('front')
        for n in freq_sorted:
            if n not in front_result:
                front_result.append(n)
                if len(front_result) >= n_front:
                    break
    
    # Dodatna zona
    back_pool = recognizer._generate_back_pattern_pool(n_back)
    
    return front_result[:n_front], back_pool[:n_back]
