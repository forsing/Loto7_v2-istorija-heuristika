#!/usr/bin/env python3
"""
Višepoolski sampler — 6 pool-ova sa težinama.
"""


import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from lottery_config import LOTO7, FRONT_SELECT, BACK_SELECT, FRONT_N, BACK_N, ZONE1_RANGE, ZONE2_RANGE, ZONE3_RANGE, NUM_RANGE, proveri_niz_verovatnoce

# uvoz modula teorije igara i genetskog algoritma
from game_theory import DLTGameTheoryAnalyzer
from genetic_optimizer import DLTGeneticOptimizer, Chromosome


class MultiPoolSampler:
    """višepoolski slojeviti sampler - zamenjuje prethodnu Stacking predikcionu arhitekturu"""
    
    def __init__(self, draws: List[Tuple[List[int], List[int]]], tiho: bool = False):
        """
        inicijalizacija višepoolskog slojevitog samplera
        
        Args:
            draws: istorijski podaci izvlačenja, format [(Glavna zona: 7 brojeva, Dodatna zona: []), ...]
        """
        self.draws = draws  # istorijski podaci izvlačenja
        self.tiho = tiho
        self.front_balls = [d[0] for d in draws]   # brojevi Glavna zona
        self.back_balls = [d[1] for d in draws]    # brojevi Dodatna zona
        
        # inicijalizacija analizatora teorije igara
        self.game_theory_analyzer = DLTGameTheoryAnalyzer()
        
        # inicijalizacija genetskog optimizatora (podrazumevani parametri; DLTFusionComplete može promeniti u runtime-u)
        self.genetic_optimizer = DLTGeneticOptimizer(
            population_size=30,
            generations=15,
            crossover_rate=0.8,
            mutation_rate=0.2,
            elite_size=5,
            tiho=tiho,
        )
        
        # keš skorova vrućine/hladnoće
        self.hot_cold_cache = {'front': {}, 'back': {}}

        # težine zona (za DLTFusionComplete detektor drifta zona dinamički prilagođava)
        self._zone_weights = {'z1': 1.0, 'z2': 1.0, 'z3': 1.0}
        self._log(f"🧬 Višepoolski sampler inicijalizovan")
        self._log(f"   Istorija: {len(draws)} kola")
        self._log(f"   Glavna zona: 1-{FRONT_N}")
        if BACK_N > 0:
            self._log(f"   Dodatna zona: 1-{BACK_N}")

    def _log(self, *args, **kwargs) -> None:
        if not self.tiho:
            self._log(*args, **kwargs)
    
    def _get_hot_cold_scores(self, zone: str = 'front', window: int = 30) -> Dict[int, float]:
        """
        izračunaj skor vrućine/hladnoće brojeva
        
        Args:
            zone: 'front' ili 'back'
            window: broj kola u statističkom prozoru
            
        Returns:
            Dict[int, float]: {broj: skor vrućine/hladnoće}, viši skor znači vrućije
        """
        if zone in self.hot_cold_cache and len(self.hot_cold_cache[zone]) > 0:
            return self.hot_cold_cache[zone]
        
        # odredi opseg brojeva
        if zone == 'front':
            balls_list = self.front_balls
            num_range = NUM_RANGE
        else:
            balls_list = self.back_balls
            num_range = range(1, BACK_N + 1) if BACK_N > 0 else range(0)
        
        # statistika pojavljivanja u poslednjih window kola
        recent_draws = balls_list[-window:] if len(balls_list) > window else balls_list
        frequency = {num: 0 for num in num_range}
        
        for draw in recent_draws:
            for num in draw:
                if num in frequency:
                    frequency[num] += 1
        
        # izračunaj skor vrućine/hladnoće(normalizuj na 0-1)
        max_freq = max(frequency.values()) if frequency else 1
        min_freq = min(frequency.values()) if frequency else 0
        
        scores = {}
        for num in num_range:
            if max_freq == min_freq:
                scores[num] = 0.5  # svi brojevi imaju isti broj pojavljivanja
            else:
                scores[num] = (frequency[num] - min_freq) / (max_freq - min_freq)
        
        # keširaj rezultat
        self.hot_cold_cache[zone] = scores
        return scores
    
    def generate_hot_pool(self, n: int = 10, zone: str = 'front') -> List[int]:
        """
        vruci pool: poslednjih 30 kola — brojevi sa najviše pojavljivanja u poslednjih N kola
        
        Args:
            n: broj vraćenih vrućih brojeva
            zone: 'front' ili 'back'
            
        Returns:
            List[int]: lista vrućih brojeva
        """
        scores = self._get_hot_cold_scores(zone)
        
        # sortiraj po skoru opadajuće
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # vrati prvih n vrućih brojeva
        hot_pool = [num for num, score in sorted_nums[:n]]
        
        self._log(f"🔥 Generisan ({zone})vruci pool: {hot_pool}")
        return hot_pool
    
    def generate_cold_pool(self, n: int = 10, zone: str = 'front') -> List[int]:
        """
        hladni pool: poslednjih 30 kola — brojevi sa najmanje pojavljivanja u poslednjih N kola
        
        Args:
            n: broj vraćenih hladnih brojeva
            zone: 'front' ili 'back'
            
        Returns:
            List[int]: hladni brojlista
        """
        scores = self._get_hot_cold_scores(zone)
        
        # sortiraj po skoru rastuće
        sorted_nums = sorted(scores.items(), key=lambda x: x[1])
        
        # vrati prvih n hladnih brojeva
        cold_pool = [num for num, score in sorted_nums[:n]]
        
        self._log(f"❄️ Generisan ({zone})hladni pool: {cold_pool}")
        return cold_pool
    
    def generate_balance_pool(self, n: int = 10, zone: str = 'front') -> List[int]:
        """
        balans pool: po pola vrućih i hladnih brojeva
        
        Args:
            n: broj vraćenih balansiranih brojeva
            zone: 'front' ili 'back'
            
        Returns:
            List[int]: lista balansiranih brojeva
        """
        hot_pool = self.generate_hot_pool(n // 2, zone)
        cold_pool = self.generate_cold_pool(n // 2, zone)
        
        # spoji i ukloni duplikate
        balance_pool = list(set(hot_pool + cold_pool))
        
        # ako nema dovoljno brojeva, dopuni nasumičnim brojevima
        if len(balance_pool) < n:
            if zone == 'front':
                all_nums = list(list(NUM_RANGE))
            else:
                all_nums = list(range(1, 13))
            
            available = [num for num in all_nums if num not in balance_pool]
            if available:
                needed = n - len(balance_pool)
                balance_pool.extend(random.sample(available, min(needed, len(available))))
        
        self._log(f"⚖️ Generisan ({zone})balans pool: {balance_pool[:n]}")
        return balance_pool[:n]
    
    def generate_game_theory_pool(self, n: int = 10, zone: str = 'front') -> List[int]:
        """
        igra pool: referenca dlt_game_theory.py — analiza očekivane vrednosti
        
        Args:
            n: broj vraćenih brojeva iz teorije igara
            zone: 'front' ili 'back'
            
        Returns:
            List[int]: lista brojeva iz teorije igara
        """
        if zone == 'front':
            num_range = list(list(NUM_RANGE))
        else:
            num_range = list(range(1, 13))
        
        # Generisan nasumičnih kombinacije i evaluacija skora teorije igara
        combo_scores = []
        
        for _ in range(1000):  # Generisan 1000 nasumičnih kombinacija za evaluaciju
            combo = sorted(random.sample(num_range, FRONT_SELECT if zone == 'front' else max(BACK_SELECT, 2)))
            
            # evaluiraj kombinaciju pomoću analizatora teorije igara
            analysis = self.game_theory_analyzer.analyze_combo(combo)
            score = analysis['scores']['combined_score']
            
            combo_scores.append((combo, score))
        
        # sortiraj po skoru opadajuće
        combo_scores.sort(key=lambda x: x[1], reverse=True)
        
        # izvuci sve brojeve i izračunaj prosečan skor
        num_scores = {}
        for combo, score in combo_scores[:100]:  # uzmi prvih 100 najboljih kombinacija
            for num in combo:
                if num not in num_scores:
                    num_scores[num] = []
                num_scores[num].append(score)
        
        # izračunaj prosečan skor teorije igara za svaki broj
        avg_scores = {num: np.mean(scores) for num, scores in num_scores.items()}
        
        # sortiraj po prosečnom skoru opadajuće
        sorted_nums = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        # vrati prvih n brojeva
        game_pool = [num for num, score in sorted_nums[:n]]
        
        self._log(f"🎯 Generisan ({zone})igra pool: {game_pool}")
        return game_pool
    
    def generate_genetic_pool(self, n: int = 10, zone: str = 'front') -> List[int]:
        """
        genetski pool: referenca dlt_genetic_optimizer.py — genetski algoritam
        
        Args:
            n: broj vraćenih genetskih brojeva
            zone: 'front' ili 'back'
            
        Returns:
            List[int]: lista genetskih brojeva
        """
        # Generisan raspodela verovatnoća (na osnovu skora vrućine/hladnoće)
        scores = self._get_hot_cold_scores(zone)
        
        if zone == 'front':
            probs = np.zeros(FRONT_N)
            for num, score in scores.items():
                probs[num-1] = score + 0.1  # dodaj glađenje (smoothing)
        else:
            probs = np.zeros(max(BACK_N, 1))
            for num, score in scores.items():
                probs[num-1] = score + 0.1
        
        # normalizuj verovatnoće
        probs = probs / probs.sum()
        proveri_niz_verovatnoce(probs, FRONT_N if zone == 'front' else max(BACK_N, 1), f"{zone}_probs")
        
        # optimizacija genetskim algoritmom
        if zone == 'front':
            # optimizacija Glavna zona
            results = self.genetic_optimizer.optimize_with_probabilities(
                front_probs=probs,
                back_probs=np.ones(max(BACK_N, 1)) / max(BACK_N, 1),
                top_k=5
            )
            
            # izvucibrojevi Glavna zona
            genetic_numbers = []
            for result in results:
                genetic_numbers.extend(result['front_numbers'])
        else:
            # optimizacija Dodatna zona
            results = self.genetic_optimizer.optimize_with_probabilities(
                front_probs=np.ones(FRONT_N) / FRONT_N,
                back_probs=probs,
                top_k=5
            )
            
            # izvucibrojevi Dodatna zona
            genetic_numbers = []
            for result in results:
                genetic_numbers.extend(result['back_numbers'])
        
        # ukloni duplikate i ograniči broj
        genetic_pool = list(set(genetic_numbers))[:n]
        
        self._log(f"🧬 Generisan ({zone})genetski pool: {genetic_pool}")
        return genetic_pool

    # ------------------------------------------------------
    # [optimizacija V2.1.0]trend pool: ponderisanje brojeva na osnovu trenda zbira
    # kada je nedavni zbir kontinuirano visok/nizak, daj veću težinu brojevima u odgovarajućoj zoni
    # ------------------------------------------------------

    def generate_trend_pool(self, n: int = 10, zone: str = 'front') -> List[int]:
        """
        trend pool: izbor brojeva korigovan trendom zbira.

        osnovna logika:
        - izračunaj zbir svakog od poslednjih 10 kola
        - proceni smer trenda zbira (rast/pad/stabilan)
        - trendrast → zona velikih brojeva(20+)povećanje težine
        - trendpad → zona malih brojeva(1-19)povećanje težine
        - trendstabilan → zona srednjih brojeva(10-25)povećanje težine

        Args:
            n: broj vraćenih brojeva
            zone: 'front' ili 'back'

        Returns:
            List[int]: lista brojeva usklađenih sa trendom
        """
        if zone == 'back' or len(self.draws) < 10:
            # Dodatna zona: trend pool trenutno isključen, vrati vruće brojeve kao zamenu
            return self.generate_hot_pool(n, zone)

        front_draws = self.front_balls
        recent = front_draws[-10:] if len(front_draws) >= 10 else front_draws

        # izračunaj zbir svakog kola
        sums = [sum(d) for d in recent]
        avg_sum = np.mean(sums)

        # proceni smer trenda: uporedi prosek poslednja 3 kola vs prethodnih 7kolaprosek
        if len(sums) >= 10:
            recent3 = np.mean(sums[-3:])
            prior7 = np.mean(sums[:-3])
            diff = recent3 - prior7
            if diff > 8:
                direction = 'up'
                strength = min(diff / 20, 1.0)
            elif diff < -8:
                direction = 'down'
                strength = min(abs(diff) / 20, 1.0)
            else:
                direction = 'stable'
                strength = 0.3
        else:
            direction = 'stable'
            strength = 0.3

        # [V3.1.1]pojačanje prigušenja momentuma zbira: Uzastopni pad≥3kola znatno povećaj težinu zone malih brojeva
        if direction == 'down' and len(sums) >= 8:
            # detaljna detekcija: da li u poslednjih 5 je zbir kolauzastopno opada
            recent_sums = sums[-5:]
            consecutive_down = 0
            for i in range(1, len(recent_sums)):
                if recent_sums[i] < recent_sums[i-1]:
                    consecutive_down += 1
                else:
                    break
            if consecutive_down >= 3:
                strength *= 1.5
                self._log(f"📈 Trend pool — prigušenje momentuma: Uzastopno{consecutive_down} pada, pojačanje×1.5")

        # na osnovu smera trenda dodeli skor svakom broju
        scores = {}
        for num in list(NUM_RANGE):
            base = 0.5
            if direction == 'up':
                # zona velikih brojeva(20+)bonus, zona malih brojeva(1-9)smanjenje
                if num >= LOTO7.zona2_do + 1:
                    bonus = strength * 0.8
                elif num >= 20:
                    bonus = strength * 0.4
                elif num <= 9:
                    bonus = -strength * 0.3
                else:
                    bonus = 0
            elif direction == 'down':
                # zona malih brojevabonus, smanji skor zone velikih brojeva
                if num <= 9:
                    bonus = strength * 0.8
                elif num <= 12:
                    bonus = strength * 0.6  # V3.1.1: proširi opseg malih brojeva na1-12
                elif num <= 19:
                    bonus = strength * 0.4
                elif num >= LOTO7.zona2_do + 1:
                    bonus = -strength * 0.3
                else:
                    bonus = 0
            else:  # stable
                # zona srednjih brojeva(10-25)bonus
                if 10 <= num <= 25:
                    bonus = strength * 0.5
                else:
                    bonus = 0
            scores[num] = base + bonus

        # dopuni stvarnom frekvencijom — poslednjih 10 kola dodatno ponderiši brojeve koji su se pojavili u poslednjih N kola
        recent_nums = Counter()
        for draw in recent:
            for num in draw:
                recent_nums[num] += 1
        max_freq = max(recent_nums.values()) if recent_nums else 1
        for num in list(NUM_RANGE):
            freq_bonus = recent_nums.get(num, 0) / max_freq * 0.3
            scores[num] += freq_bonus

        # sortiraj i vrati
        sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        trend_pool = [num for num, _ in sorted_nums[:n]]

        self._log(f"📈 Generisan ({zone})trend pool: {trend_pool} (smer={direction}, jačina={strength:.2f}, prosečan zbir={avg_sum:.0f})")
        return trend_pool

    def stratified_sample(self, n_combinations: int = 5, zone: str = 'front') -> List[List[int]]:
        """
        slojevito uzorkovanje: 5 pool-ova + trend pool, svaki uzima udeo; generiše n grupa različitih kombinacija

        udeo uzorkovanja: vrući broj30% + hladni broj15% + balans20% + igra 10% + genetski 5% + trend20%
        
        Args:
            n_combinations: broj generisanih kombinacija
            zone: 'front' ili 'back'
            
        Returns:
            List[List[int]]: lista generisanih kombinacija
        """
        # Generisan svaki pool
        hot_pool = self.generate_hot_pool(20, zone)
        cold_pool = self.generate_cold_pool(20, zone)
        balance_pool = self.generate_balance_pool(20, zone)
        game_pool = self.generate_game_theory_pool(20, zone)
        genetic_pool = self.generate_genetic_pool(20, zone)
        trend_pool = self.generate_trend_pool(20, zone)
        
        # odredi veličinu kombinacije
        if zone == 'front':
            combo_size = FRONT_SELECT
        else:
            combo_size = BACK_SELECT if BACK_SELECT > 0 else 0
        
        # osiguraj nezavisnost pool-ova (ukloni duplikate), izbegni ponovno uzorkovanje
        cold_pool = [n for n in cold_pool if n not in hot_pool]
        balance_pool = [n for n in balance_pool if n not in set(hot_pool) | set(cold_pool)]
        game_pool = [n for n in game_pool if n not in set(hot_pool) | set(cold_pool) | set(balance_pool)]
        genetic_pool = [n for n in genetic_pool if n not in set(hot_pool) | set(cold_pool) | set(balance_pool) | set(game_pool)]
        trend_pool = [n for n in trend_pool if n not in set(hot_pool) | set(cold_pool) | set(balance_pool) | set(game_pool) | set(genetic_pool)]
        
        combinations = []
        
        # definiši opseg tri zone (samo Glavna zona)
        ZONE1 = set(ZONE1_RANGE)
        ZONE2 = set(ZONE2_RANGE)
        ZONE3 = set(ZONE3_RANGE)
        
        for _ in range(n_combinations):
            combo = []
            seen = set()
            
            def add_pool(pool, count):
                nonlocal combo, seen
                if pool and count > 0:
                    available = [n for n in pool if n not in seen]
                    k = min(count, len(available))
                    if k > 0:
                        picks = random.sample(available, k)
                        combo.extend(picks)
                        seen.update(picks)
            
            # vruci pool doprinosi30%
            add_pool(hot_pool, int(combo_size * 0.30))
            # hladni pool doprinosi15%
            add_pool(cold_pool, int(combo_size * 0.15))
            # balans pool doprinosi20%
            add_pool(balance_pool, int(combo_size * 0.20))
            # igra pool doprinosi10%
            add_pool(game_pool, int(combo_size * 0.10))
            # genetski pool doprinosi5%
            add_pool(genetic_pool, int(combo_size * 0.05))
            # trend pool doprinosi20%
            add_pool(trend_pool, int(combo_size * 0.20))
            
            # [optimizacija]Glavna zona prisilno pokrivanje tri zone: ako zona nije zastupljena, aktivno dopuni jedan broj iz te zone
            if zone == 'front' and len(combo) > 0:
                cur_set = set(combo)
                zones_present = [
                    (1 if cur_set & ZONE1 else 0),
                    (1 if cur_set & ZONE2 else 0),
                    (1 if cur_set & ZONE3 else 0),
                ]
                missing_zones = [i+1 for i, p in enumerate(zones_present) if p == 0]
                if missing_zones:
                    # iz pool-a nedostajuće zone pronađi brojeve koji nisu seen u brojevima
                    all_zone_pools = {
                        1: ZONE1 - seen,
                        2: ZONE2 - seen,
                        3: ZONE3 - seen,
                    }
                    for z in missing_zones:
                        candidates_z = list(all_zone_pools[z])
                        if candidates_z:
                            pick = random.choice(candidates_z)
                            combo.append(pick)
                            seen.add(pick)
            
            # dopuni do combo_size
            if len(combo) < combo_size:
                needed = combo_size - len(combo)
                available = [num for num in trend_pool if num not in seen]
                if len(available) >= needed:
                    picks = random.sample(available, needed)
                    combo.extend(picks)
                    seen.update(picks)
            
            # ako i dalje nema dovoljno, dopuni iz celokupnog opsega
            if len(combo) < combo_size:
                needed = combo_size - len(combo)
                if zone == 'front':
                    all_nums = list(list(NUM_RANGE))
                else:
                    all_nums = list(range(1, 13))
                available = [num for num in all_nums if num not in seen]
                if len(available) >= needed:
                    picks = random.sample(available, needed)
                    combo.extend(picks)
                    seen.update(picks)
            
            # konačna odbrana od duplikata i uzmi prvih combo_size
            combo = sorted(set(combo))[:combo_size]
            
            # ako set ukloni duplikate pa nema dovoljno, dopuni
            while len(combo) < combo_size:
                if zone == 'front':
                    fill = random.randint(LOTO7.min_broj, LOTO7.max_broj)
                else:
                    fill = random.randint(1, max(BACK_N, 12))
                if fill not in combo:
                    combo.append(fill)
            combo.sort()
            
            if combo not in combinations:
                combinations.append(combo)
        
        self._log(f"🎲 Generisan ({zone})kombinacije iz slojeva: {len(combinations)} kombinacija")
        return combinations
    
    def generate_6_plus_4(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Generisan n grupa 6+4 kombinovane igre
        
        Glavna zona 6 brojeva + Dodatna zona 4 brojeva
        
        Args:
            n: broj generisanih kombinovanih igara
            
        Returns:
            List[Dict[str, Any]]: lista kombinovanih igara, sadržiGlavna zonaibrojevi Dodatna zona
        """
        results = []
        
        for i in range(n):
            # Generisan Glavna zona 6 brojeva (koristi slojevito uzorkovanje)
            front_combinations = self.stratified_sample(n_combinations=10, zone='front')
            
            # izaberi najbolju Glavna zona kombinaciju (na osnovu skora teorije igara)
            best_front = None
            best_score = -1
            
            for combo in front_combinations:
                analysis = self.game_theory_analyzer.analyze_combo(combo)
                score = analysis['scores']['combined_score']
                
                if score > best_score:
                    best_score = score
                    best_front = combo
            
            # ako se ne pronađeNajbolja kombinacija, koristi prvu grupu
            if best_front is None and front_combinations:
                best_front = front_combinations[0]
            
            # proširi Glavna zona na 6 brojeva (iz vruci pool-a dopuni)
            if best_front and len(best_front) < 6:
                hot_pool = self.generate_hot_pool(20, 'front')
                available = [num for num in hot_pool if num not in best_front]
                if available:
                    needed = 6 - len(best_front)
                    best_front.extend(random.sample(available, min(needed, len(available))))
                    best_front = sorted(best_front)
            
            # ako i dalje nema 6, nasumično dopuni
            if len(best_front) < 6:
                needed = 6 - len(best_front)
                all_nums = list(list(NUM_RANGE))
                available = [num for num in all_nums if num not in best_front]
                if available:
                    best_front.extend(random.sample(available, min(needed, len(available))))
                    best_front = sorted(best_front)
            
            # Generisan Dodatna zona 4 broja (koristi slojevito uzorkovanje)
            back_combinations = self.stratified_sample(n_combinations=10, zone='back')
            
            # izaberi najbolju Dodatna zona kombinaciju (na osnovu skora vrućine/hladnoće)
            best_back = None
            best_back_score = -1
            
            for combo in back_combinations:
                # izračunaj prosečan skor Dodatna zona vrućine/hladnoće kombinacije
                scores = self._get_hot_cold_scores('back')
                avg_score = np.mean([scores.get(num, 0.5) for num in combo])
                
                if avg_score > best_back_score:
                    best_back_score = avg_score
                    best_back = combo
            
            # ako se ne pronađeNajbolja kombinacija, koristi prvu grupu
            if best_back is None and back_combinations:
                best_back = back_combinations[0]
            
            # proširi Dodatna zona na 4 broja (iz vruci pool-a dopuni)
            if best_back and len(best_back) < 4:
                hot_pool_back = self.generate_hot_pool(10, 'back')
                available = [num for num in hot_pool_back if num not in best_back]
                if available:
                    needed = 4 - len(best_back)
                    best_back.extend(random.sample(available, min(needed, len(available))))
                    best_back = sorted(best_back)
            
            # ako i dalje nema 4, nasumično dopuni
            if len(best_back) < 4:
                needed = 4 - len(best_back)
                available = [num for num in range(1, 13) if num not in best_back]
                if available:
                    best_back.extend(random.sample(available, min(needed, len(available))))
                    best_back = sorted(best_back)

            # izračunaj skor strategije
            try:
                gt_analysis = self.game_theory_analyzer.analyze_combo(best_front)
                strategy_score = gt_analysis['scores']['combined_score']
            except:
                strategy_score = 0.5

            results.append({
                'front': sorted(best_front) if best_front else [],
                'back': sorted(best_back) if best_back else [],
                'front_count': len(best_front) if best_front else 0,
                'back_count': len(best_back) if best_back else 0,
                'pool_type': 'genetic',
                'strategy_score': strategy_score,
                'bet_type': f'6+4'
            })

        return results

    def generate_compound_bet(self, front_count: int, back_count: int,
                              strategy: str = 'game_theory', n: int = 1) -> List[Dict[str, Any]]:
        """
        Generisankombinovana igra zadatog formata (Glavna zonafront_countbrojeva + Dodatna zonaback_countbrojeva)

        Args:
            front_count: brojevi Glavna zonabroj (5-9)
            back_count: brojevi Dodatna zonabroj (2-6)
            strategy: strategijapool ('hot', 'cold', 'balance', 'game_theory', 'genetic', 'mixed')
            n: Generisankolikogrupa

        Returns:
            List[Dict]: svakagrupasadrži front, back, pool_type, strategy_score
        """
        results = []
        for _ in range(n):
            # premastrategyizaberipool
            if strategy == 'hot':
                front_pool = self.generate_hot_pool(20, 'front')
                back_pool = self.generate_hot_pool(8, 'back')
            elif strategy == 'cold':
                front_pool = self.generate_cold_pool(20, 'front')
                back_pool = self.generate_cold_pool(8, 'back')
            elif strategy == 'balance':
                front_pool = self.generate_balance_pool(20, 'front')
                back_pool = self.generate_balance_pool(8, 'back')
            elif strategy == 'game_theory':
                front_pool = self.generate_game_theory_pool(20, 'front')
                back_pool = self.generate_game_theory_pool(8, 'back')
            elif strategy == 'genetic':
                front_pool = self.generate_genetic_pool(20, 'front')
                back_pool = self.generate_genetic_pool(8, 'back')
            else:  # mixed
                pools = [
                    self.generate_hot_pool(10, 'front'),
                    self.generate_cold_pool(10, 'front'),
                    self.generate_balance_pool(10, 'front'),
                ]
                front_pool = []
                for p in pools:
                    for num in p:
                        if num not in front_pool:
                            front_pool.append(num)
                back_pools = [
                    self.generate_hot_pool(4, 'back'),
                    self.generate_cold_pool(4, 'back'),
                    self.generate_balance_pool(4, 'back'),
                ]
                back_pool = []
                for p in back_pools:
                    for num in p:
                        if num not in back_pool:
                            back_pool.append(num)

            # osiguraj da je pool dovoljno velik
            if len(front_pool) < front_count or len(back_pool) < back_count:
                all_front = list(list(NUM_RANGE))
                all_back = list(range(1, 13))
                if len(front_pool) < front_count:
                    front_pool = all_front[:front_count]
                if len(back_pool) < back_count:
                    back_pool = all_back[:back_count]

            # sortiraj i biraj prema skoru teorije igaraGlavna zona
            scored = []
            for num in front_pool[:max(front_count, len(front_pool))]:
                try:
                    analysis = self.game_theory_analyzer.analyze_combo([num])
                    score = analysis['scores']['combined_score']
                except:
                    score = 0.5
                scored.append((num, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            front = sorted([x[0] for x in scored[:front_count]])

            # Dodatna zona: direktnouzmipreback_count
            back = sorted(back_pool[:back_count])

            # izračunaj skor strategije
            try:
                gt_analysis = self.game_theory_analyzer.analyze_combo(front)
                strategy_score = gt_analysis['scores']['combined_score']
            except:
                strategy_score = 0.5

            results.append({
                'front': front,
                'back': back,
                'front_count': len(front),
                'back_count': len(back),
                'pool_type': strategy,
                'strategy_score': strategy_score,
                'bet_type': f'{len(front)}+{len(back)}'
            })

        return results

    def generate_6_plus_3(self, n: int = 5, strategy: str = 'game_theory') -> List[Dict[str, Any]]:
        """Generisanngrupa6+3kombinovana igra"""
        return self.generate_compound_bet(6, 3, strategy, n)

    def generate_7_plus_2(self, n: int = 5, strategy: str = 'mixed') -> List[Dict[str, Any]]:
        """Generisanngrupa7+2kombinovana igra"""
        return self.generate_compound_bet(7, 2, strategy, n)

    def generate_7_plus_3(self, n: int = 5, strategy: str = 'mixed') -> List[Dict[str, Any]]:
        """Generisanngrupa7+3kombinovana igra"""
        return self.generate_compound_bet(7, 3, strategy, n)

    def generate_8_plus_2(self, n: int = 5, strategy: str = 'hot') -> List[Dict[str, Any]]:
        """Generisanngrupa8+2kombinovana igra"""
        return self.generate_compound_bet(8, 2, strategy, n)

    def generate_8_plus_3(self, n: int = 5, strategy: str = 'mixed') -> List[Dict[str, Any]]:
        """Generisanngrupa8+3kombinovana igra"""
        return self.generate_compound_bet(8, 3, strategy, n)

    def generate_9_plus_3(self, n: int = 5, strategy: str = 'cold') -> List[Dict[str, Any]]:
        """Generisanngrupa9+3kombinovana igra"""
        return self.generate_compound_bet(9, 3, strategy, n)

    def generate_9_plus_4(self, n: int = 5, strategy: str = 'cold') -> List[Dict[str, Any]]:
        """Generisanngrupa9+4kombinovana igra"""
        return self.generate_compound_bet(9, 4, strategy, n)

    def generate_9_plus_6(self, n: int = 5, strategy: str = 'balance') -> List[Dict[str, Any]]:
        """Generisanngrupa9+6kombinovana igra"""
        return self.generate_compound_bet(9, 6, strategy, n)
    # ============================================================
    # kombinovana igraGenerisanmetode (Glavna zona5-9 + Dodatna zona2-6)
    # ============================================================

    def generate_compound_bet(self, front_count: int, back_count: int,
                              strategy: str = 'game_theory', n: int = 1) -> List[Dict[str, Any]]:
        """
        Generisankombinovana igra zadatog formata (Glavna zonafront_countbrojeva + Dodatna zonaback_countbrojeva)

        Args:
            front_count: brojevi Glavna zonabroj (5-9)
            back_count: brojevi Dodatna zonabroj (2-6)
            strategy: strategijapool ('hot', 'cold', 'balance', 'game_theory', 'genetic', 'mixed')
            n: Generisankolikogrupa

        Returns:
            List[Dict]: svakagrupasadrži front, back, pool_type, strategy_score
        """
        results = []
        all_front = list(list(NUM_RANGE))
        all_back = list(range(1, 13))

        for _ in range(n):
            # premastrategyizaberipool
            if strategy == 'hot':
                front_pool = self.generate_hot_pool(20, 'front')
                back_pool = self.generate_hot_pool(8, 'back')
            elif strategy == 'cold':
                front_pool = self.generate_cold_pool(20, 'front')
                back_pool = self.generate_cold_pool(8, 'back')
            elif strategy == 'balance':
                front_pool = self.generate_balance_pool(20, 'front')
                back_pool = self.generate_balance_pool(8, 'back')
            elif strategy == 'game_theory':
                front_pool = self.generate_game_theory_pool(20, 'front')
                back_pool = self.generate_game_theory_pool(8, 'back')
            elif strategy == 'genetic':
                front_pool = self.generate_genetic_pool(20, 'front')
                back_pool = self.generate_genetic_pool(8, 'back')
            else:  # mixed
                pools_f = [self.generate_hot_pool(8, 'front'),
                           self.generate_cold_pool(8, 'front'),
                           self.generate_balance_pool(8, 'front')]
                front_pool = []
                for p in pools_f:
                    for num in p:
                        if num not in front_pool:
                            front_pool.append(num)
                pools_b = [self.generate_hot_pool(4, 'back'),
                           self.generate_cold_pool(4, 'back'),
                           self.generate_balance_pool(4, 'back')]
                back_pool = []
                for p in pools_b:
                    for num in p:
                        if num not in back_pool:
                            back_pool.append(num)

            # osiguraj da je pool dovoljno velik
            if len(front_pool) < front_count:
                for num in all_front:
                    if num not in front_pool:
                        front_pool.append(num)
                    if len(front_pool) >= front_count:
                        break
            if len(back_pool) < back_count:
                for num in all_back:
                    if num not in back_pool:
                        back_pool.append(num)
                    if len(back_pool) >= back_count:
                        break

            # sortiraj i biraj prema skoru teorije igaraGlavna zona
            scored = []
            for num in front_pool[:max(front_count, len(front_pool))]:
                try:
                    analysis = self.game_theory_analyzer.analyze_combo([num])
                    score = analysis['scores']['combined_score']
                except:
                    score = 0.5
                scored.append((num, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            front = sorted([x[0] for x in scored[:front_count]])

            # Dodatna zona: direktnouzmipreback_count
            back = sorted(back_pool[:back_count])

            # izračunaj skor strategije
            try:
                gt_analysis = self.game_theory_analyzer.analyze_combo(front)
                strategy_score = gt_analysis['scores']['combined_score']
            except:
                strategy_score = 0.5

            results.append({
                'front': front,
                'back': back,
                'front_count': len(front),
                'back_count': len(back),
                'pool_type': strategy,
                'strategy_score': strategy_score,
                'bet_type': f'{len(front)}+{len(back)}'
            })

        return results

    def generate_6_plus_3(self, n: int = 5, strategy: str = 'game_theory') -> List[Dict[str, Any]]:
        """Generisanngrupa6+3kombinovana igra"""
        return self.generate_compound_bet(6, 3, strategy, n)

    def generate_7_plus_2(self, n: int = 5, strategy: str = 'mixed') -> List[Dict[str, Any]]:
        """Generisanngrupa7+2kombinovana igra"""
        return self.generate_compound_bet(7, 2, strategy, n)

    def generate_7_plus_3(self, n: int = 5, strategy: str = 'mixed') -> List[Dict[str, Any]]:
        """Generisanngrupa7+3kombinovana igra"""
        return self.generate_compound_bet(7, 3, strategy, n)

    def generate_8_plus_2(self, n: int = 5, strategy: str = 'hot') -> List[Dict[str, Any]]:
        """Generisanngrupa8+2kombinovana igra"""
        return self.generate_compound_bet(8, 2, strategy, n)

    def generate_8_plus_3(self, n: int = 5, strategy: str = 'mixed') -> List[Dict[str, Any]]:
        """Generisanngrupa8+3kombinovana igra"""
        return self.generate_compound_bet(8, 3, strategy, n)

    def generate_9_plus_3(self, n: int = 5, strategy: str = 'cold') -> List[Dict[str, Any]]:
        """Generisanngrupa9+3kombinovana igra"""
        return self.generate_compound_bet(9, 3, strategy, n)

    def generate_9_plus_4(self, n: int = 5, strategy: str = 'cold') -> List[Dict[str, Any]]:
        """Generisanngrupa9+4kombinovana igra"""
        return self.generate_compound_bet(9, 4, strategy, n)

    def generate_9_plus_6(self, n: int = 5, strategy: str = 'balance') -> List[Dict[str, Any]]:
        """Generisanngrupa9+6kombinovana igra"""
        return self.generate_compound_bet(9, 6, strategy, n)

    def generate_6_plus_4(self, n: int = 5, strategy: str = 'game_theory') -> List[Dict[str, Any]]:
        """Generisan n grupa 6+4 kombinovane igre (Glavna zona6+Dodatna zona4)"""
        return self.generate_compound_bet(6, 4, strategy, n)

    def generate_7_plus_3(self, n: int = 5, strategy: str = 'mixed') -> List[Dict[str, Any]]:
        """Generisanngrupa7+3kombinovana igra (Glavna zona7+Dodatna zona3)"""
        return self.generate_compound_bet(7, 3, strategy, n)

    def generate_7_plus_4(self, n: int = 5, strategy: str = 'balance') -> List[Dict[str, Any]]:
        """Generisanngrupa7+4kombinovana igra (Glavna zona7+Dodatna zona4)"""
        return self.generate_compound_bet(7, 4, strategy, n)

    def generate_8_plus_4(self, n: int = 5, strategy: str = 'cold') -> List[Dict[str, Any]]:
        """Generisanngrupa8+4kombinovana igra (Glavna zona8+Dodatna zona4)"""
        return self.generate_compound_bet(8, 4, strategy, n)

    def generate_8_plus_5(self, n: int = 5, strategy: str = 'cold') -> List[Dict[str, Any]]:
        """Generisanngrupa8+5kombinovana igra (Glavna zona8+Dodatna zona5)"""
        return self.generate_compound_bet(8, 5, strategy, n)
