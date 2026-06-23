#!/usr/bin/env python3
"""
Bajesov model pojavljivanja brojeva.
"""


import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy.special import beta as beta_func, betaln, gammaln
from scipy.stats import beta as beta_dist

from lottery_config import FRONT_N, BACK_N, FRONT_SELECT, BACK_SELECT, NUM_RANGE


class BetaBinomialMissing:
    """
    Bajesovski model vrednosti nedostajanja
    
    Koristi Beta-Binomial konjugovani apriorni model za verovatnoću pojavljivanja svakog broja
    - Apriori: Beta(α, β), predstavlja apriorno uverenje bez podataka
    - Verovatnoća: Binomial(n, p), broj se pojavio k puta u n pokušaja
    - Aposteriori: Beta(α + k, β + n - k)
    
    Za vrednost nedostajanja (missing value) računamo aposteriorne kvantile kao obeležje
    """
    
    def __init__(self, 
                 alpha_prior: float = 1.0, 
                 beta_prior: float = 1.0,
                 # Koristi Jeffreys apriori alpha=0.5, beta=0.5
                 # ili uniformni apriori alpha=1, beta=1
                 method: str = 'jeffreys'):
        """
        Inicijalizacija bajesovskog modela vrednosti nedostajanja
        
        Args:
            alpha_prior: α parametar Beta apriorija
            beta_prior: β parametar Beta apriorija
            method: metoda apriorija
                - 'jeffreys': Jeffreys apriori α=0.5, β=0.5
                - 'uniform': uniformni apriori α=1, β=1
                - 'laplacian': Laplace izglađivanje α=1, β=1 (ekvivalentno uniform)
                - 'custom': prilagođeno
        """
        if method == 'jeffreys':
            self.alpha_prior = 0.5
            self.beta_prior = 0.5
        elif method == 'uniform' or method == 'laplacian':
            self.alpha_prior = 1.0
            self.beta_prior = 1.0
        elif method == 'custom':
            self.alpha_prior = alpha_prior
            self.beta_prior = beta_prior
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.method = method
        self.history_ = {}  # čuva istorijsku statistiku za svaki broj
    
    def update(self, numbers: List[int], n_trials: int = 1):
        """
        Ažuriranje bajesovskog modela novim podacima
        numbers: lista brojeva koji su se pojavili u ovom kolu
        n_trials: broj pokušaja (obično 1)
        """
        for num in numbers:
            if num not in self.history_:
                self.history_[num] = {'alpha': self.alpha_prior, 
                                      'beta': self.beta_prior}
            
            # Ažuriranje aposteriorija: Beta(α + k, β + n - k), k je broj pojavljivanja
            self.history_[num]['alpha'] += 1 * n_trials  # k=1
            self.history_[num]['beta'] += (1 - 1) * n_trials  # n-k=0
    
    def update_batch(self, all_draws: List[List[int]], n_trials: int = 1):
        """
        Grupno ažuriranje
        all_draws: lista istorijskih izvlačenja, svaki zapis je lista brojeva
        """
        for numbers in all_draws:
            self.update(numbers, n_trials)
    
    def get_posterior_params(self, num: int) -> Tuple[float, float]:
        """
        Dobavlja aposteriorne parametre Beta raspodele za određeni broj
        """
        if num not in self.history_:
            return self.alpha_prior, self.beta_prior
        
        return self.history_[num]['alpha'], self.history_[num]['beta']
    
    def get_expected_prob(self, num: int) -> float:
        """
        Dobavlja očekivanu vrednost verovatnoće pojavljivanja (aposteriorni srednji)
        E[p | data] = α / (α + β)
        """
        alpha, beta = self.get_posterior_params(num)
        return alpha / (alpha + beta)
    
    def get_missing_value(self, num: int, current_missing: int) -> float:
        """
        Računa bajesovsko obeležje vrednosti nedostajanja
        
        Vrednost nedostajanja sama nije nezavisna slučajna promenljiva, modelujemo je aposteriornom prediktivnom raspodelom
        Uzimajući u obzir trenutni broj perioda nedostajanja current_missing, računamo:
        1. Aposteriorni srednji
        2. Aposteriorne kvantile
        3. Aposteriornu verovatnoću da nedostajanje premaši current_missing
        """
        alpha, beta = self.get_posterior_params(num)
        
        # Aposteriorni srednji
        mean_prob = alpha / (alpha + beta)
        
        # Aposteriorna verovatnoća da nedostajanje premaši current_missing perioda
        # tj. P(X > current_missing | data), gde je X broj perioda nedostajanja pre sledećeg pojavljivanja
        # Ekvivalentno računanju kako se P(p < mean_prob) menja sa porastom nedostajanja u aposteriornoj prediktivnoj raspodeli
        
        # Praktičniji pristup: računanje aposteriornih kvantila
        # P(p <= x | data) = CDF_Beta(x; alpha, beta)
        # Obležja kvantila:
        # - medijanski kvantil
        # - niski kvantil (ukazuje da je trenutni broj perioda nedostajanja relativno neuobičajen)
        # - visoki kvantil
        
        # Računanje "stepenosti neuobičajenosti" kada nedostajanje premašuje trenutni broj perioda
        # Na osnovu Beta-Binomial prediktivne raspodele
        # P(missed > m) = sum_{k=0}^{m} C(n,k) p^k (1-p)^(n-k) integrisano preko aposteriornog p
        
        # Pojednostavljeno: koristi kumulativnu verovatnoću aposteriorne raspodele kao stepen neuobičajenosti nedostajanja
        # Ako je mean_prob nizak a nedostajanje dugo traje, možda uskoro izlazi
        
        return current_missing * mean_prob  # očekivana verovatnoća ponderisana nedostajanjem
    
    def get_missing_quantile(self, num: int, current_missing: int) -> float:
        """
        Računa Bajesovski aposteriorni kvantil vrednosti nedostajanja kao obeležje
        
        Vraća: aposteriorni kvantil koji odgovara trenutnom broju perioda nedostajanja
        tj. vrednost P(p <= observed_freq | data)
        """
        alpha, beta = self.get_posterior_params(num)
        
        # Računanje apriornog/aposteriornog srednjeg kao "posmatrane frekvencije"
        observed_freq = alpha / (alpha + beta)
        
        # Računanje kvantila nedostajanja
        # Ovo je pojednostavljen heuristički pristup
        # Pravi kvantil nedostajanja zahteva raspodelu vremena čekanja
        
        # CDF Beta raspodele: P(p <= x) = I_x(alpha, beta) / B(alpha, beta)
        # Koristim srednji kao referentnu vrednost, računamo uticaj nedostajanja na procenu verovatnoće
        
        # Kvantil nedostajanja = 1 - CDF(beta; alpha_post, beta_post) na current_missing
        # Ova formula je složena, koristi se aproksimacija:
        
        # Pojednostavljeno: duže nedostajanje -> ekstremniji kvantil (blizu 0 ili 1)
        # Ako dugo nedostaje i srednji je nizak -> blizu 0 (neuobičajeno)
        # Ako dugo nedostaje i srednji je visok -> blizu 1 (uskoro izlazi)
        
        # Računanje kvantila za trenutni period nedostajanja
        # Koristi funkciju preživljavanja Beta raspodele
        try:
            # P(X > current_missing) koristeći Beta-Binomial prediktivnu raspodelu
            # Aproksimacija: koristi funkciju preživljavanja aposteriorne raspodele
            # Ova aproksimacija je tačnija za manje brojeve perioda nedostajanja
            if current_missing <= 0:
                return 0.5
            
            # Koristi CDF Beta raspodele
            # P(p > mean) = 1 - CDF(mean; alpha, beta)
            mean_prob = alpha / (alpha + beta)
            
            # Normalizacija broja perioda nedostajanja
            # Pretpostavka: broj perioda nedostajanja sledi geometrijsku raspodelu, srednja vrednost 1/p
            # tada standardizovana vrednost observed_missing = current_missing * mean_prob
            
            # Računanje aposteriornog kvantila Beta raspodele
            # Kvantil = "ponderisano nedostajanjem" P(p <= mean_prob | data)
            # Aproksimacija:
            missing_factor = min(current_missing * mean_prob / 10.0, 1.0)
            
            # Aposteriorni kvantil
            try:
                quantile = beta_dist.cdf(mean_prob, alpha, beta)
            except:
                quantile = 0.5
            
            # Kvantil ponderisan nedostajanjem
            # Ako dugo nedostaje ali je srednji visok, kvantil se pomeri ka 1 (uskoro izlazi)
            # Ako dugo nedostaje ali je srednji nizak, kvantil se pomeri ka 0 (malo verovatno da izađe)
            weighted_quantile = quantile + (1 - quantile) * missing_factor if mean_prob > 0.15 else quantile * (1 - missing_factor)
            
            return np.clip(weighted_quantile, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def compute_missing_features(self, all_numbers: List[int], 
                                 current_missing: Dict[int, int]) -> np.ndarray:
        """
        Računa vektor obeležja nedostajanja za sve brojeve
        
        Args:
            all_numbers: lista svih mogućih brojeva
            current_missing: rečnik trenutnog broja perioda nedostajanja za svaki broj
            
        Returns:
            matrica obeležja [n_numbers, n_features]
            Svaki red je obeležje jednog broja: [expected_prob, missing_quantile, missing_value, posterior_alpha, posterior_beta]
        """
        features = []
        
        for num in all_numbers:
            alpha, beta = self.get_posterior_params(num)
            exp_prob = self.get_expected_prob(num)
            miss = current_missing.get(num, 0)
            miss_quantile = self.get_missing_quantile(num, miss)
            miss_value = self.get_missing_value(num, miss)
            
            features.append([
                exp_prob,       # aposteriorna srednja verovatnoća
                miss_quantile,  # bajesovski kvantil nedostajanja
                miss_value,    # obeležje ponderisano nedostajanjem
                alpha,         # aposteriorni α
                beta,          # aposteriorni β
                miss,          # trenutni broj perioda nedostajanja
            ])
        
        return np.array(features)
    
    def get_all_number_probs(self, all_numbers: List[int]) -> Dict[int, float]:
        """
        Dobavlja rečnik verovatnoća pojavljivanja za sve brojeve
        """
        return {num: self.get_expected_prob(num) for num in all_numbers}


class MissingValueTracker:
    """
    Praćenje vrednosti nedostajanja
    Prati broj perioda nedostajanja svakog broja od poslednjeg pojavljivanja
    """
    
    def __init__(self, n_front: int = FRONT_N, n_back: int = BACK_N):
        self.n_front = n_front
        self.n_back = n_back
        self.last_appeared_ = {}
        self.reset()
    
    def reset(self):
        """Resetuje praćenje"""
        self.last_appeared_ = {}
        self.current_missing_ = {}
        
        # Inicijalizacija nedostajanja svih brojeva na 0
        for i in range(1, self.n_front + 1):
            self.current_missing_[i] = 0
        for i in range(1, self.n_back + 1):
            self.current_missing_[i] = 0
    
    def update(self, draw: Tuple[List[int], List[int]]):
        """
        Ažuriranje vrednosti nedostajanja
        draw: (lista brojeva Glavne zone, lista brojeva Dodatne zone)
        """
        front_nums, back_nums = draw
        
        # Povećanje broja perioda nedostajanja za sve brojeve
        for num in self.current_missing_:
            self.current_missing_[num] += 1
        
        # Reset nedostajanja za brojeve koji su se pojavili
        for num in front_nums:
            self.current_missing_[num] = 0
            self.last_appeared_[num] = self.last_appeared_.get(num, 0) + 1
        
        for num in back_nums:
            self.current_missing_[num] = 0
            self.last_appeared_[num] = self.last_appeared_.get(num, 0) + 1
    
    def update_batch(self, draws: List[Tuple[List[int], List[int]]]):
        """Grupno ažuriranje"""
        for draw in draws:
            self.update(draw)
    
    def get_current_missing(self) -> Dict[int, int]:
        """Dobavlja trenutni broj perioda nedostajanja"""
        return self.current_missing_.copy()
    
    def get_missing_vector(self, numbers: List[int]) -> np.ndarray:
        """Dobavlja vektor nedostajanja za zadate brojeve"""
        return np.array([self.current_missing_.get(n, 0) for n in numbers])
    
    def get_stats(self) -> Dict:
        """Dobavlja statistiku vrednosti nedostajanja"""
        front_missing = [self.current_missing_[i] for i in range(1, self.n_front + 1)]
        back_missing = [self.current_missing_[i] for i in range(1, self.n_back + 1)]
        
        return {
            'front': {
                'mean': np.mean(front_missing),
                'std': np.std(front_missing),
                'max': max(front_missing),
                'min': min(front_missing),
                'missing_vector': front_missing
            },
            'back': {
                'mean': np.mean(back_missing),
                'std': np.std(back_missing),
                'max': max(back_missing),
                'min': min(back_missing),
                'missing_vector': back_missing
            }
        }


class BayesianNumberModel:
    """
    Integrisani bajesovski model brojeva
    Objedinjuje Beta-Binomial model nedostajanja i analizu ko-pojavljivanja
    """
    
    def __init__(self, n_front: int = FRONT_N, n_back: int = BACK_N):
        self.n_front = n_front
        self.n_back = n_back
        self.front_model = BetaBinomialMissing(method='jeffreys')
        self.back_model = BetaBinomialMissing(method='jeffreys')
        self.front_missing_tracker = MissingValueTracker(n_front=n_front)
        self.back_missing_tracker = MissingValueTracker(n_back=n_back)
        
        # Statistika ko-pojavljivanja
        self.front_cooccurrence_ = defaultdict(lambda: defaultdict(int))
        self.back_cooccurrence_ = defaultdict(lambda: defaultdict(int))
    
    def fit(self, draws: List[Tuple[List[int], List[int]]]):
        """
        Prilagođavanje modela Istoriji
        draws: [(front_nums, back_nums), ...]
        """
        # Ažuriranje bajesovskog modela
        for front_nums, back_nums in draws:
            self.front_model.update_batch([front_nums])
            self.back_model.update_batch([back_nums])
        
        # Ažuriranje praćenja nedostajanja
        self.front_missing_tracker.update_batch(draws)
        self.back_missing_tracker.update_batch(draws)
        
        # Ažuriranje statistike ko-pojavljivanja
        for front_nums, _ in draws:
            for i, n1 in enumerate(front_nums):
                for n2 in front_nums[i+1:]:
                    self.front_cooccurrence_[n1][n2] += 1
                    self.front_cooccurrence_[n2][n1] += 1
        
        for _, back_nums in draws:
            for i, n1 in enumerate(back_nums):
                for n2 in back_nums[i+1:]:
                    self.back_cooccurrence_[n1][n2] += 1
                    self.back_cooccurrence_[n2][n1] += 1
    
    def get_front_features(self) -> np.ndarray:
        """Dobavlja obeležja svih brojeva Glavne zone"""
        all_nums = list(range(1, self.n_front + 1))
        missing = self.front_missing_tracker.get_current_missing()
        return self.front_model.compute_missing_features(all_nums, missing)
    
    def get_back_features(self) -> np.ndarray:
        """Dobavlja obeležja svih brojeva Dodatne zone"""
        all_nums = list(range(1, self.n_back + 1))
        missing = self.back_missing_tracker.get_current_missing()
        return self.back_model.compute_missing_features(all_nums, missing)
    
    def get_cooccurrence_strength(self, num1: int, num2: int, zone: str = 'front') -> float:
        """
        Dobavlja jačinu ko-pojavljivanja dva broja
        zone: 'front' ili 'back'
        """
        cooc = self.front_cooccurrence_ if zone == 'front' else self.back_cooccurrence_
        total_appearances = sum(
            1 for front, _ in []  # ovo treba proslediti spolja
        )
        
        count = cooc.get(num1, {}).get(num2, 0)
        return count
    
    def get_strong_pairs(self, zone: str = 'front', top_n: int = 10) -> List[Tuple[int, int, int]]:
        """
        Dobavlja najjače parove ko-pojavljivanja
        Vraća: [(num1, num2, count), ...]
        """
        cooc = self.front_cooccurrence_ if zone == 'front' else self.back_cooccurrence_
        
        pairs = set()
        for n1 in cooc:
            for n2 in cooc[n1]:
                if n1 < n2:  # izbegavanje duplikata
                    pairs.add((n1, n2, cooc[n1][n2]))
        
        # Sortiranje po broju ko-pojavljivanja
        sorted_pairs = sorted(pairs, key=lambda x: -x[2])
        return sorted_pairs[:top_n]


if __name__ == '__main__':
    # Test kod
    np.random.seed(39)
    
    # Simulacija Istorije
    n_draws = 500
    front_draws = []
    for _ in range(n_draws):
        front = sorted(np.random.choice(list(NUM_RANGE), FRONT_SELECT, replace=False).tolist())
        back = sorted(np.random.choice(range(1, max(BACK_N, 1) + 1), max(BACK_SELECT, 1), replace=False).tolist()) if BACK_SELECT > 0 else []
        front_draws.append((front, back))
    
    # Prilagođavanje modela
    model = BayesianNumberModel()
    model.fit(front_draws)
    
    # Dobavljanje obeležja
    front_features = model.get_front_features()
    back_features = model.get_back_features()
    
    print("Glavna zona — oblik:", front_features.shape)
    print("Dodatna zona — oblik:", back_features.shape)
    
    print("\nGlavna zona — prvih 5 brojeva:")
    print("  br   frekv  trend  zbir   α     β    nedostaje")
    for i in range(5):
        nums = list(NUM_RANGE)
        f = front_features[i]
        print(f"  {nums[i]:2d}  {f[0]:.4f}   {f[1]:.4f}    {f[2]:.4f}   {f[3]:.2f}  {f[4]:.2f}    {int(f[5])}")
    
    # Dobavljanje najjačih parova ko-pojavljivanja
    print("\nGlavna zona — najjači parovi:")
    pairs = model.get_strong_pairs('front', top_n=5)
    for n1, n2, count in pairs:
        print(f"  {n1}-{n2}: {count}×")
    
    # Statistika nedostajanja
    front_missing = model.front_missing_tracker.get_current_missing()
    back_missing = model.back_missing_tracker.get_current_missing()
    
    print(f"\nGlavna zona — prosečno nedostaje: {np.mean(list(front_missing.values())):.1f}, "
          f"maks={max(front_missing.values())}")
    print(f"Dodatna zona — prosečno nedostaje: {np.mean(list(back_missing.values())):.1f}, "
          f"maks={max(back_missing.values())}")



"""
Glavna zona — oblik: (39, 6)
Dodatna zona — oblik: (0,)

Glavna zona — prvih 5 brojeva:
  br   frekv  trend  zbir   α     β    nedostaje
   1  0.9934   0.6572    4.9671   75.50  0.50    5
   2  0.9934   0.5000    0.0000   75.50  0.50    0
   3  0.9942   0.5000    0.0000   85.50  0.50    0
   4  0.9949   0.4541    1.9898   97.50  0.50    2
   5  0.9940   0.4542    1.9881   83.50  0.50    2

Glavna zona — najjači parovi:
  8-35: 28×
  13-15: 25×
  4-32: 24×
  16-23: 24×
  30-39: 24×

Glavna zona — prosečno nedostaje: 4.8, maks=26
Dodatna zona — prosečno nedostaje: 4.8, maks=26
"""
