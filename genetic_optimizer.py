"""
Genetski optimizator kombinacija.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
genetski optimizator — genetskim algoritmom optimizuje kombinacije brojeva
Loto 7/39: 7 brojeva iz opsega 1–39
"""

import numpy as np
import random
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from lottery_config import LOTO7, FRONT_SELECT, BACK_SELECT, FRONT_RANGE, BACK_RANGE, NUM_RANGE, FRONT_N, BACK_N, proveri_niz_verovatnoce

@dataclass
class Chromosome:
    """Hromozom — Loto 7/39: 7 brojeva Glavna zona"""
    front_numbers: List[int]  # Glavna zona 7 brojeva
    back_numbers: List[int]   # Dodatna zona 0 brojeva
    fitness: float = 0.0      # skor podobnosti
    
    def __post_init__(self):
        """Obrada posle inicijalizacije"""
        self.front_numbers = sorted(self.front_numbers)
        self.back_numbers = sorted(self.back_numbers)
    
    def __str__(self):
        return f"Glavna zona: {self.front_numbers} | Dodatna zona: {self.back_numbers} | podobnost: {self.fitness:.4f}"


class DLTGeneticOptimizer:
    """DLT genetski optimizator"""
    
    def __init__(self, 
                 population_size: int = 30,
                 generations: int = 15,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 elite_size: int = 5,
                 tiho: bool = False):
        """Inicijalizacija genetskog optimizatora"""
        self.tiho = tiho
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # prostor pretrage
        self.front_range = FRONT_RANGE
        self.back_range = BACK_RANGE
        
        # populacija
        self.population: List[Chromosome] = []
        self.best_chromosome: Chromosome = None
        self.fitness_history: List[float] = []
        
        # funkcija podobnosti
        self.fitness_function: Callable = None
        
        self._log(f"🧬 Genetski optimizator — inicijalizacija")
        self._log(f"   Veličina populacije: {population_size}")
        self._log(f"   Generacije: {generations}")
        self._log(f"   Stopa ukrštanja: {crossover_rate}")
        self._log(f"   Stopa mutacije: {mutation_rate}")
        self._log(f"   Elitna rezerva: {elite_size}")
        self._log(f"   Opseg Glavna zona: 1–{FRONT_N} ({FRONT_SELECT} brojeva)")

    def _log(self, *args, **kwargs) -> None:
        if not self.tiho:
            print(*args, **kwargs)
    
    def set_fitness_function(self, fitness_func: Callable):
        """Postavljanje funkcije podobnosti"""
        self.fitness_function = fitness_func
        self._log("✅ Funkcija podobnosti postavljena")
    
    def initialize_population(self):
        """Inicijalizacija populacije"""
        self._log("🧬 Inicijalizacija populacije...")
        
        self.population = []
        
        for _ in range(self.population_size):
            # nasumično generiši FRONT_SELECT brojeva Glavna zona
            front_numbers = random.sample(
                range(self.front_range[0], self.front_range[1] + 1), 
                FRONT_SELECT
            )
            
            back_numbers = []
            if BACK_SELECT > 0:
                back_numbers = random.sample(
                    range(self.back_range[0], self.back_range[1] + 1), 
                    BACK_SELECT
                )
            
            chromosome = Chromosome(
                front_numbers=sorted(front_numbers),
                back_numbers=sorted(back_numbers)
            )
            
            self.population.append(chromosome)
        
        self._log(f"✅ Populacija inicijalizovana: {len(self.population)}  jedinica")
    
    def evaluate_fitness(self, hot_numbers: Dict[str, List[int]] = None):
        """Evaluacija podobnosti populacije"""
        if self.fitness_function is None:
            # koristi podrazumevanu funkciju podobnosti
            self._default_fitness_evaluation(hot_numbers)
        else:
            # koristi prilagođenu funkciju podobnosti
            for chromosome in self.population:
                chromosome.fitness = self.fitness_function(chromosome)
        
        # sortiraj po podobnosti
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # ažuriraj najbolji hromozom
        if self.best_chromosome is None or self.population[0].fitness > self.best_chromosome.fitness:
            self.best_chromosome = deepcopy(self.population[0])
        
        # zabeleži istoriju podobnosti
        avg_fitness = np.mean([c.fitness for c in self.population])
        self.fitness_history.append(avg_fitness)
    
    def _default_fitness_evaluation(self, hot_numbers: Dict[str, List[int]] = None):
        """Podrazumevana funkcija evaluacije podobnosti"""
        if hot_numbers is None:
            hot_numbers = {
                'front_hot': list(NUM_RANGE)[:15],
                'back_hot': list(range(self.back_range[0], self.back_range[1] + 1))[:6] if BACK_SELECT > 0 else []
            }
        
        for chromosome in self.population:
            fitness = 0.0
            
            # 1. udeo vrućih brojeva (40%)
            front_hot_count = len(set(chromosome.front_numbers) & set(hot_numbers['front_hot']))
            back_hot_count = len(set(chromosome.back_numbers) & set(hot_numbers['back_hot']))
            
            fitness += (front_hot_count / FRONT_SELECT) * 0.3
            if BACK_SELECT > 0:
                fitness += (back_hot_count / BACK_SELECT) * 0.1
            
            # 2. parnost (20%)
            front_odd_count = sum(1 for n in chromosome.front_numbers if n % 2 == 1)
            back_odd_count = sum(1 for n in chromosome.back_numbers if n % 2 == 1)
            
            # idealna parnost: Glavna zona 3:3 ili 4:2, Dodatna zona 1:2 ili 2:1
            ideal_odd = FRONT_SELECT // 2
            front_odd_score = 1 - abs(front_odd_count - ideal_odd) / max(ideal_odd, 1)
            back_odd_score = 0.0
            if BACK_SELECT > 0:
                back_odd_score = 1 - abs(back_odd_count - BACK_SELECT / 2) / max(BACK_SELECT / 2, 1)
            
            fitness += front_odd_score * 0.15
            if BACK_SELECT > 0:
                fitness += back_odd_score * 0.05
            
            # 3. raspodela brojeva (15%)
            # raspodela brojeva 1–39 (FRONT_N)
            front_spread = max(chromosome.front_numbers) - min(chromosome.front_numbers)
            spread_score = min(front_spread / (LOTO7.max_broj - LOTO7.min_broj), 1.0)
            
            fitness += spread_score * 0.15
            
            # 4. kontrola uzastopnih brojeva (10%)
            # proveri da li ima previše uzastopnih brojeva
            front_sorted = sorted(chromosome.front_numbers)
            consecutive_count = 0
            for i in range(len(front_sorted) - 1):
                if front_sorted[i+1] - front_sorted[i] == 1:
                    consecutive_count += 1
            
            consecutive_score = 1 - min(consecutive_count / 2, 1.0)  # najviše 2 uzastopna dozvoljena
            
            fitness += consecutive_score * 0.10
            
            # 5. opseg zbira (10%)
            front_sum = sum(chromosome.front_numbers)
            # idealan opseg zbira: 90-130
            zbir_ideal = sum(LOTO7.zbir_ideal) / 2
            if LOTO7.zbir_ideal[0] <= front_sum <= LOTO7.zbir_ideal[1]:
                sum_score = 1.0
            else:
                sum_score = 1 - min(abs(front_sum - zbir_ideal) / 50, 1.0)
            
            fitness += sum_score * 0.10
            
            # 6. kazna za ponavljajući obrazac (5%)
            # proveri da li postoji ponavljajući obrazac parova brojeva
            pattern_penalty = 0
            for i in range(len(front_sorted) - 1):
                diff = front_sorted[i+1] - front_sorted[i]
                if diff < 3:  # brojevi su previše zbijeni
                    pattern_penalty += 0.02
            
            fitness -= pattern_penalty
            
            # osiguraj podobnost u opsegu 0-1
            chromosome.fitness = max(0, min(fitness, 1))
    
    def selection(self) -> List[Chromosome]:
        """Selekcija — ruletska selekcija"""
        # izračunaj zbir podobnosti
        total_fitness = sum(c.fitness for c in self.population)
        
        # izračunaj verovatnoće selekcije
        if total_fitness == 0:
            # ako su sve podobnosti 0, ravnomerna selekcija
            probabilities = [1/len(self.population)] * len(self.population)
        else:
            probabilities = [c.fitness / total_fitness for c in self.population]
        
        # ruletska selekcija
        selected = []
        for _ in range(self.population_size - self.elite_size):
            r = random.random()
            cumulative = 0
            for i, prob in enumerate(probabilities):
                cumulative += prob
                if r <= cumulative:
                    selected.append(deepcopy(self.population[i]))
                    break
        
        return selected
    
    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Ukrštanje — dvotačkasto ukrštanje"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # ukrštanje Glavna zona
        front_crossover_point = random.randint(1, max(FRONT_SELECT - 1, 1))
        
        child1_front = parent1.front_numbers[:front_crossover_point] + parent2.front_numbers[front_crossover_point:]
        child2_front = parent2.front_numbers[:front_crossover_point] + parent1.front_numbers[front_crossover_point:]
        
        # ukloni duplikate i dopuni
        child1_front = self._repair_chromosome(child1_front, self.front_range, FRONT_SELECT)
        child2_front = self._repair_chromosome(child2_front, self.front_range, FRONT_SELECT)
        
        child1_back = []
        child2_back = []
        if BACK_SELECT > 0:
            back_crossover_point = random.randint(1, max(BACK_SELECT - 1, 1))
            child1_back = parent1.back_numbers[:back_crossover_point] + parent2.back_numbers[back_crossover_point:]
            child2_back = parent2.back_numbers[:back_crossover_point] + parent1.back_numbers[back_crossover_point:]
            child1_back = self._repair_chromosome(child1_back, self.back_range, BACK_SELECT)
            child2_back = self._repair_chromosome(child2_back, self.back_range, BACK_SELECT)
        
        child1 = Chromosome(front_numbers=child1_front, back_numbers=child1_back)
        child2 = Chromosome(front_numbers=child2_front, back_numbers=child2_back)
        
        return child1, child2
    
    def mutation(self, chromosome: Chromosome) -> Chromosome:
        """Mutacija"""
        mutated = deepcopy(chromosome)
        
        # mutacija Glavna zona
        if random.random() < self.mutation_rate:
            # nasumično zameni jedan broj
            idx = random.randint(0, FRONT_SELECT - 1)
            current = mutated.front_numbers[idx]
            
            # izaberi broj koji nije u trenutnoj kombinaciji
            available = [n for n in range(self.front_range[0], self.front_range[1] + 1) 
                        if n not in mutated.front_numbers]
            
            if available:
                mutated.front_numbers[idx] = random.choice(available)
                mutated.front_numbers.sort()
        
        # mutacija Dodatna zona
        if BACK_SELECT > 0 and random.random() < self.mutation_rate:
            idx = random.randint(0, BACK_SELECT - 1)
            
            # izaberi broj koji nije u trenutnoj kombinaciji
            available = [n for n in range(self.back_range[0], self.back_range[1] + 1) 
                        if n not in mutated.back_numbers]
            
            if available:
                mutated.back_numbers[idx] = random.choice(available)
                mutated.back_numbers.sort()
        
        return mutated
    
    def _repair_chromosome(self, numbers: List[int], num_range: Tuple[int, int], target_size: int) -> List[int]:
        """Popravka hromozoma (ukloni duplikate i dopuni do ciljne veličine)"""
        # ukloni duplikate
        unique_numbers = list(set(numbers))
        
        # ako nema dovoljno, dopuni nasumičnim brojevima
        if len(unique_numbers) < target_size:
            available = [n for n in range(num_range[0], num_range[1] + 1) 
                        if n not in unique_numbers]
            needed = target_size - len(unique_numbers)
            
            if len(available) >= needed:
                unique_numbers.extend(random.sample(available, needed))
            else:
                # ako nema dovoljno dostupnih brojeva, nasumično generiši
                while len(unique_numbers) < target_size:
                    new_num = random.randint(num_range[0], num_range[1])
                    if new_num not in unique_numbers:
                        unique_numbers.append(new_num)
        
        # ako ima previše, nasumično obriši
        elif len(unique_numbers) > target_size:
            unique_numbers = random.sample(unique_numbers, target_size)
        
        return sorted(unique_numbers)
    
    def evolve(self, hot_numbers: Dict[str, List[int]] = None) -> List[Chromosome]:
        """Izvršava evolucioni proces"""
        self._log(f"🔁 Početak genetske evolucije ({self.generations} generacija)...")
        
        # Inicijalizacija populacije
        self.initialize_population()
        
        # evoluciona petlja
        for generation in range(self.generations):
            # evaluacija podobnosti
            self.evaluate_fitness(hot_numbers)
            
            # ispiši informacije o trenutnoj generaciji
            if generation % 10 == 0 or generation == self.generations - 1:
                self._log(f"   Grupa {generation+1:3d} | prosečna podobnost: {self.fitness_history[-1]:.4f} | "
                      f"Najbolja podobnost: {self.best_chromosome.fitness:.4f}")
            
            # selekcija elite
            elites = self.population[:self.elite_size]
            
            # selekcija
            selected = self.selection()
            
            # ukrštanje i mutacija generišu novu populaciju
            new_population = []
            
            # zadrži elitu
            new_population.extend(elites)
            
            # generiši potomstvo
            while len(new_population) < self.population_size:
                # nasumično izaberi roditelje
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                # ukrštanje
                child1, child2 = self.crossover(parent1, parent2)
                
                # mutacija
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # ažuriraj populaciju
            self.population = new_population
        
        # konačna evaluacija
        self.evaluate_fitness(hot_numbers)
        
        self._log(f"✅ Evolucija završena")
        self._log(f"   Najbolja podobnost: {self.best_chromosome.fitness:.4f}")
        self._log(f"   Najbolja kombinacija: {self.best_chromosome}")
        
        return self.population
    
    def get_best_solutions(self, top_k: int = 5) -> List[Chromosome]:
        """Dobij najbolja rešenja"""
        # osiguraj da je populacija evaluirana
        if not self.population or self.population[0].fitness == 0:
            self.evaluate_fitness()
        
        # sortiraj po podobnosti
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        
        # ukloni duplikate
        unique_solutions = []
        seen = set()
        
        for chrom in sorted_population:
            key = (tuple(chrom.front_numbers), tuple(chrom.back_numbers))
            if key not in seen:
                seen.add(key)
                unique_solutions.append(chrom)
        
        return unique_solutions[:top_k]
    
    def optimize_with_probabilities(self, front_probs: np.ndarray, back_probs: np.ndarray, 
                                  top_k: int = 5) -> List[Dict[str, Any]]:
        """Optimizacija na osnovu raspodele verovatnoća"""
        front_probs = proveri_niz_verovatnoce(front_probs, FRONT_N, "front_probs")
        back_probs = proveri_niz_verovatnoce(back_probs, max(BACK_N, 1), "back_probs")
        self._log(f"🎯 Genetska optimizacija po verovatnoći...")
        
        # definiši funkciju podobnosti na osnovu verovatnoća
        def probability_fitness(chromosome: Chromosome) -> float:
            fitness = 0.0
            
            # skor verovatnoće Glavna zona
            front_prob_score = np.mean([front_probs[num-1] for num in chromosome.front_numbers])
            
            back_prob_score = 0.0
            if chromosome.back_numbers:
                back_prob_score = np.mean([back_probs[num-1] for num in chromosome.back_numbers])
            
            # kombinovani skor
            if BACK_SELECT > 0:
                fitness = front_prob_score * 0.7 + back_prob_score * 0.3
            else:
                fitness = front_prob_score
            
            # dodaj nagradu za raznovrsnost (izbegni preveliku koncentraciju)
            front_spread = max(chromosome.front_numbers) - min(chromosome.front_numbers)
            spread_bonus = min(front_spread / (LOTO7.max_broj - LOTO7.min_broj), 0.2)
            
            fitness += spread_bonus
            
            return fitness
        
        # postavi funkciju podobnosti
        self.set_fitness_function(probability_fitness)
        
        # izvrši evoluciju
        self.evolve()
        
        # dobij najbolja rešenja
        best_solutions = self.get_best_solutions(top_k)
        
        # konvertuj u rečnik
        results = []
        for i, chrom in enumerate(best_solutions, 1):
            results.append({
                'rank': i,
                'front_numbers': chrom.front_numbers,
                'back_numbers': chrom.back_numbers,
                'confidence': chrom.fitness,
                'front_confidence': np.mean([front_probs[num-1] for num in chrom.front_numbers]),
                'back_confidence': np.mean([back_probs[num-1] for num in chrom.back_numbers]),
                'optimization_method': 'genetski algoritam'
            })
        
        self._log(f"✅ Genetska optimizacija završena: generisano {len(results)}  optimizovanih kombinacija")
        
        return results


if __name__ == "__main__":
    print("🔬 Demo test genetskog optimizatora")
    print("=" * 60)
    
    # kreiraj optimizator
    optimizer = DLTGeneticOptimizer(
        population_size=50,
        generations=20,
        crossover_rate=0.8,
        mutation_rate=0.2,
        elite_size=5
    )
    
    # simuliraj raspodelu verovatnoća
    np.random.seed(39)
    front_probs = np.random.rand(FRONT_N)
    front_probs = front_probs / front_probs.sum()
    
    back_probs = np.random.rand(max(BACK_N, 1))
    back_probs = back_probs / back_probs.sum()
    
    # izvrši optimizaciju
    results = optimizer.optimize_with_probabilities(front_probs, back_probs, top_k=3)
    
    print(f"\\n🏆 Rezultat optimizacije:")
    for result in results:
        print(f"   Grupa {result['rank']} (pouzdanost: {result['confidence']:.4f}):")
        print(f"       Glavna zona: {result['front_numbers']}")
        print(f"       Dodatna zona: {result['back_numbers']}")
    
    print(f"\\n📈 Statistika evolucije:")
    print(f"   Konačna prosečna podobnost: {optimizer.fitness_history[-1]:.4f}")
    print(f"   Najbolja podobnost: {optimizer.best_chromosome.fitness:.4f}")
    
    print(f"\\n🎯 Demo: Genetski optimizator integrisan")


"""
🔬 Demo test genetskog optimizatora
============================================================
🧬 Genetski optimizator — inicijalizacija
   Veličina populacije: 50
   Generacije: 20
   Stopa ukrštanja: 0.8
   Stopa mutacije: 0.2
   Elitna rezerva: 5
   Opseg Glavna zona: 1–39 (7 brojeva)
🎯 Genetska optimizacija po verovatnoći...
✅ Funkcija podobnosti postavljena
🔁 Početak genetske evolucije (20 generacija)...
🧬 Inicijalizacija populacije...
✅ Populacija inicijalizovana: 50  jedinica
   Grupa   1 | prosečna podobnost: 0.2257 | Najboljapodobnost: 0.2334
   Grupa  11 | prosečna podobnost: 0.2319 | Najboljapodobnost: 0.2374
   Grupa  20 | prosečna podobnost: 0.2339 | Najboljapodobnost: 0.2377
✅ Evolucija završena
   Najbolja podobnost: 0.2377
   Najbolja kombinacija: Glavna zona: [3, x, 12, y, 20, z, 34] | Dodatna zona: [] | podobnost: 0.2377
✅ Genetska optimizacija završena: generisano 3  optimizovanih kombinacija
\n🏆 Rezultat optimizacije:
   Grupa 1 (pouzdanost: 0.2377):
       Glavna zona: [3, x, 12, y, 20, z, 34]
       Dodatna zona: []
   Grupa 2 (pouzdanost: 0.2376):
       Glavna zona: [2, x, 12, y, 20, z, 34]
       Dodatna zona: []
   Grupa 3 (pouzdanost: 0.2373):
       Glavna zona: [3, x, 12, y, 20, z, 34]
       Dodatna zona: []
\n📈 Statistika evolucije:
   Konačna prosečna podobnost: 0.2341
   Najbolja podobnost: 0.2377
\n🎯 Demo: Genetski optimizator integrisan
"""
