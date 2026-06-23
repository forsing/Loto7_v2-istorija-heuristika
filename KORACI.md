# Pokretanje koraka — Loto 7/39 (ceo CSV)

```bash
cd ....
```

Okruženje: ** venv **

---
 
## Korak 1 — backtest (ceo CSV)

```bash
python loto7_korak1.py
```

Izlaz: **`loto7_korak1.txt`**

Opciono drugi nivo modela:
```bash
python loto7_korak1.py --korak 3
```

---

## Korak 2 — pool + pattern + struktura

```bash
python loto7_korak2.py
```

Izlaz: **`loto7_korak2.txt`**

---

## Korak 3 — + constraint

```bash
python loto7_korak3.py
```

Izlaz: **`loto7_korak3.txt`**

---

## Korak 4 — + Bajes

```bash
python loto7_korak4.py
```

Izlaz: **`loto7_korak4.txt`**

---

## Korak 5 — + neuronska mreža 

```bash
python loto7_korak5.py
```

Izlaz: **`loto7_korak5.txt`**

---

## Korak 6 — pun pipeline

```bash
python loto7_korak6.py
```

Izlaz: **`loto7_korak6.txt`**

---

## Redosled

1 → 2 → 3 → 4 → 5 → 6, svaki posebno.

CSV: `/data/loto7hh_4636_k49.csv`.



*******************



## Pipeline

```
loto7_sampler (pool_sampler) → loto7_filter → loto7_fusion
  pattern_recognizer + difference_sequence + number_gravity + matrix_displacement
  + constraint_engine + bayesian (+ neural_models u koraku 5–6)
```
 
## Pokretanje — 6 koraka

Detalji: **`KORACI.md`**

| Korak | Komanda | Izlaz |
|-------|---------|-------|
| 1 | `python loto7_korak1.py` | `loto7_korak1.txt` (backtest) |
| 2 | `python loto7_korak2.py` | `loto7_korak2.txt` |
| 3 | `python loto7_korak3.py` | `loto7_korak3.txt` |
| 4 | `python loto7_korak4.py` | `loto7_korak4.txt` |
| 5 | `python loto7_korak5.py` | `loto7_korak5.txt` |
| 6 | `python loto7_korak6.py` | `loto7_korak6.txt` |

CSV: `/data/loto7hh_4636_k49.csv`. Seed: **39**.

## Fajlovi

| Fajl | Uloga |
|------|--------|
| `loto7_korak1.py` … `loto7_korak6.py` | Koraci 1–6 |
| `loto7_engine.py` | Motor (scoring po koraku) |
| `loto7_fusion.py` | Fuzija skills modula |
| `loto7_sampler.py` | Wrapper za `pool_sampler` |
| `loto7_filter.py` | Filter kombinacija |
| `loto7_neural.py` | Neural sloj (korak 5–6) |
| `loto7_koraci_io.py` | Upis txt izlaza |
| `ucitaj_podatke.py` | Učitavanje CSV |
| `lottery_config.py` | Parametri 7/39 |
| `pool_sampler.py` | 6 pool-ova |
| `pattern_recognizer.py` | Strukturalni obrasci |
| `constraint_engine.py` | Ograničenja |
| `bayesian.py` | Beta-Binomial |
| `difference_sequence.py` | Razmake |
| `number_gravity.py` | Co-occurrence |
| `matrix_displacement.py` | Matrica pozicija |
| `game_theory.py` | Pool „igra“ |
| `genetic_optimizer.py` | Pool genetika |
| `neural_models.py` | PyTorch modeli |



****************



CSV (4636 kola) → 6 pool-ova → ~8000 kandidata → filter → skor (fusion) → najbolja kombinacija
Sistem ne predviđa sledeće kolo direktno 
— generiše i rangira kombinacije po skladu sa istorijom i heuristikama. 
Seed 39, opsezi zbira/raspnona iz CSV.

6 koraka (strategija)
Korak	Šta radi	Slojevi u skoru
1
Walk-forward backtest (ceo CSV) vs nasumična kombinacija
Testira korak 2–6 model
2
Predikcija
pattern (55%) + struktura (45%)
3
+ constraint
+ constraint (30%)
4
+ Bajes
+ Bajes (25%)
5
+ neural (PyTorch)
+ neural (15%)
6
Pun pipeline
TEZINE_FUZIJE: pattern 35%, struktura 20%, bajes 20%, constraint 15%, neural 10%
Koraci 2–5 su postepeno uključivanje slojeva; korak 6 koristi fiksne težine iz lottery_config.py.

1. Generisanje kandidata — pool_sampler (6 bazena)
Pool	Metoda	Prednost	Mana
Vrući (30%)
Najčešći brojevi u poslednjih 30 kola
Hvata kratkoročni „talas“
Gambler's fallacy — prošlost ne garantuje budućnost
Hladni (15%)
Najređi u prozoru
Balansira vruće
Isti problem — „dug“ brojeva ne postoji u pravom lotou
Balans (20%)
Mix vrućih + hladnih
Diversifikacija
Heuristika bez teorijskog osnova
Trend (20%)
Smer zbira (rast/pad) → težina zona
Koristi makro-obrazac sume
Trend se lako preokrene; osetljiv na prozor
Igra (10%)
game_theory — „hladne“ kombinacije
Manje deljenja jackpota ako pogodi
Ne povećava šansu za pogodak, samo EV nagrade
Genetika (5%)
GA optimizuje fitness iz istorije
Pretražuje prostor pametnije od čistog randoma
Fitness je i dalje istorijski — lako overfit
stratified_sample meša pool-ove po POOL_TEZINE i gradi 7 brojeva.

2. Filter — loto7_filter (tvrda ograničenja)
Odbacuje kombinacije van:

zbir 105–176, raspon 23–36 (iz CSV)
svi parni / svi neparni
prazna zona (1–13 / 14–26 / 27–39)
≥5 uzastopnih brojeva
Sužava prostor na „realistične“ kombinacije (~većina istorijskih kola).
I nasumičan izbor + filter može izgledati „pametno“ bez prediktivne snage.

3. Pattern — pattern_recognizer
Skor poklapanja sa distribucijama iz svih kola:

raspon, uzastopni parovi, zbir, parnost
ponavljanje iz prethodnog kola
krajnja cifra, zone, prosti brojevi, AC vrednost
Bogat deskriptivni model; uzima u obzir prethodno kolo.
Mnogo težina (OBRAZAC_TEZINE) — podešavanje na istoriji; rizik da „uči šum“.

4. Struktura (3 modula)
difference_sequence
Razmake između susednih brojeva, drugi red razlika, entropija.

+ hvata tipične „razmake“ u sortiranim kombinacijama
− fiksne liste „visokofrekventnih“ razlika mogu zastareti

number_gravity
Matrica 39×39 co-occurrence (ko se sa kim pojavljuje).

+ parovi brojeva koji često idu zajedno
− u nezavisnom lotou veze su slučajne fluktuacije, ne uzrok

matrix_displacement
Brojevi u matrici 7×6 — pozicija, dijagonale, frekvencija ćelija.

+ originalan geometrijski pogled
− raspored 1–39 u matricu je arbitraran; slaba prediktivna logika

Skor strukture = prosek (razmake + gravitacija + matrica).

5. Constraint — constraint_engine
5 strategija (balans, široki opseg, visoka/niska suma, trend) sa hard i soft pravilima (zbir, parnost, zone, raspon, prosti, AC, uzastopni).

Eksplicitne, proverljive strategije; soft skor za fusion.
Strategije su ručno definisane; ne zna se koja je „prava“ za sledeće kolo.

6. Bajes — bayesian.py
Beta-Binomial (Jeffreys apriori) → očekivana verovatnoća pojave svakog broja;
skor = prosek po brojevima u kombinaciji.

Statistički korektan način za frekvencije; izglađuje retke brojeve.
Pretpostavlja stacionarnu verovatnoću — u fair lotou svaki broj ima istu šansu.

7. Neural — neural_models + loto7_neural
Ansambl TabNet + LSTM + Transformer na 12 osobina po broju
(frekvencija, recency, itd.). Korak 5–6, 8 epoha treninga.

Uči nelinearne obrasce iz sekvence kola.
Sporo; lako overfit na 4636 kola; bez PyTorch-a preskače se; mali uticaj (10–15% u skoru).

8. Game theory — samo u pool-u
Bira „nepopularne“ kombinacije (visoki brojevi, ekstremna parnost, veliki raspon…).

Nije u glavnom skoru engine-a — utiče samo kroz 10% „igra“ pool-a.

9. Genetika — genetic_optimizer
GA (populacija 30, 15 generacija) za pool i internu optimizaciju.

Evolutivna pretraga prostora kombinacija.
fitness = istorijski skor → optimizuje prošlost, ne budućnost.

Korak 1 — backtest strategija
Walk-forward: za svako kolo t trenira na kola[:t], predviđa kola[t], poredi sa random.

Metrike: prosečan broj pogodaka, % ≥2 pogotka, vs teorija (7²)/39 ≈ 1.26.

Jedini objektivan test.
Spor (hiljade kola × engine); 1200 pokušaja u backtestu vs 8000 u predikciji 
— razlicit broj pokušaja (max_pokusaja). 


Statistička — frekvencije, Bajes, pattern distribucije
Strukturalna — razmake, gravitacija, matrica, AC, zone
Ograničavajuća — filter + constraint strategije
Metaheuristička — GA, 8000 pokušaja, pool mešavina
ML — neural ansambl
EV — game theory (izbegavanje popularnih brojeva)

Sve to rangira kombinacije koje liče na istoriju 
— ne dokazuje prediktivnu prednost dok korak 1 ne pokaže stabilno bolje od randoma na celom CSV.






Učitava CSV → iz 6 pool-ova (vrući, hladni, balans, trend, igra, genetika) pravi kandidate 
→ filter (zbir, raspon, zone, parnost…) → skor po aktivnom koraku 
→ bira najbolju od ~8000 pokušaja. Seed 39.

Korak 1: walk-forward backtest na celom CSV-u (model vs random).
Koraci 2–6: ista predikcija, ali se postepeno dodaju slojevi skora 
(pattern + struktura → constraint → Bajes → neural → pun fusion).


Pool: frekvencije, trend zbira, game theory (nepopularne kombinacije), genetski algoritam
Filter: statistički opsezi iz CSV-a
Pattern: raspon, zbir, parnost, ponavljanje, zone, prosti, AC…
Struktura: razmake, co-occurrence, matrica 7×6
Constraint: 5 strategija (hard/soft pravila)
Bajes: Beta-Binomial frekvencije
Neural: TabNet + LSTM + Transformer (korak 5–6)
Modularan, koraci se testiraju jedan po jedan; 
backtest postoji; filter i opsezi su vezani za tvoj 7/39 CSV; 
reproduktibilan (seed 39).
