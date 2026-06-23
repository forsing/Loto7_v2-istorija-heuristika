#!/usr/bin/env python3
"""
Korak 1 — walk-forward backtest na celom CSV → loto7_korak1.txt
  python loto7_korak1.py
  python loto7_korak1.py --korak 3
"""

from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from loto7_engine import Loto7Engine
from loto7_koraci_io import KORAK_IZLAZ, KORAK_OPIS
from lottery_config import DEFAULT_CSV, LOTO7
from ucitaj_podatke import ucitaj_kola


def pogodci(pred: List[int], stvarno: List[int]) -> int:
    return len(set(pred) & set(stvarno))


def random_kombinacija(rng: random.Random) -> List[int]:
    return sorted(rng.sample(range(LOTO7.min_broj, LOTO7.max_broj + 1), LOTO7.brojeva_po_kombinaciji))


def run_backtest(
    kola: List[List[int]],
    min_istorija: int,
    pokusaja: int,
    seed: int,
    korak_model: int,
    progress_svakih: int = 200,
) -> dict:
    rng = random.Random(seed)
    start = min_istorija
    model_hits: List[int] = []
    rand_hits: List[int] = []
    ukupno = len(kola) - start
    print(
        f"  ceo CSV: {len(kola)} kola učitano, backtest na {ukupno} kola "
        f"(indeks {start}..{len(kola) - 1})",
        flush=True,
    )

    for i, t in enumerate(range(start, len(kola)), 1):
        istorija = kola[:t]
        stvarno = kola[t]
        preth = istorija[-1]
        engine = Loto7Engine(istorija, korak=korak_model, seed=seed + t, tiho=True)
        pred, _ = engine.najbolja(preth=preth, max_pokusaja=pokusaja)
        model_hits.append(pogodci(pred, stvarno))
        rand_hits.append(pogodci(random_kombinacija(rng), stvarno))
        if progress_svakih and i % progress_svakih == 0:
            print(f"  ceo CSV — backtest {i}/{ukupno}", flush=True)

    n = len(model_hits)
    p_teorija = (LOTO7.brojeva_po_kombinaciji ** 2) / LOTO7.max_broj
    return {
        "kola_testirano": n,
        "od_indeksa": start,
        "korak_model": korak_model,
        "prosek_pogodaka_model": sum(model_hits) / n if n else 0,
        "prosek_pogodaka_random": sum(rand_hits) / n if n else 0,
        "teorija_random": p_teorija,
        "hit_at_least_2_model": sum(1 for h in model_hits if h >= 2) / n if n else 0,
        "hit_at_least_2_random": sum(1 for h in rand_hits if h >= 2) / n if n else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Korak 1 — backtest")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--min-istorija", type=int, default=400)
    parser.add_argument("--pokusaja", type=int, default=8000, help="Pokušaji za finalnu kombinaciju (kao korak 2–6)")
    parser.add_argument("--pokusaja-backtest", type=int, default=1200, help="Pokušaji po kolu u walk-forward backtestu")
    parser.add_argument("--bez-backtest", action="store_true", help="Preskoči backtest — samo predikcija na celom CSV")
    parser.add_argument("--seed", type=int, default=LOTO7.seed)
    parser.add_argument("--korak", type=int, default=2, help="Koji model koraka testirati (2–6)")
    parser.add_argument("--izlaz", type=Path, default=KORAK_IZLAZ[1])
    args = parser.parse_args()
    t_start = time.perf_counter()

    kola = ucitaj_kola(args.csv)
    if args.bez_backtest:
        summary = {
            "kola_testirano": 0,
            "od_indeksa": args.min_istorija,
            "korak_model": args.korak,
            "prosek_pogodaka_model": 0.0,
            "prosek_pogodaka_random": 0.0,
            "teorija_random": (LOTO7.brojeva_po_kombinaciji ** 2) / LOTO7.max_broj,
            "hit_at_least_2_model": 0.0,
            "hit_at_least_2_random": 0.0,
        }
        print(f"Preskočen backtest — predikcija na celom CSV ({len(kola)} kola)", flush=True)
    else:
        print(f"Retrospektivni test: {len(kola)} kola u CSV, test od indeksa {args.min_istorija}", flush=True)
        summary = run_backtest(
            kola, args.min_istorija, args.pokusaja_backtest, args.seed, args.korak
        )

    linije = [
        "Loto 7/39 — KORAK 1",
        KORAK_OPIS[1],
        f"Generisano: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        f"CSV: {args.csv}",
        f"Kola u CSV: {len(kola)}",
        f"Testirano kola: {summary['kola_testirano']}"
        + (f" (od indeksa {summary['od_indeksa']})" if summary["kola_testirano"] else " (backtest preskočen)"),
        f"Model koraka u testu: {summary['korak_model']}",
        f"Seed: {args.seed}",
        "",
        f"Prosek pogodaka model:   {summary['prosek_pogodaka_model']:.3f}",
        f"Prosek pogodaka slučajno: {summary['prosek_pogodaka_random']:.3f}",
        f"Teorija (7 brojeva):     {summary['teorija_random']:.3f}",
        f"Pogodaka≥2 model:   {summary['hit_at_least_2_model']:.1%}",
        f"Pogodaka≥2 slučajno: {summary['hit_at_least_2_random']:.1%}",
        "",
    ]
    args.izlaz.write_text("\n".join(linije), encoding="utf-8")
    print("\n".join(linije))

    # predikcija na celom CSV-u (isti model kao u backtestu)
    eng = Loto7Engine(kola, korak=args.korak, seed=args.seed)
    combo, sc = eng.najbolja(preth=kola[-1], max_pokusaja=args.pokusaja)
    tekst = "  ".join(f"{b:02d}" for b in combo)
    trajanje_s = int(time.perf_counter() - t_start)
    trajanje = f"{trajanje_s // 60} min {trajanje_s % 60} s"
    with args.izlaz.open("a", encoding="utf-8") as f:
        f.write(f"\nKombinacija: {tekst}\n")
        f.write(f"Skor: {sc:.4f}\n")
        f.write(f"Trajanje: {trajanje}\n")
    print(f"Kombinacija: {tekst}  skor={sc:.4f}")
    print(f"Trajanje: {trajanje}")
    print(f"Korak 1 → {args.izlaz}")


if __name__ == "__main__":
    main()




"""
python loto7_korak1.py --bez-backtest

Preskočen backtest — predikcija na celom CSV (4636 kola)
Loto 7/39 — KORAK 1
Retrospektivni test (walk-forward, ceo CSV)
Generisano: 23.06.2026 11:29
CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4636_k49.csv
Kola u CSV: 4636
Testirano kola: 0 (backtest preskočen)
Model koraka u testu: 2
Seed: 39

Prosek pogodaka model:   0.000
Prosek pogodaka slučajno: 0.000
Teorija (7 brojeva):     1.256
Pogodaka≥2 model:   0.0%
Pogodaka≥2 slučajno: 0.0%

Kombinacija: 07  09  19  22  24  32  38  skor=0.4413
Trajanje: 3 min 33 s
Korak 1 → /Users/4c/Desktop/GHQ/STATISTIKA/lottery_v2/loto7_korak1.txt
"""





"""
Loto 7/39 — KORAK 1
Retrospektivni test (walk-forward, ceo CSV)
CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4636_k49.csv
Kola u CSV: 4636
Testirano kola: 4236 (od indeksa 400)
Model koraka u testu: 2
Seed: 39

Prosek pogodaka model:   1.253
Prosek pogodaka slučajno: 1.247
Teorija (7 brojeva):     1.256
Pogodaka≥2 model:   36.9%
Pogodaka≥2 slučajno: 36.7%

Kombinacija: 05  07  08  10  14  31  32
Skor: 02592
"""



"""
Retrospektivni test: 4636 kola u CSV, test od indeksa 400
  ... 200/4236 kola
  ... 400/4236 kola
  ... 600/4236 kola
  ... 800/4236 kola
  ... 1000/4236 kola
  ... 1200/4236 kola
  ... 1400/4236 kola
  ... 1600/4236 kola
  ... 1800/4236 kola
  ... 2000/4236 kola
  ... 2200/4236 kola
  ... 2400/4236 kola
  ... 2600/4236 kola
  ... 2800/4236 kola
  ... 3000/4236 kola
  ... 3200/4236 kola
  ... 3400/4236 kola
  ... 3600/4236 kola
  ... 3800/4236 kola
  ... 4000/4236 kola
  ... 4200/4236 kola
Loto 7/39 — KORAK 1
Retrospektivni test (walk-forward, ceo CSV)
CSV: /Users/4c/Desktop/GHQ/data/loto7hh_4636_k49.csv
Kola u CSV: 4636
Testirano kola: 4236 (od indeksa 400)
Model koraka u testu: 2
Seed: 39

Prosek pogodaka model:   1.253
Prosek pogodaka slučajno: 1.247
Teorija (7 brojeva):     1.256
Pogodaka≥2 model:   36.9%
Pogodaka≥2 slučajno: 36.7%

Kombinacija: 05  07  08  10  14  31  32  skor=0.2592
.Skor: 0.2592

Korak 1 → /Users/4c/Desktop/GHQ/STATISTIKA/lottery_v2/loto7_korak1.txt
"""
