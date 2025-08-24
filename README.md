
# S&P 500 Mean‑Reversion — Daily‑Only Backtest & Optimizer (v3)

**Sygnał sprawdzany codziennie.** Benchmark to **Buy & Hold S&P 500** (porównywany w każdym raporcie).
Wszystkie zmienne do zabawy masz **na górze** pliku `src/run_backtest.py` (sekcja `DEFAULT_CONFIG`)
oraz w **JSON-ie**: `config/config.json`.

## Idea strategii (w skrócie)
- Liczymy **trailing N‑day** (N = `lookback_days`) total return = ∏(1 + r_d) − 1 z dziennych zwrotów.
- **SELL → 0%** gdy trailing > `sell_th` (np. mediana ~ +11.1%).
- **BUY → 100%** gdy trailing ≤ `buy_th` (np. p30 ~ −1.7%).
- Zlecenie wykonujemy **następnego dnia** (`signal_lag_days`), bez shortów.
- Koszt transakcyjny w bps (np. 2 bps) nakładany przy każdej zmianie pozycji.
- Opcjonalne `min_hold_days` ogranicza „piłowanie”.

## Struktura
```
sp500-meanreversion-backtest-v3/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── config/
│   └── config.json
├── src/
│   ├── backtest_core.py
│   ├── run_backtest.py
│   └── optimize_thresholds.py
└── data/
    └── .gitkeep
```

## Instalacja
```
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie (dziennie, benchmark = B&H)
```
python src/run_backtest.py                 # użyje config/config.json
# lub szybkie nadpisanie progów/okna z CLI:
python src/run_backtest.py --buy_th -0.03 --sell_th 0.10 --lookback_days 252
```
Wyniki:
- CSV: `data/backtest_results.csv`
- Wykresy PNG: `data/equity_strategy.png`, `data/drawdown_strategy.png`, `data/equity_bench.png`, `data/drawdown_bench.png`

## Optymalizacja
```
# Grid (in-sample) pod Sharpe
python src/optimize_thresholds.py --metric sharpe --mode grid --step 0.01 --lookbacks 126,189,252,378,504

# Walk-forward (średnie OOS)
python src/optimize_thresholds.py --metric calmar --mode walk --n_splits 4 --test_years 3 --min_train_years 5
```
Podsumowanie zapisze się do: `data/optimizer_summary.json`.

## Dane
- Domyślnie: `yahoo_spy` (SPY, Adj Close ≈ total return). Możesz zmienić na `yahoo_gspc`, `stooq_spx`, `fred_sp500` lub `csv`.
- Dla `csv` ustaw `csv_path` i kolumny `date_col`, `price_col`.

## Uwaga metodyczna
- Dzienna częstotliwość zwiększa liczbę decyzji i ryzyko „piłowania” — rozważ `hysteresis` i `min_hold_days`.
- Unikaj przetrenowania: preferuj walk‑forward do oceny OOS.

**Edukacyjnie, nie jest to porada inwestycyjna.**
