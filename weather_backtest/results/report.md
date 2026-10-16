# Resultados del backtest clima NY -> S&P 500

Costo: 1 pb por lado. Permutaciones: 2000.

## Distribucion de dias

| Variante          |   Lindo |   Neutral |   Feo |
|:------------------|--------:|----------:|------:|
| 1_contemporaneo   |    0.13 |      0.48 |  0.39 |
| 2_manana_intradia |    0.19 |      0.43 |  0.38 |
| 3_rezagado        |    0.12 |      0.50 |  0.38 |

## Performance (periodo completo)

| Variante          | Estrategia            | CAGR   | Vol   |   Sharpe | MaxDD   | HitRate   | Exposicion   |   Dias |
|:------------------|:----------------------|:-------|:------|---------:|:--------|:----------|:-------------|-------:|
| 1_contemporaneo   | Long/Short clima      | -2.7%  | 13.6% |    -0.13 | -64.0%  | 47.8%     | 51.6%        |   5030 |
| 1_contemporaneo   | Solo long dias lindos | 1.0%   | 6.5%  |     0.19 | -12.4%  | 52.1%     | 12.5%        |   5030 |
| 1_contemporaneo   | Solo short dias feos  | -3.7%  | 12.0% |    -0.25 | -69.3%  | 46.4%     | 39.1%        |   5030 |
| 1_contemporaneo   | Long salvo dias feos  | 1.3%   | 14.8% |     0.16 | -51.0%  | 52.9%     | 60.9%        |   5030 |
| 1_contemporaneo   | Buy & hold            | 3.6%   | 19.1% |     0.28 | -56.8%  | 53.1%     | 100.0%       |   5030 |
| 2_manana_intradia | Long/Short clima      | -2.9%  | 13.4% |    -0.15 | -48.9%  | 46.8%     | 57.1%        |   3027 |
| 2_manana_intradia | Solo long dias lindos | 1.6%   | 7.2%  |     0.25 | -14.7%  | 53.5%     | 18.6%        |   3027 |
| 2_manana_intradia | Solo short dias feos  | -4.4%  | 11.3% |    -0.34 | -55.0%  | 43.6%     | 38.5%        |   3027 |
| 2_manana_intradia | Long salvo dias feos  | -0.9%  | 14.6% |     0.01 | -43.9%  | 52.1%     | 61.5%        |   3027 |
| 2_manana_intradia | Buy & hold            | 3.6%   | 18.5% |     0.28 | -55.7%  | 54.1%     | 100.0%       |   3027 |
| 3_rezagado        | Long/Short clima      | -2.1%  | 14.0% |    -0.08 | -60.5%  | 47.8%     | 49.6%        |   5030 |
| 3_rezagado        | Solo long dias lindos | 0.9%   | 6.5%  |     0.18 | -15.4%  | 52.0%     | 11.9%        |   5030 |
| 3_rezagado        | Solo short dias feos  | -3.0%  | 12.4% |    -0.19 | -58.8%  | 46.7%     | 37.7%        |   5030 |
| 3_rezagado        | Long salvo dias feos  | 2.1%   | 14.6% |     0.21 | -42.9%  | 52.8%     | 62.3%        |   5030 |
| 3_rezagado        | Buy & hold            | 3.6%   | 19.1% |     0.28 | -56.8%  | 53.1%     | 100.0%       |   5030 |

## Sub-periodos (L/S vs buy & hold)

| Variante          | Estrategia       | Periodo   | CAGR   | Vol   |   Sharpe | MaxDD   | HitRate   | Exposicion   |   Dias |
|:------------------|:-----------------|:----------|:-------|:------|---------:|:--------|:----------|:-------------|-------:|
| 1_contemporaneo   | Long/Short clima | 1999-2008 | 1.2%   | 15.5% |     0.15 | -38.4%  | 49.1%     | 53.4%        |   2514 |
| 1_contemporaneo   | Long/Short clima | 2009-2018 | -6.5%  | 11.5% |    -0.53 | -57.2%  | 46.4%     | 49.8%        |   2516 |
| 1_contemporaneo   | Buy & hold       | 1999-2008 | -3.0%  | 21.3% |    -0.04 | -51.9%  | 51.8%     | 100.0%       |   2514 |
| 1_contemporaneo   | Buy & hold       | 2009-2018 | 10.8%  | 16.6% |     0.70 | -27.6%  | 54.5%     | 100.0%       |   2516 |
| 2_manana_intradia | Long/Short clima | 1999-2008 | 8.1%   | 18.7% |     0.51 | -26.9%  | 48.8%     | 59.5%        |    672 |
| 2_manana_intradia | Long/Short clima | 2009-2018 | -5.8%  | 11.5% |    -0.47 | -48.0%  | 46.2%     | 56.4%        |   2355 |
| 2_manana_intradia | Buy & hold       | 1999-2008 | -8.5%  | 26.2% |    -0.21 | -53.5%  | 53.1%     | 100.0%       |    672 |
| 2_manana_intradia | Buy & hold       | 2009-2018 | 7.3%   | 15.6% |     0.53 | -23.4%  | 54.4%     | 100.0%       |   2355 |
| 3_rezagado        | Long/Short clima | 1999-2008 | 3.9%   | 15.8% |     0.32 | -36.9%  | 50.8%     | 52.2%        |   2514 |
| 3_rezagado        | Long/Short clima | 2009-2018 | -7.8%  | 11.8% |    -0.63 | -60.5%  | 44.6%     | 47.1%        |   2516 |
| 3_rezagado        | Buy & hold       | 1999-2008 | -3.0%  | 21.3% |    -0.04 | -51.9%  | 51.8%     | 100.0%       |   2514 |
| 3_rezagado        | Buy & hold       | 2009-2018 | 10.8%  | 16.6% |     0.70 | -27.6%  | 54.5%     | 100.0%       |   2516 |

## p-valor por permutacion (L/S, etiquetas barajadas dentro de cada estacion)

- 1_contemporaneo: p = 0.389
- 2_manana_intradia: p = 0.351
- 3_rezagado: p = 0.292

## Regresion retorno ~ nubosidad desestacionalizada (close->close, dia completo)

beta = 4.13 pb por pasar de despejado a cubierto, t = 0.66, n = 5030

## Retorno medio por clase — 1_contemporaneo

| Estacion   | Clase     |    N |   Media_bp |     t |   PctPositivo |
|:-----------|:----------|-----:|-----------:|------:|--------------:|
| Todos      | Lindo     |  629 |       5.40 |  1.18 |          0.53 |
| Todos      | Neutral   | 2435 |       1.34 |  0.54 |          0.53 |
| Todos      | Feo       | 1966 |       2.10 |  0.77 |          0.53 |
| Todos      | Lindo-Feo | 2595 |       3.30 |  0.62 |        nan    |
| Invierno   | Lindo     |  124 |      -7.72 | -0.77 |          0.52 |
| Invierno   | Neutral   |  506 |       5.03 |  0.98 |          0.55 |
| Invierno   | Feo       |  577 |      -3.45 | -0.74 |          0.52 |
| Invierno   | Lindo-Feo |  701 |      -4.27 | -0.39 |        nan    |
| Otono      | Lindo     |  186 |      14.59 |  1.75 |          0.56 |
| Otono      | Neutral   |  656 |      -0.31 | -0.06 |          0.53 |
| Otono      | Feo       |  412 |       2.40 |  0.30 |          0.51 |
| Otono      | Lindo-Feo |  598 |      12.19 |  1.06 |        nan    |
| Primavera  | Lindo     |  167 |       9.52 |  1.01 |          0.53 |
| Primavera  | Neutral   |  600 |       2.97 |  0.64 |          0.52 |
| Primavera  | Feo       |  508 |       8.47 |  1.74 |          0.54 |
| Primavera  | Lindo-Feo |  675 |       1.05 |  0.10 |        nan    |
| Verano     | Lindo     |  152 |       0.35 |  0.04 |          0.51 |
| Verano     | Neutral   |  673 |      -1.30 | -0.30 |          0.52 |
| Verano     | Feo       |  469 |       1.76 |  0.37 |          0.56 |
| Verano     | Lindo-Feo |  621 |      -1.41 | -0.14 |        nan    |

## Retorno medio por clase — 2_manana_intradia

| Estacion   | Clase     |    N |   Media_bp |     t |   PctPositivo |
|:-----------|:----------|-----:|-----------:|------:|--------------:|
| Todos      | Lindo     |  564 |       5.84 |  1.33 |          0.55 |
| Todos      | Neutral   | 1299 |       0.51 |  0.15 |          0.53 |
| Todos      | Feo       | 1164 |       1.98 |  0.59 |          0.55 |
| Todos      | Lindo-Feo | 1728 |       3.86 |  0.70 |        nan    |
| Invierno   | Lindo     |   82 |       2.51 |  0.20 |          0.55 |
| Invierno   | Neutral   |  277 |      -2.10 | -0.31 |          0.53 |
| Invierno   | Feo       |  334 |      -0.25 | -0.04 |          0.55 |
| Invierno   | Lindo-Feo |  416 |       2.76 |  0.20 |        nan    |
| Otono      | Lindo     |  156 |      -4.27 | -0.51 |          0.51 |
| Otono      | Neutral   |  375 |       4.96 |  0.63 |          0.57 |
| Otono      | Feo       |  255 |      -1.75 | -0.19 |          0.54 |
| Otono      | Lindo-Feo |  411 |      -2.53 | -0.20 |        nan    |
| Primavera  | Lindo     |  140 |       7.86 |  1.22 |          0.59 |
| Primavera  | Neutral   |  311 |       0.50 |  0.08 |          0.49 |
| Primavera  | Feo       |  283 |       9.64 |  1.68 |          0.55 |
| Primavera  | Lindo-Feo |  423 |      -1.78 | -0.21 |        nan    |
| Verano     | Lindo     |  186 |      14.28 |  1.64 |          0.54 |
| Verano     | Neutral   |  336 |      -2.28 | -0.43 |          0.52 |
| Verano     | Feo       |  292 |       0.36 |  0.06 |          0.57 |
| Verano     | Lindo-Feo |  478 |      13.92 |  1.32 |        nan    |

## Retorno medio por clase — 3_rezagado

| Estacion   | Clase     |    N |   Media_bp |     t |   PctPositivo |
|:-----------|:----------|-----:|-----------:|------:|--------------:|
| Todos      | Lindo     |  600 |       5.42 |  1.12 |          0.53 |
| Todos      | Neutral   | 2533 |       1.92 |  0.84 |          0.53 |
| Todos      | Feo       | 1897 |       1.40 |  0.48 |          0.53 |
| Todos      | Lindo-Feo | 2497 |       4.02 |  0.71 |        nan    |
| Invierno   | Lindo     |  119 |       7.58 |  0.84 |          0.52 |
| Invierno   | Neutral   |  524 |       0.96 |  0.21 |          0.53 |
| Invierno   | Feo       |  564 |      -3.20 | -0.62 |          0.54 |
| Invierno   | Lindo-Feo |  683 |      10.78 |  1.03 |        nan    |
| Otono      | Lindo     |  183 |      14.91 |  1.44 |          0.55 |
| Otono      | Neutral   |  680 |       0.55 |  0.10 |          0.52 |
| Otono      | Feo       |  391 |       1.01 |  0.13 |          0.53 |
| Otono      | Lindo-Feo |  574 |      13.91 |  1.07 |        nan    |
| Primavera  | Lindo     |  165 |      -1.91 | -0.21 |          0.48 |
| Primavera  | Neutral   |  625 |       7.98 |  1.87 |          0.55 |
| Primavera  | Feo       |  485 |       6.20 |  1.14 |          0.52 |
| Primavera  | Lindo-Feo |  650 |      -8.11 | -0.77 |        nan    |
| Verano     | Lindo     |  133 |      -0.49 | -0.05 |          0.58 |
| Verano     | Neutral   |  704 |      -1.40 | -0.35 |          0.51 |
| Verano     | Feo       |  457 |       2.32 |  0.44 |          0.55 |
| Verano     | Lindo-Feo |  590 |      -2.81 | -0.27 |        nan    |

## Robustez: grilla de 48 combinaciones de umbrales por variante

| variante          |   Sharpe_IS_mediana |   Sharpe_OOS_mediana |   pct_OOS_positivo |   corr_IS_OOS |
|:------------------|--------------------:|---------------------:|-------------------:|--------------:|
| 1_contemporaneo   |                0.04 |                -0.47 |               0.00 |          0.09 |
| 2_manana_intradia |                0.35 |                -0.53 |               0.00 |         -0.01 |
| 3_rezagado        |                0.21 |                -0.63 |               0.00 |          0.67 |
