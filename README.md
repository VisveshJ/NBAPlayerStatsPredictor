# NBAPlayerStatsPredictor

This project leverages **Hidden Markov Models (HMM)** and defensive matchup analytics to predict NBA player statistics for upcoming games. The goal is to provide a data-driven approach to forecasting individual performance by combining player-specific trends with opponent defensive strength.  

The hybrid methodology integrates three core components: a **HMM-based state analysis** to capture player form and streaks, **defensive rating adjustments** to scale predictions based on opponent quality, and **weighted historical comparisons** that prioritize recent performance and similar matchups. The system also evaluates prediction confidence by assessing data reliability and matchup history.  

Implementation Details: 
Built with Python (`hmmlearn`, `scikit-learn`, `pandas`), the pipeline processes NBA game logs and custom defensive ratings. Key steps include feature engineering (per-minute stats, rolling averages), Gaussian HMM training, and regression-based defensive adjustments. The model outputs are designed for flexibility, supporting use cases like fantasy basketball or tactical analysis.  

Alternatively, this project also provides an approach to predicting NBA player stats using regression analysis and defensive matchup metrics, without relying on Hidden Markov Models. The system focuses on quantifying the impact of opponent defensive strength while accounting for a player's recent performance trends.

The model combines three analytical techniques: defensive rating adjustments to scale predictions based on opponent quality, weighted rolling averages to emphasize recent performance, and matchup-based regression to identify historical trends against similar opponents. The system includes a confidence scoring mechanism that evaluates prediction reliability based on data quality and consistency of historical matchups.

| Player                                 | Actual Stats (P/R/A) | Hybrid HMM Output | Traditional HMM Output | Linear Adjustment** |
|----------------------------------------|------------------------|--------------------|-------------------------|----------------------|
| Tyrese Haliburton (PG, IND)            | 14 / 10 / 6           | 18 / 4 / 9         | 19 / 4 / 9              | 20 / 6 / 9           |
| Shai Gilgeous-Alexander (PG, OKC)      | 38 / 5 / 3            | 31 / 5 / 7         | 28 / 3 / 7              | 30 / 6 / 7           |
| Pascal Siakam (PF, IND)                | 19 / 10 / 3           | 22 / 7 / 3         | 20 / 7 / 3              | 24 / 5 / 4           |
| Aaron Nesmith (SF, IND)                | 10 / 12 / 1           | 13 / 5 / 1         | 9 / 3 / 1               | 14 / 5 / 1           |
| Andrew Nembhard (SG, IND)              | 14 / 5 / 6            | 13 / 5 / 6         | 11 / 3 / 5              | 11 / 4 / 5           |
| Isaiah Hartenstein (C, OKC)            | 9 / 9 / 0             | 9 / 10 / 4         | 13 / 11 / 4             | 9 / 7 / 1            |
| Jalen Williams (SF, OKC)               | 17 / 4 / 6            | 21 / 5 / 5         | 19 / 5 / 4              | 19 / 5 / 6           |
| Myles Turner (C, IND)                  | 15 / 9 / 0            | 16 / 6 / 2         | 16 / 6 / 2              | 16 / 4 / 1           |
| **Average Error per Stat (P/R/A)**     | —                     | **2.416**          | **2.958**               | **2.750**            |

