# Vehicle CO2 Emissions Predictor

Predicts a vehicle's CO2 emissions (g/km) from its power, weight, fuel type, gearbox, body style and fuel consumption using linear regression with gradient descent written from scratch (pandas, matplotlib, seaborn, scikit-learn).

![AI101 Mini-Project](https://img.shields.io/badge/AI101-Mini--Project-blue) ![Team](https://img.shields.io/badge/Team-47-lightblue) ![Python](https://img.shields.io/badge/Python-3.10%20to%203.12-blue)

## Team and roles

**Team no.: 47**

| Member | Entry no. | GitHub handle | Role |
|---|---|---|---|
| Ishaan Shetye | 2026MCB1568 | Ishaan-1568 | Documentation lead |
| Prit Vora | 2026AIB1097 | prit-iitrpr | Execution lead |
| Ashish Meena | 2026ESB1473 | 2026esb1472-star | PPT lead |
| Geetansh Verma | 2026EEB1351 | gtansh-08 | Video lead |

*All four members contributed code (see commit history).*

## Project structure

```
co2-emissions-regression/
├── README.md
├── requirements.txt            # pinned library versions
├── main.py                     # entry point: runs everything, saves plots to outputs/
├── explore.py                  # (a) statistics and plots of the raw data
├── preprocess.py               # (b) cleaning, encoding, train/test split, scaling
├── train.py                    # (c) gradient descent from scratch, loss plots
├── evaluate.py                 # (d) metrics and plots for train and test data
├── co2_project.ipynb           # same pipeline as a notebook, with the final analysis
├── data/
│   └── vehicles_co2_emissions.csv    # 44,850 rows, 25 columns
├── outputs/                    # plots + metrics.csv (created by main.py)
└── docs/                       # slides (5) 
```

**Dataset:** `data/vehicles_co2_emissions.csv` (44,850 rows, 25 columns), source: https://www.kaggle.com/datasets/hetax17/france-vehicles-co2-emission-2013
. The target is `CO2(g/km)`. To keep things fast, the project uses a random sample of 10,000 rows (`random_state=42`).

## Setup

- **Python:** 3.10, 3.11 or 3.12 (tested on Python 3.12.3, Ubuntu). Not 3.13, because the pinned `numpy==1.26.4` has no build for it.
- **Libraries:** pinned in `requirements.txt` (`numpy 1.26.4`, `pandas 2.2.2`, `matplotlib 3.8.4`, `seaborn 0.13.2`, `scikit-learn 1.4.2`).
git clone https://github.com/[team-or-user]/co2-emissions-regression.git
cd co2-emissions-regression
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

1. Run `python main.py`
2. The script prints the statistics and the evaluation tables, and saves 11 plots and `metrics.csv` in `outputs/`.
3. Expected runtime: about 10 seconds. No internet is needed (the data is in the repo).

*Optional - notebook version with the written analysis:* `pip install notebook`, then `jupyter notebook`, open `co2_project.ipynb` and choose **Kernel > Restart & Run All**. Start Jupyter from the repository folder so the data path works.

## Expected output

Terminal screenshot after `python main.py`:

![Terminal output](docs/terminal_output.png)

*(abridged)* console output:

```text
===== STEP 1: UNDERSTAND THE DATA =====
Shape (rows, columns): (44850, 25)
...
Duplicate rows: 0

===== STEP 2: PREPROCESSING =====
Rows removed because of missing values: 16
Rows left: 9984
Training rows: 7987 | Testing rows: 1997

===== STEP 3: TRAINING (gradient descent from scratch) =====
Learning rate 0.001 -> final loss: 0.0742
Learning rate 0.01 -> final loss: 0.0201
Learning rate 0.1 -> final loss: 0.0135
Learning rate 0.3 -> final loss: 0.0134
Largest weight difference vs scikit-learn: 1e-05

===== STEP 4: EVALUATION =====

FULL MODEL (uses fuel consumption)
      Training  Testing
MAE     1.8320   1.7796
MSE    20.0517  17.9589
RMSE    4.4779   4.2378
R2      0.9866   0.9884

SPECS-ONLY MODEL (no fuel consumption)
      Training   Testing
MAE    11.4805   11.3974
MSE   235.9876  258.1937
RMSE   15.3619   16.0684
R2      0.8419    0.8330
```

**Key results** (test set): the full model reaches R² = 0.988 (average error about 1.8 g/km). Without fuel consumption (specs-only model) R² = 0.833 (average error about 11.4 g/km). Training and testing scores are close, so there is no overfitting, and our gradient descent matches scikit-learn.

| Predicted vs actual (full model) | Effect of each feature |
|---|---|
| ![Actual vs predicted](outputs/7_actual_vs_predicted_full.png) | ![Feature effects](outputs/11_feature_effects.png) |

All plots and `metrics.csv` are in [`outputs/`](outputs/).

## Links

- Slides (5): [docs/Team[NUMBER]_slides.pdf](docs/) [or paste the Drive link]
- Video (max 5:00): [paste the unlisted YouTube / Drive link]

## AI-use disclosure

- **Claude (Anthropic):** The initial draft for the code was written by each of the teammate, AI(Claude) was used to refine and present the code in a clearer manner.
Furthermore, it was used to assist in the process of documentation which the includes the Readme file aswell.
