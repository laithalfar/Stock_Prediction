# Stock Prediction 📈

Predicting a stock’s **daily closing price** using neural networks, with a focus on **time-series modelling**, **walk-forward validation**, and **realistic evaluation** of model performance.

This project is structured as a small, reproducible ML pipeline rather than a single notebook experiment.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Goals](#goals)
- [Data](#data)
- [Methodology](#methodology)
  - [Feature Engineering](#feature-engineering)
  - [Modelling](#modelling)
  - [Evaluation](#evaluation)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [How to Use](#how-to-use)
  - [Notebooks](#notebooks)
  - [Python Modules](#python-modules)
- [Reports](#reports)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

The aim of this repository is to build and evaluate neural-network models for **univariate and/or multivariate stock price prediction**, with the primary target being the **closing price**.

Key ideas:

- Treating the problem as a **time-series forecasting** task, not a random regression problem.
- Using **sliding / walk-forward validation** to better approximate “real life” deployment.
- Comparing different architectures (e.g. dense networks vs recurrent architectures such as LSTM/GRU, depending on the active branch of the project).
- Keeping the work reproducible through a clear structure: `data/`, `notebooks/`, `src/`, `models/`, and `reports/`.

---

## Goals

- Build a neural network that predicts the **next-day closing price** given recent history and engineered features.
- Use **proper time-aware splits** (no leakage from the future).
- Track and document performance using sensible metrics (e.g. MAE, RMSE, MAPE / sMAPE).
- Serve as a portfolio-quality project that is:
  - Easy to read,
  - Easy to re-run,
  - Easy to extend.

---

## Data

The repository assumes you have historical stock data in CSV format under the `data/` directory.

Typical columns (depending on the dataset you use):

- `Date`
- `Open`
- `High`
- `Low`
- `Close`
- `Adj Close`
- `Volume`

You can:

- Place your raw CSVs under `data/raw/`
- Store cleaned / processed versions under `data/processed/` (depending on your local workflow)

*(Directory details can be adapted to your own structure, but the idea is to keep raw vs processed clearly separated.)*

---

## Methodology

### Feature Engineering

Typical transformations used in this project include:

- Sorting strictly by time (`Date`) and using only past information.
- Creating **lag features** of the closing price (e.g. `Close_t-1`, `Close_t-5`, `Close_t-10`, …).
- Optionally creating **technical indicators**, such as:
  - Moving averages (e.g. 5-day, 10-day, 20-day).
  - Rolling volatility (rolling standard deviation).
  - Daily returns or log-returns.
- Scaling numeric features using something like `StandardScaler` / `MinMaxScaler` from `scikit-learn`.

The exact configuration of features is controlled via Python code in `src/` and settings in `config.py`.

### Modelling

Core ideas on the modelling side:

- Neural-network based regression on time-series data.
- Experiments with different architectures, for example:
  - Fully-connected feedforward networks on lagged features.
  - Recurrent architectures (LSTM / GRU) treating the history as a sequence.
- Hyperparameters such as:
  - Window length (how many past days are used as input),
  - Hidden units / layers,
  - Learning rate,
  - Batch size,
  - Train/validation/test time ranges,
  are configured via `config.py` and/or notebook parameters.

Typical stack (as defined in `requirements.txt`):

- `Python 3.x`
- `numpy`, `pandas`
- `scikit-learn`
- `matplotlib` / `seaborn` (for plotting)
- `tensorflow` / `keras` or `torch` (depending on the model implementation in this repo)

### Evaluation

Time-series evaluation is done with **chronological splits**, not random splits, to avoid data leakage.

Common metrics:

- **MAE** – Mean Absolute Error  
- **RMSE** – Root Mean Squared Error  
- **MAPE / sMAPE** – percentage error metrics (when applicable)  
- Visual comparison of **predicted vs actual closing prices** over time.

Where relevant, **walk-forward validation** (rolling re-fit or rolling evaluation) is used to simulate re-training over time.

---

## Repository Structure

At a high level:

```text
Stock_Prediction/
├── data/          # Raw and/or processed stock data (not all tracked in git)
├── models/        # Saved model weights / artifacts (optional, usually gitignored)
├── notebooks/     # Jupyter notebooks for EDA and experiments
├── reports/       # Generated reports, figures, markdown summaries
├── src/           # Reusable Python modules for data prep, modelling, evaluation
├── config.py      # Central configuration (paths, model params, etc.)
├── notes.md       # Personal / development notes
├── report.md      # Project report in markdown format
├── requirements.txt
└── README.md
```

Setup
Clone the repository:
git clone https://github.com/laithalfar/Stock_Prediction.git
cd Stock_Prediction

Create and activate a virtual environment (recommended):
python -m venv .venv

# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate

# Install dependencies:
1. pip install --upgrade pip


2. pip install -r requirements.txt


# How to Use
## Notebooks
If you mainly want to explore the project interactively (EDA, visualisations, trying out ideas), use the notebooks.

Start Jupyter:
jupyter lab
 or
jupyter notebook

Open the relevant notebook in notebooks/, for example:

**notebooks/01_exploration.ipynb** – exploratory data analysis

**notebooks/02_modeling.ipynb** – model training and evaluation

**notebooks/03_walk_forward.ipynb** – walk-forward evaluation workflow

(Names above are illustrative; adjust them to the actual notebook names in your clone.)

Run the cells from top to bottom. Make sure your data files under data/ match what the notebook expects.


## Python Modules
The src/ directory contains reusable code so that the logic is not locked into notebooks.
You might find modules along the lines of:

    **src/data_utils.py** – loading and pre-processing raw CSVs.

    **src/features.py** – feature engineering helpers (lags, rolling windows, indicators).

    **src/model_defs.py** – model-building functions (e.g. to create an LSTM or dense network).

    **src/train.py** – training loop or CLI entry point (if implemented).

    **src/evaluation.py** – evaluation metrics and plotting utilities.

A typical pattern for running a training script could be:

    python -m src.train or,

if there is a main script that uses config.py:

    python src/train.py --config config.py

Check the actual file names and docstrings in src/ to see the intended entry point in your version of the project.

## Reports
The reports/ directory, together with report.md, contains:

    - Plots comparing actual vs predicted closing prices.

    - Error metrics summarised across validation / test periods.

    - Short write-ups of findings, limitations, and conclusions.

If you are just reviewing the project (e.g. as a recruiter or collaborator), report.md is a good place to start after skimming this README.

## Future Work
Possible extensions for this project include:

    - Multi-step forecasting (predict multiple days ahead).

    - Multi-asset models – training on several tickers simultaneously.

Adding more market features:

    - Index data (e.g. S&P 500),

    - Sector ETFs,

    - Macro indicators.

## Model comparison:

- ARIMA / SARIMA,

- Prophet,

- Gradient-boosted trees (XGBoost, LightGBM),

- More advanced deep-learning architectures (Temporal CNNs, Transformers).

- Packaging the pipeline into a single CLI command or a small web API.


## License
This project is currently provided as-is for educational and experimental purposes.
You can add a formal license (MIT, Apache 2.0, etc.) here once you decide how you’d like others to use it.

## Contact
Maintainer: Laith Al-Far
GitHub: @laithalfar
For collaboration or questions, feel free to open an issue on the repository.

You can tweak:

- The notebook filenames in the README to match your real ones.
- The description of modules inside `src/` to match your actual file names.
- Add specific metrics / results once you’re happy with them (e.g. “Best model achieved X MAE on the test set”).
::contentReference[oaicite:0]{index=0}
