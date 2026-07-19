# Time Series Forecasting Project

A config-driven ARIMA/SARIMAX forecasting pipeline. Drop in your dataset, tweak `config.yaml`, and run.

## How to use

### 1. Add your dataset
Put your CSV file inside the `data/` folder, e.g. `data/dataset.csv`.

Your CSV needs at minimum:
- A date/datetime column (any parseable format)
- A numeric target column you want to forecast

### 2. Update `config/config.yaml`
```yaml
data:
  raw_data_path: "data/dataset.csv"   # your file name
  date_column: "Date"                 # your date column name
  target_column: "Close"              # your target column name
```
That's the only file you *need* to touch. Everything else (train/test split ratio,
seasonality, log transform, ARIMA search grid, forecast horizon) has sensible
defaults you can tune later.

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train
```bash
python src/pipeline/train_pipeline.py
```
This will:
- Load and sort your data chronologically
- Run an ADF stationarity test (logged, informational)
- Split train/test by time (never randomly)
- Grid search ARIMA (or SARIMAX if `model.seasonal: true`) over the p/d/q ranges in config
- Save the best model to `artifacts/model.pkl`
- Print RMSE / MAE / MAPE on the held-out test set

### 5. Predict
```bash
python src/pipeline/predict_pipeline.py
```
Or via Flask:
```bash
python app.py
# POST http://localhost:5000/train
# GET  http://localhost:5000/predict?steps=30
```

## Project structure
```
time_series_project/
├── config/config.yaml          # all tunables — edit this
├── data/                       # <-- put your CSV here
├── src/
│   ├── exception.py            # custom exception (sys.exc_info based)
│   ├── logger.py                # timestamped logging
│   ├── utils.py                 # save/load objects, ARIMA grid search, metrics
│   ├── components/
│   │   ├── data_ingestion.py       # load, parse dates, sort chronologically
│   │   ├── data_transformation.py  # ADF test, optional log transform, time-based split
│   │   └── model_trainer.py        # ARIMA/SARIMAX grid search + save best model
│   └── pipeline/
│       ├── train_pipeline.py       # orchestrates the full training run
│       └── predict_pipeline.py     # loads model, forecasts N steps ahead
├── artifacts/                  # saved model.pkl, train/test CSVs (auto-generated)
├── notebooks/eda.ipynb         # starter EDA notebook (ACF/PACF, decomposition)
├── app.py                      # Flask serving layer
├── requirements.txt
└── setup.py
```

## Notes
- Train/test split is always **chronological** — the model never sees the future during training.
- ARIMA `d` (differencing order) is searched directly, so you don't need to manually
  difference the series — the ADF test result is just there to sanity-check what `d` should look like.
- Set `model.seasonal: true` in config if your data has clear seasonality (e.g. weekly/monthly cycles).
- If your series has an exponential-looking trend, try `transformation.apply_log_transform: true`.

## Exogenous features (e.g. stock OHLCV data)
Set `data.exog_columns` in config to use other columns (e.g. `Volume`, `Open`, `High`, `Low`)
as regressors — this switches the model from plain ARIMA to SARIMAX with exog.

```yaml
data:
  target_column: "Close"
  exog_columns: ["Volume", "Open", "High", "Low"]
```

**Important limitation:** on the test set, exog values are real historical data, so the
reported RMSE/MAE/MAPE are trustworthy. But for genuinely *future* dates (beyond your
dataset), tomorrow's Volume/Open/High/Low are just as unknown as tomorrow's Close — the
pipeline can't know them either. `predict_pipeline.py` handles this by carrying the last
observed exog row forward for every forecast step (`future_exog.strategy: "last_value"`
in config). This is a simplifying assumption, not a real prediction — treat forecasts
beyond a few steps with proportionally more skepticism as this assumption gets less
realistic. Set `exog_columns: []` to fall back to plain ARIMA if you'd rather avoid this
assumption entirely.
