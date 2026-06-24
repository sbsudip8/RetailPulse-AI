import os
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import mlflow, joblib
from pathlib import Path

def retrain():
    os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
    mlflow.set_tracking_uri(Path('mlruns').resolve().as_uri())
    mlflow.set_experiment('RetailPulse-Retrain')

    df = pd.read_csv('data/prophet_input.csv', parse_dates=['ds'])
    train, test = df[:-30], df[-30:]

    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(train)

    future_periods = (test['ds'].max() - train['ds'].max()).days
    future   = model.make_future_dataframe(periods=future_periods)
    forecast = model.predict(future)
    pred     = forecast.set_index('ds').reindex(test['ds'])['yhat']
    mape     = mean_absolute_percentage_error(test['y'], pred.values)

    with mlflow.start_run(run_name='auto_retrain'):
        mlflow.log_metric('mape', mape)
        mlflow.prophet.log_model(model, 'prophet_model')

    print(f'Retrain complete. MAPE: {mape*100:.2f}%')

if __name__ == '__main__':
    retrain()