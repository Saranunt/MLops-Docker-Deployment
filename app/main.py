from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from keras.models import load_model
import pandas as pd
import joblib
import os


def create_sequences(data, target_column, sequence_length):
    X = []
    feature_data = data.drop(target_column, axis=1).values

    for i in range(len(data) - sequence_length + 1):
        X.append(feature_data[i:i + sequence_length])

    return np.array(X)


def predict_model(model, input_data):
    y_pred = model.predict(input_data)
    return y_pred

def rolling_forecast(model, data, target_col, sequence_length, forecast_horizon):
    rolling_data = data.copy()
    rolling_forecast_df = pd.DataFrame(columns=['time', 'predicted_value'])

    for _ in range(forecast_horizon):
        input_sequence = create_sequences(rolling_data.tail(sequence_length), target_col, sequence_length)
        predicted_value = predict_model(model, input_sequence)

        new_row = rolling_data.tail(1)
        new_timestamp = pd.to_datetime(new_row.index[0]) + pd.Timedelta(hours=1)
        new_row.index = pd.DatetimeIndex([new_timestamp])

        rolling_data = pd.concat([rolling_data, new_row], ignore_index=True)

        rolling_forecast_df = pd.concat(
            [rolling_forecast_df, pd.DataFrame([[pd.to_datetime(rolling_data.tail(1).index), predicted_value]], columns=['time', 'predicted_value'])],
            ignore_index=True
        )

    return rolling_forecast_df

app = FastAPI()

modelsrc = 'app/lstm_model_3.h5'
datasrc = 'app/data.csv'

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI"}

@app.post("/predict/onetime")
async def predict_onetime():
    try:
        lstm_model = load_model(modelsrc, compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    try:
        data  = pd.read_csv(datasrc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")
    
    X = create_sequences(data, target_column='pm2_5_(μg/m³)', sequence_length=24)
    prediction = lstm_model.predict(X).reshape(-1, 1)

    return {"predictions": prediction.flatten().tolist()}

@app.post("/predict/rolling")
async def predict_rolling():
    try:
        lstm_model = load_model(modelsrc, compile=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {e}")

    try:
        data  = pd.read_csv(datasrc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading or preprocessing data: {e}")
    
    prediction = rolling_forecast(lstm_model, data, target_col='pm2_5_(μg/m³)', sequence_length=24, forecast_horizon=48)

    if hasattr(prediction, 'numpy'):
        prediction = prediction.numpy()
    prediction = np.array(prediction)
        
    if isinstance(prediction, pd.DataFrame):
        return {"predictions": prediction.values.flatten().tolist()}
    elif isinstance(prediction, np.ndarray):
        return {"predictions": prediction.flatten().tolist()}
    else:
        raise HTTPException(status_code=500, detail="Unsupported prediction output format.")


##debugger
# if __name__ == "__main__":
#     df = pd.read_csv(datasrc)
#     X = create_sequences(df, target_column='pm2_5_(μg/m³)', sequence_length=24   )
#     lstm_model = load_model(modelsrc, compile=False)
#     y_pred = lstm_model.predict(X)
#     y_pred = y_pred.reshape(-1, 1)
#     print(y_pred)

