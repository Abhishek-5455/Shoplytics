
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ecommerce_ml.models.predict_delivery_time import predict_delivery_time
from ecommerce_ml.models.predict_late_delivery import predict_delivery_risk

app = FastAPI(
    title='Shoplytics-API',
    description='E-commerce delivery prediction services',
    version='1.0.0'
)

class DeliveryFeatures(BaseModel):
    purchase_hour: int = Field(ge=0, le=23)
    purchase_day_of_week: int = Field(ge=0, le=6)
    purchase_month: int = Field(ge=1, le=12)
    item_count: int = Field(ge=1)
    total_price: int = Field(ge=0)
    total_freight: float = Field(ge=0)

class LateDeliveryRequest(DeliveryFeatures):
    days_to_estimated_delivery: float = Field(ge=0)

class DeliveryTimeRequest(DeliveryFeatures):
    estimated_delivery_days: float = Field(ge=0)

@app.get('/health')
def health_check():
    return { "status": "ok" }

@app.post('/predict/late-delivery')
def predict_late_delivery(request: LateDeliveryRequest):
    try:
        features = pd.DataFrame([request.model_dump()])
        result = predict_delivery_risk(features)

        return {
            'late_delivery_probability': float(
                result['late_delivery_probability']
            ),
            'delivery_prediction': int(
                result['delivery_prediction']
            ),
            'prediction_label': (
                'late' if int(result['delivery_prediction'] == 1)
                else 'not_late'
            )
        }
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

@app.post('/predict/delivery-time')
def predict_delivery(request: DeliveryTimeRequest):
    try:
        features = pd.DataFrame([request.model_dump()])
        result = predict_delivery_time(features).iloc[0]

        return {
            'predicted_delivery_days': float(
                result['predicted_delivery_days']
            )
        }
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
