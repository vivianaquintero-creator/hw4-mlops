# Olist Customer Satisfaction Prediction API

A REST API that predicts whether an Olist customer will leave a positive review (4-5 stars) based on order and delivery features. Built with Flask, containerized with Docker, and deployed on Render.com.

## Live URL
https://your-app.onrender.com

## Model Information
- **Model:** Tuned LightGBM Classifier (from HW2)
- **Task:** Binary classification — positive review (1) vs negative review (0)
- **Key Metrics:** AUC: 0.74, F1: 0.72, Accuracy: 0.73

## API Endpoints

### GET /health
Health check — confirms the API is running and model is loaded.

**Response:**
```json
{"status": "healthy", "model": "loaded", "model_type": "Pipeline", "n_features": 10}
```

### POST /predict
Single prediction. Send a JSON object with order features.

**Request:**
```json
{
  "delivery_days": 7,
  "delivery_vs_estimated": -2,
  "price": 149.99,
  "freight_value": 15.50,
  "item_category": "health_beauty",
  "seller_state": "SP",
  "payment_type_main": "credit_card",
  "num_items": 1,
  "order_hour": 14,
  "order_dayofweek": "Monday"
}
```

**Response:**
```json
{"prediction": 1, "probability": 0.73, "label": "positive"}
```

### POST /predict/batch
Batch prediction. Send a JSON array of up to 100 records.

**Response:**
```json
{"predictions": [{"prediction": 1, "probability": 0.73, "label": "positive"}], "count": 1}
```

## Input Schema

| Feature | Type | Description |
|---|---|---|
| delivery_days | int | Days from purchase to delivery |
| delivery_vs_estimated | int | Days early (negative) or late (positive) |
| price | float | Product price in BRL |
| freight_value | float | Shipping cost in BRL |
| item_category | string | Product category name |
| seller_state | string | Brazilian state code (e.g. SP) |
| payment_type_main | string | Payment method (credit_card, boleto, etc.) |
| num_items | int | Number of items in the order |
| order_hour | int | Hour of day the order was placed (0-23) |
| order_dayofweek | string | Day of week the order was placed |

## Local Setup

```bash
pip install -r requirements.txt
python app.py
# API runs on http://localhost:5000
```

## Docker

```bash
docker build -t hw4-api .
docker run -p 5000:5000 hw4-api
```

## Testing

```bash
python test_api.py                               # test local
python test_api.py https://your-app.onrender.com # test deployed
```