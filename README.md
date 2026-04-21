# Olist Customer Satisfaction Prediction API

For HW4, I depoloyed the LIght GBM model that I built in HW2 as a live REST API. The model will predict whether an Olist customer will leave a positive or negative review based on order and delivery feature. This will be more like a early warning system for dissatisfied customers. 

## Live URL
[https://your-app.onrender.com](https://hw4-mlops-k1h1.onrender.com) 

## Model Information
- **Model:** Tuned LightGBM Classifier (from HW2)
- **Task:** Binary classification — positive review (1) vs negative review (0)
- **Key Metrics:** AUC: 0.74, Recall: 0.52, F1: 0.72 for Negative Reviews
- **Features:** 12 order and delivery features available before review is written to avoid data leakage

## What I Learned
This was my first time deploying a machine learning model outside of a notebook. The biggest challenge was working in the terminal. Debugging Docker containers and dependency errors was completely new to me. Getting everything to work locally first, then in Docker, then live on Render was a big relief when it finally came together!

*AI assistance was used to help set up the Flask API, Docker configuration, and deployment steps.*

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
| freight_value | float | Shipping cost in BRL |
| product_category_name | string | Product category name |
| seller_state | string | Brazilian state code (e.g. SP) |
| payment_type_main | string | Payment method (credit_card, boleto, etc.) |
| seller_historical_average_rating | float | Seller's average rating |
| is_new_seller | int | Whether seller is new (0 or 1) |
| num_items | int | Number of items in the order |
| payment_value_total | float | Total payment value in BRL |
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
