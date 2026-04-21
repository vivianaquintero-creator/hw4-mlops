#!/usr/bin/env python3
import sys
import requests

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"

SAMPLE = {
    "delivery_days": 7,
    "delivery_vs_estimated": -2,
    "freight_value": 15.50,
    "product_category_name": "health_beauty",
    "seller_state": "SP",
    "payment_type_main": "credit_card",
    "seller_historical_average_rating": 4.2,
    "is_new_seller": 0,
    "num_items": 1,
    "payment_value_total": 165.49,
    "order_hour": 14,
    "order_dayofweek": "Monday"
}

def test_health():
    print("\n--- Test 1: Health Check ---")
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()['status'] == 'healthy'
    print(f"  PASS: {r.json()}")

def test_single_prediction():
    print("\n--- Test 2: Single Prediction ---")
    r = requests.post(f"{BASE_URL}/predict", json=SAMPLE)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert 'prediction' in data
    assert 'probability' in data
    assert 'label' in data
    print(f"  PASS: prediction={data['prediction']}, prob={data['probability']}, label={data['label']}")

def test_batch_prediction():
    print("\n--- Test 3: Batch Prediction (5 records) ---")
    batch = [SAMPLE for _ in range(5)]
    r = requests.post(f"{BASE_URL}/predict/batch", json=batch)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    assert data['count'] == 5
    print(f"  PASS: {data['count']} predictions returned")

def test_missing_fields():
    print("\n--- Test 4: Missing Fields (expect 400) ---")
    r = requests.post(f"{BASE_URL}/predict", json={"delivery_days": 7})
    assert r.status_code == 400
    print(f"  PASS: Got 400 with error: {r.json()['error']}")

def test_invalid_type():
    print("\n--- Test 5: Invalid Type (expect 400) ---")
    bad = SAMPLE.copy()
    bad['freight_value'] = "not_a_number"
    r = requests.post(f"{BASE_URL}/predict", json=bad)
    assert r.status_code == 400
    print(f"  PASS: Got 400 with error: {r.json()['error']}")

if __name__ == '__main__':
    print(f"Testing API at: {BASE_URL}")
    print("=" * 50)
    tests = [test_health, test_single_prediction, test_batch_prediction,
             test_missing_fields, test_invalid_type]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'='*50}")
