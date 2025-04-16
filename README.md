# 🧠 ML Model Deployment Guide

**Project:** `mldeployment-cpe393`  
**Author:** _Bhagya Saranunt_ – `65070501092`

---

## 📦 Model Export

1. Train and export the model by running the training script:

```bash
python train.py
```

> This will save the model file as `model_iris.pkl` and `model_house.pkl` in the `app` directory.

---

## 🐳 Docker Instructions

### 1. Navigate to the Project Directory

```bash
cd path/to/your/project
```

### 2. Build the Docker Image

```bash
docker build -t ml-model .
```

### 3. Run the Docker Container

```bash
docker run -p 9000:9000 ml-model
```
![Docker Terminal](img\docker.png)

The Flask app will now be running at: [http://localhost:9000](http://localhost:9000)

---

## 🧪 API Testing

### Sample Prediction Result

![Test Result](img\test_result.png)

### ✅ 1. Health Check

```bash
curl http://localhost:9000/health
```

**Expected Output:**

```json
{ "status": "ok" }
```

---

### 🌸 2. Predict Iris Flower Species

```bash
curl -X POST http://localhost:9000/predict/iris ^
-H "Content-Type: application/json" ^
-d "{\"features\": [[5.1, 3.5, 1.4, 0.2], [6.5, 3.5, 1.8, 0.5]]}"
```

**Expected Output:**

```json
{
  "confidence": [
    [1.0, 0.0, 0.0],
    [0.93, 0.07, 0.0]
  ],
  "prediction": [0, 0]
}
```

---

### 🏠 3. Predict House Price

```bash
curl -X POST http://localhost:9000/predict/housing ^
-H "Content-Type: application/json" ^
-d "{\"features\": [[0, 7420, 4, 2, 3, \"yes\", \"no\", \"no\", \"no\", \"yes\", 2, \"yes\", \"furnished\"], [0, 3360, 2, 1, 1, \"yes\", \"no\", \"no\", \"no\", \"no\", 1, \"no\", \"unfurnished\"]]}"

```

**Expected Output:**

```json
{
  "predicted_prices": [8590821.4, 2484580.0]
}
```

> Note: The first value (`0`) is a placeholder if your model does not require the actual price in inputs.

---
