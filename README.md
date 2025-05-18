# 🧠 BAQ API Testing 


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

---

## 🧪 API Testing


### ✅ 1.  Check

```bash
curl http://localhost:9000
```

### 🌸 2. Predict Iris Flower Species
#### 2.1 
```bash
curl -X POST http://localhost:9000/predict/onetime
curl -X POST http://localhost:9000/predict/rollingapi
```
