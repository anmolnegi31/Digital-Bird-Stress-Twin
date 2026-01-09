# 🐦 Digital Bird Stress Twin - COMPLETE GUIDE

## 🎯 PROJECT GOAL

**Predict natural disasters 24-72 hours in advance by analyzing bird stress behavior patterns.**

### How It Works
```
Normal Day: Bird Stress = 0.1-0.3 (calm)
    ↓
72h Before: Bird Stress = 0.3-0.5 (monitoring) 🟡
    ↓
48h Before: Bird Stress = 0.5-0.7 (warning) 🟠
    ↓
24h Before: Bird Stress = 0.7-0.9 (critical) 🔴
    ↓
DISASTER OCCURS: Earthquake/Storm/Flood
```

Birds act as natural sensors - they sense atmospheric/seismic changes before disasters!

---

## 🚀 QUICK START (5 Minutes)

### 1. Setup Environment
```powershell
cd DIGITAL-STRESS-TWIN

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies (if not already installed)
pip install streamlit==1.28.0 plotly==5.17.0
```

### 2. Launch Streamlit UI
```powershell
streamlit run app.py
```

Visit: **http://localhost:8501**

### 3. Collect Disaster Data (Historical)
```powershell
python scripts\collect_data.py --collect disasters --disaster-years 5
```

This collects 5 years of earthquake data from USGS for India.

### 4. Create Training Dataset
```powershell
python -c "
from pathlib import Path
from src.dataset_creator import create_dataset_from_raw_data

features_path = Path('data/raw/features.csv')  # After feature engineering
disasters_path = Path('data/raw/disasters_20210107_20260107.csv')
output_dir = Path('data/processed')

create_dataset_from_raw_data(features_path, disasters_path, output_dir)
"
```

### 5. Train LSTM Model
```powershell
python src\train.py --data data/processed/train_dataset.npz --epochs 50
```

---

## 📊 COMPLETE DATA FLOW

### Step 1: Data Collection
```
python scripts\collect_data.py --days 14 --collect all --disaster-years 5
```

Collects:
- **Bird observations** (eBird API)
- **Audio recordings** (Xeno-Canto API)
- **Weather data** (OpenWeatherMap API)
- **Disaster records** (USGS Earthquake API)

### Step 2: Feature Engineering
```python
# Process audio files
from src.feature_engineering import AudioFeatureExtractor

extractor = AudioFeatureExtractor()
features = extractor.extract_features("crow_call.mp3")
# Output: 63 audio features (MFCCs, spectral entropy, etc.)
```

### Step 3: Time Series Creation
```python
# Create 168-hour sequences with disaster labels
from src.dataset_creator import TimeSeriesDatasetCreator

creator = TimeSeriesDatasetCreator(sequence_hours=168)
dataset = creator.create_full_dataset(
    features_path="data/raw/features.csv",
    disasters_path="data/raw/disasters.csv",
    output_dir="data/processed"
)
# Output: train/val/test .npz files with shape (N, 168, 92)
```

### Step 4: Model Training
```python
# Train LSTM with attention
python src\train.py \
    --data data/processed/train_dataset.npz \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --hidden-size 128
```

### Step 5: Make Predictions
```python
# Real-time prediction
from src.models import StressPredictionLSTM

model = StressPredictionLSTM.load("models/best_model.pt")
stress_score = model.predict(current_168h_sequence)

if stress_score > 0.7:
    print("🚨 HIGH RISK: Disaster likely within 24-48 hours!")
```

---

## 🧠 MODEL ARCHITECTURE

### LSTM with Attention
```
Input: (batch, 168 hours, 92 features)
    ↓
Embedding Layer: 92 → 128
    ↓
Bidirectional LSTM Layer 1: hidden_size=128
    ↓
Bidirectional LSTM Layer 2: hidden_size=128
    ↓
Multi-Head Attention: 4 heads (focuses on critical hours)
    ↓
Dense Layers: 128 → 64 → 32
    ↓
Output Layer: Sigmoid activation
    ↓
Stress Score: 0.0-1.0
```

### Feature Breakdown (92 total)
- **Audio Features (63)**:
  - 13 MFCCs + 13 delta + 13 delta-delta = 39
  - Spectral: centroid, bandwidth, rolloff, entropy, flatness = 5
  - Chroma: 12 pitch classes = 12
  - RMS energy, ZCR = 2
  - Statistical: mean, std, min, max, median = 5

- **Weather Features (29)**:
  - Temperature, pressure, humidity, wind = 4
  - Pressure drops: 1h, 6h, 24h, 48h = 4
  - Temperature changes: 1h, 6h, 24h = 3
  - Weather gradients and derivatives = 10
  - Categorical: weather conditions (rain, cloudy) = 8

- **Temporal Features (8)**:
  - Hour (sin/cos encoding) = 2
  - Day of week (sin/cos) = 2
  - Month (sin/cos) = 2
  - Is night, is weekend = 2

---

## 🗂️ PROJECT STRUCTURE

```
DIGITAL-STRESS-TWIN/
├── app.py                          # 🎨 Streamlit UI (main entry)
├── .env                            # 🔐 API keys configuration
├── requirements.txt                # 📦 Python dependencies
│
├── src/                            # 🧠 Core source code
│   ├── data_ingestion/
│   │   ├── ebird_client.py         # eBird API
│   │   ├── xenocanto_client.py     # Audio recordings API
│   │   ├── weather_client.py       # Weather API
│   │   └── disaster_client.py      # 🆕 USGS earthquake API
│   │
│   ├── feature_engineering/
│   │   ├── audio_features.py       # 63 audio features
│   │   ├── environmental_features.py # 29 weather features
│   │   └── stress_index.py         # Stress calculation
│   │
│   ├── models/
│   │   ├── lstm_model.py           # LSTM with attention
│   │   ├── vae_model.py            # VAE for audio generation
│   │   └── trainer.py              # Training loops
│   │
│   ├── api/
│   │   └── main.py                 # FastAPI production server
│   │
│   ├── monitoring/
│   │   └── drift_detection.py      # Model monitoring
│   │
│   ├── dataset_creator.py          # 🆕 Time series dataset with disaster labels
│   └── train.py                    # 🆕 Training pipeline
│
├── scripts/
│   └── collect_data.py             # 🆕 Data collection (now includes disasters)
│
├── data/
│   ├── raw/
│   │   ├── ebird_observations_*.csv
│   │   ├── weather_data_*.csv
│   │   └── disasters_*.csv         # 🆕 Historical disaster records
│   └── processed/
│       ├── train_dataset.npz       # 🆕 Training sequences
│       ├── val_dataset.npz
│       └── test_dataset.npz
│
├── models/
│   ├── checkpoints/
│   └── best_model.pt
│
└── configs/
    ├── config.yaml                 # System configuration
    └── species_config.yaml         # Species-specific settings
```

---

## 🎓 TRAINING WORKFLOW

### Complete Training Pipeline

```powershell
# 1. Collect 5 years of historical disaster data
python scripts\collect_data.py --collect disasters --disaster-years 5

# 2. Collect recent bird/weather data (for current predictions)
python scripts\collect_data.py --days 30 --region IN --collect birds,weather

# 3. Extract audio features (if you have audio files)
python -c "
from src.feature_engineering import AudioFeatureExtractor
import glob

extractor = AudioFeatureExtractor()
for audio_file in glob.glob('data/raw/audio/*.mp3'):
    features = extractor.extract_features(audio_file)
    # Save features
"

# 4. Create time series dataset with disaster labels
python -c "
from src.dataset_creator import create_dataset_from_raw_data
from pathlib import Path

create_dataset_from_raw_data(
    features_path=Path('data/raw/features.csv'),
    disasters_path=Path('data/raw/disasters_*.csv'),
    output_dir=Path('data/processed')
)
"

# 5. Train LSTM model
python src\train.py \
    --data data/processed/train_dataset.npz \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001

# 6. Evaluate on historical disasters
python src\train.py \
    --data data/processed/test_dataset.npz \
    --mode evaluate \
    --model models/best_model.pt

# 7. Start production API
cd src
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## 🔮 MAKING PREDICTIONS

### Option 1: Streamlit UI
```powershell
streamlit run app.py
```
Navigate to "Live Predictions" page.

### Option 2: Python Script
```python
import torch
import numpy as np
from src.models import StressPredictionLSTM

# Load trained model
model = StressPredictionLSTM.load("models/best_model.pt")
model.eval()

# Prepare 168-hour sequence
# Shape: (168, 92)
current_sequence = np.load("data/current_168h_features.npy")

# Predict
with torch.no_grad():
    sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
    stress_score = model(sequence_tensor).item()

print(f"Current Stress Level: {stress_score:.3f}")

if stress_score > 0.7:
    print("🚨 CRITICAL: High probability of disaster within 24-48 hours!")
elif stress_score > 0.5:
    print("🟠 WARNING: Elevated stress detected, monitor closely")
elif stress_score > 0.3:
    print("🟡 MODERATE: Some environmental changes")
else:
    print("🟢 NORMAL: All systems normal")
```

### Option 3: FastAPI Endpoint
```bash
curl -X POST "http://localhost:8000/api/predict/stress" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Delhi",
    "latitude": 28.6139,
    "longitude": 77.2090
  }'
```

---

## 📈 HISTORICAL VALIDATION

### Test Against Known Disasters

```python
# Example: Validate against 2023 Delhi earthquake
from datetime import datetime

# Load model
model = StressPredictionLSTM.load("models/best_model.pt")

# Load 7 days of data BEFORE the earthquake
disaster_date = datetime(2023, 3, 15, 14, 30)  # March 15, 2023, 2:30 PM
data_start = disaster_date - timedelta(days=7)

# Get features for this period
features_df = pd.read_csv("data/historical/delhi_march_2023.csv")
sequence = features_df[data_start:disaster_date].values

# Predict
stress = model.predict(sequence)

print(f"Model prediction 1 day before earthquake: {stress:.3f}")
# Expected: > 0.7 (high stress)

# Actual earthquake occurred: March 15, 2023
# If stress > 0.7 on March 14 → SUCCESS ✅
```

---

## 🌍 DEPLOYMENT

### Option 1: Docker
```powershell
docker-compose up -d
```
Services start:
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Streamlit: http://localhost:8501

### Option 2: Cloud (AWS/GCP/Azure)
```bash
# Build image
docker build -t bird-stress-twin .

# Push to registry
docker tag bird-stress-twin:latest your-registry/bird-stress-twin

# Deploy to Kubernetes/ECS/Cloud Run
kubectl apply -f k8s/deployment.yaml
```

---

## 🎯 PERFORMANCE GOALS

### Target Metrics
- **Recall**: 90% (detect 9 out of 10 disasters)
- **Precision**: 84% (16% false alarm rate)
- **F1 Score**: 87%
- **Lead Time**: 36 hours average advance warning
- **Confidence**: 92% average model certainty

### Species Focus
- **Primary**: House Crow (Corvus splendens)
- **Secondary**: House Sparrow, Rock Pigeon
- **Reason**: Most sensitive to environmental changes, abundant data

### Locations Monitored
1. **Delhi NCR** - Seismic Zone IV (earthquake risk)
2. **Ahmedabad** - Zone III (earthquake)
3. **Mumbai** - Flood/cyclone risk
4. **Guwahati** - Zone V (high seismic)
5. **Srinagar** - Zone V (high seismic)

---

## 🐛 TROUBLESHOOTING

### API Key Errors
```powershell
# Check .env file exists and has all keys
cat .env

# Required keys:
# EBIRD_API_KEY=jqgchtjhgj8e
# XENO_CANTO_API_KEY=9136606b22b22128a3d2224ae36c00daf718d749
# OPENWEATHER_API_KEY=9bef7e48804cfa19fb9141d615334483
```

### Import Errors
```powershell
# Add src to Python path
$env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
```

### Model Not Found
```powershell
# Check model path
ls models\checkpoints\

# Train a model first if missing
python src\train.py --data data/processed/train_dataset.npz
```

---

## 📚 KEY CONCEPTS

### Digital Twin
A **virtual representation** of the physical bird population that:
- Continuously learns from real bird behavior
- Simulates "what-if" scenarios
- Predicts future stress states

### Time Series Labeling
```python
# How we calculate labels
hours_before_disaster = 36  # 1.5 days before earthquake

if hours_before_disaster <= 24:
    stress_label = 0.7 + (0.3 * (24 - hours) / 24)  # 0.7-1.0
elif hours_before_disaster <= 48:
    stress_label = 0.5 + (0.2 * (48 - hours) / 24)  # 0.5-0.7
elif hours_before_disaster <= 72:
    stress_label = 0.3 + (0.2 * (72 - hours) / 24)  # 0.3-0.5
else:
    stress_label = 0.1  # Normal
```

### Attention Mechanism
The model automatically focuses on critical time periods:
- **High attention**: Hours with rapid pressure drops
- **Low attention**: Stable periods
- **Result**: Model learns WHICH hours matter most

---

## 🎉 SUCCESS CRITERIA

Your system is working correctly if:

✅ **Data Collection**: Gathers bird/weather/disaster data successfully
✅ **Dataset Creation**: Produces .npz files with shape (N, 168, 92)
✅ **Training**: Model loss decreases, validation metrics improve
✅ **Historical Validation**: Predicts high stress before known disasters
✅ **Real-time Predictions**: API returns stress scores for current conditions
✅ **UI**: Streamlit app displays dashboards and predictions

---

## 📧 NEXT STEPS

1. **Collect Historical Data**: Run disaster collection for 5 years
2. **Feature Engineering**: Extract audio features from recordings
3. **Create Dataset**: Build time series sequences with labels
4. **Train Model**: LSTM training for 50+ epochs
5. **Validate**: Test against 2020-2024 disasters
6. **Deploy**: Launch production API and UI
7. **Monitor**: Track predictions vs actual events

---

**🎯 This is a production-grade ML system demonstrating:**
- Real-world data integration
- Time series modeling
- Deep learning (LSTM + Attention)
- MLOps best practices
- Full-stack deployment

**Perfect for a portfolio project showcasing end-to-end ML engineering!** 🚀
