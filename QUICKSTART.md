# 🚀 Quick Start Guide

## Setup in 5 Minutes

### Step 1: Clone and Install

```bash
git clone <your-repo-url>
cd DIGITAL-STRESS-TWIN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API keys:
# - EBIRD_API_KEY=jqgchtjhgj8e
# - XENO_CANTO_API_KEY=9136606b22b22128a3d2224ae36c00daf718d749
# - OPENWEATHER_API_KEY=9bef7e48804cfa19fb9141d615334483
```

### Step 3: Test Data Ingestion

```bash
# Test eBird API
python -c "
from src.data_ingestion import create_ebird_client
client = create_ebird_client()
obs = client.get_recent_observations('IN', days=7)
print(f'✅ Found {len(obs)} bird observations')
"

# Test Weather API
python -c "
from src.data_ingestion import create_weather_client
client = create_weather_client()
weather = client.get_current_weather(city='Delhi')
print(f'✅ Weather: {weather.get(\"main\", {}).get(\"temp\")}°C')
"
```

### Step 4: Start API Server

```bash
cd src
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit: http://localhost:8000/docs

### Step 5: Make Your First Prediction

```bash
curl -X POST "http://localhost:8000/api/predict/stress" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Delhi",
    "species": "house_crow"
  }'
```

---

## Using Docker (Even Faster!)

```bash
# Start all services
docker-compose up -d

# Access:
# - API: http://localhost:8000
# - MLflow: http://localhost:5000
# - Docs: http://localhost:8000/docs
```

---

## Common Tasks

### Collect Data

```bash
python scripts/collect_data.py \
  --region IN \
  --days 14 \
  --species "Corvus splendens" \
  --collect all
```

### Train Model (with sample data)

```bash
# You need training data first!
# Once you have data/processed/training_data.csv:

python src/train.py \
  --data data/processed/training_data.csv \
  --run-name "my_first_model"
```

### Monitor MLflow

```bash
mlflow ui --port 5000
# Visit: http://localhost:5000
```

---

## Next Steps

1. ✅ API is running
2. 📊 Collect real data using the scripts
3. 🎓 Train your first model
4. 📈 Monitor with MLflow
5. 🚀 Deploy to production

---

## Troubleshooting

### API Key Errors

Make sure `.env` file exists and contains valid API keys.

### Import Errors

```bash
# Make sure you're in the project root
export PYTHONPATH="${PYTHONPATH}:${PWD}/src"
```

### Port Already in Use

```bash
# Kill process on port 8000
# Windows: netstat -ano | findstr :8000
# Linux/Mac: lsof -ti:8000 | xargs kill -9
```

---

## Example API Calls

### Get Weather Data

```bash
curl "http://localhost:8000/api/data/weather?location=Delhi"
```

### Get Bird Observations

```bash
curl "http://localhost:8000/api/data/observations?location=IN&days=7"
```

### Predict Stress with Forecast

```bash
curl -X POST "http://localhost:8000/api/predict/forecast" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "Mumbai",
    "latitude": 19.0760,
    "longitude": 72.8777
  }'
```

---

## Getting Help

- 📖 Full documentation: See `README.md`
- 🐛 Issues: Open a GitHub issue
- 💬 Questions: Check the docs or contact maintainers

**Happy bird stress prediction! 🐦📊**
