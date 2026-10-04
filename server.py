import os
import io
import json
import uvicorn
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from predictor import TitanicPredictor

app = FastAPI(
    title="Titanic Survival & Mortality Predictor",
    description="Machine learning prediction API and batch death count simulator for the RMS Titanic manifest.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = TitanicPredictor()

class SinglePredictRequest(BaseModel):
    model_key: str = "svc"
    pclass: int = Field(default=3, ge=1, le=3)
    sex: str = Field(default="male")
    age: float = Field(default=29.0, ge=0.1, le=100.0)
    sibsp: int = Field(default=0, ge=0, le=10)
    parch: int = Field(default=0, ge=0, le=10)
    fare: float = Field(default=14.0, ge=0.0, le=600.0)
    embarked: str = Field(default="S")

class ManifestFilterRequest(BaseModel):
    model_key: str = "svc"
    pclass_filter: Optional[str] = "all"
    sex_filter: Optional[str] = "all"
    embarked_filter: Optional[str] = "all"

class CohortItem(BaseModel):
    label: Optional[str] = "Cohort"
    pclass: int = 3
    sex: str = "male"
    age: float = 28.0
    count: int = 50
    fare: float = 12.0
    sibsp: int = 0
    parch: int = 0
    embarked: str = "S"

class SimulateGroupRequest(BaseModel):
    model_key: str = "svc"
    cohorts: List[CohortItem]

# Endpoints
@app.get("/api/models")
def get_models():
    return predictor.get_models_list()

@app.post("/api/predict")
def predict_passenger(req: SinglePredictRequest):
    try:
        res = predictor.predict_single(
            model_key=req.model_key,
            pclass=req.pclass,
            sex=req.sex,
            age=req.age,
            sibsp=req.sibsp,
            parch=req.parch,
            fare=req.fare,
            embarked=req.embarked
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/death-count/manifest")
def predict_manifest(req: ManifestFilterRequest):
    try:
        res = predictor.predict_manifest_death_count(
            model_key=req.model_key,
            pclass_filter=req.pclass_filter,
            sex_filter=req.sex_filter,
            embarked_filter=req.embarked_filter
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/death-count/simulate")
def simulate_group(req: SimulateGroupRequest):
    try:
        cohort_dicts = [c.dict() for c in req.cohorts]
        res = predictor.simulate_custom_group_deaths(
            model_key=req.model_key,
            cohorts=cohort_dicts
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/historical-manifest")
def get_manifest_records(limit: int = 50, offset: int = 0, search: str = ""):
    return predictor.get_historical_manifest_records(limit=limit, offset=offset, search=search)

class CSVUploadRequest(BaseModel):
    filename: Optional[str] = "manifest.csv"
    model_key: Optional[str] = "svc"
    csv_text: str

@app.post("/api/upload-csv")
def upload_csv_and_predict(req: CSVUploadRequest):
    try:
        df = pd.read_csv(io.StringIO(req.csv_text))
        
        # Standardize column names (lowercase)
        df.columns = [c.strip().lower() for c in df.columns]
        
        required_cols = ['pclass', 'sex']
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"CSV missing required column '{col}'. Required columns: pclass, sex, age (optional), fare (optional), embarked (optional).")
                
        results = []
        total = len(df)
        death_count = 0
        survivor_count = 0
        
        for idx, row in df.iterrows():
            pclass = int(row.get('pclass', 3))
            sex = str(row.get('sex', 'male'))
            age = float(row['age']) if pd.notnull(row.get('age')) else 29.0
            sibsp = int(row['sibsp']) if pd.notnull(row.get('sibsp')) else 0
            parch = int(row['parch']) if pd.notnull(row.get('parch')) else 0
            fare = float(row['fare']) if pd.notnull(row.get('fare')) else 15.0
            embarked = str(row['embarked']) if pd.notnull(row.get('embarked')) else 'S'
            
            p = predictor.predict_single(req.model_key, pclass, sex, age, sibsp, parch, fare, embarked)
            if p['prediction'] == 0:
                death_count += 1
            else:
                survivor_count += 1
                
            results.append({
                'row': idx + 1,
                'name': str(row.get('name', f"Passenger #{idx+1}")),
                'pclass': pclass,
                'sex': sex,
                'age': age,
                'fare': fare,
                'prediction': p['prediction'],
                'outcome': p['outcome'],
                'survival_probability': p['survival_probability'],
                'death_probability': p['death_probability']
            })
            
        return {
            'filename': req.filename,
            'total_passengers': total,
            'predicted_deaths': death_count,
            'predicted_survivors': survivor_count,
            'mortality_rate_pct': round((death_count / max(1, total)) * 100, 1),
            'predictions': results[:200]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
