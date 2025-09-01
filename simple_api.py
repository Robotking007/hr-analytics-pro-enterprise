"""
Simple FastAPI Backend - Minimal Dependencies
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = FastAPI(title="HR Analytics API", version="1.0.0")

class EmployeeData(BaseModel):
    name: str
    department: str
    position: str
    salary: float
    age: int

@app.get("/")
async def root():
    return {"message": "HR Performance Analytics Pro API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict")
async def predict_performance(employee: EmployeeData):
    # Simple prediction logic
    base_score = 0.7
    
    # Adjust based on salary (normalized)
    salary_factor = min(employee.salary / 100000, 1.0) * 0.1
    
    # Adjust based on age (experience curve)
    age_factor = 0.1 if employee.age > 30 else 0.05
    
    # Department factor
    dept_factors = {
        "Engineering": 0.15,
        "Sales": 0.12,
        "Marketing": 0.10,
        "HR": 0.08,
        "Finance": 0.09
    }
    dept_factor = dept_factors.get(employee.department, 0.08)
    
    predicted_score = base_score + salary_factor + age_factor + dept_factor
    predicted_score = min(predicted_score, 1.0)
    
    return {
        "employee": employee.name,
        "predicted_performance": round(predicted_score, 3),
        "confidence": 0.85,
        "factors": {
            "salary_factor": salary_factor,
            "experience_factor": age_factor,
            "department_factor": dept_factor
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
