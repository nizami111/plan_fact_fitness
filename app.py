from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class CalcInput(BaseModel):
    values: List[float]
    multiplier: float = 1.0

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/calculate")
def calculate(data: CalcInput):
    total = sum(data.values)
    result = total * data.multiplier
    return {
        "sum": total,
        "multiplier": data.multiplier,
        "result": result
    }