# api.py
from typing import Any, Dict, Optional
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pipeline import LgbmPipeline  # твой класс из pipeline.py


MODEL_PATH = "models/lgbm_final.pkl"
ARTIFACTS_PATH = "models/feature_artifacts.pkl"

app = FastAPI(title="Credit LGBM API")

# Глобальная переменная для пайплайна
pipe: Optional[LgbmPipeline] = None


# ---------- Схема для файла ----------

class FileRequest(BaseModel):
    file_path: str           # локальный путь, например "test_dataset/test_unprepeated_data.parquet"
    file_type: Optional[str] = None  # "csv" или "parquet"; если None — угадаем по расширению
    sep: str = ","


# ---------- Старт приложения: загружаем модель ----------

@app.on_event("startup")
def load_pipeline() -> None:
    global pipe
    print("Loading LGBM pipeline...")
    try:
        pipe = LgbmPipeline.from_files(
            model_path=MODEL_PATH,
            artifacts_path=ARTIFACTS_PATH,
        )
    except Exception as e:
        # Если тут упадёт — сервер не стартанёт, и в консоли будет понятный traceback
        print("ERROR while loading pipeline:", e)
        raise
    print("Pipeline loaded OK")


# ---------- Эндпоинты ----------

@app.get("/")
async def root():
    return {"status": "ok", "message": "LGBM pipeline is ready"}


@app.post("/predict/json")
async def predict_json(record: Dict[str, Any]):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=500, detail="Pipeline is not loaded")

    try:
        result = pipe.predict_one(record)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return result  # {"id": ..., "proba": ...}


@app.post("/predict/file")
async def predict_file(req: FileRequest):
    global pipe
    if pipe is None:
        raise HTTPException(status_code=500, detail="Pipeline is not loaded")

    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {path}")

    # определяем тип, если не указали
    file_type = req.file_type
    if file_type is None:
        if path.suffix.lower() == ".csv":
            file_type = "csv"
        elif path.suffix.lower() in {".parquet", ".pq"}:
            file_type = "parquet"
        else:
            raise HTTPException(status_code=400, detail="Unknown file type, set file_type explicitly")

    try:
        if file_type == "csv":
            df = pd.read_csv(path, sep=req.sep)
        elif file_type == "parquet":
            df = pd.read_parquet(path, engine="pyarrow")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file_type: {file_type}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if "id" not in df.columns:
        raise HTTPException(status_code=400, detail="Input file must contain 'id' column")

    pred_df = pipe.predict_batch(df.to_dict(orient="records"))
    return pred_df.to_dict(orient="records")
