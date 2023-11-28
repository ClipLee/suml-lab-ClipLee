# kod naszego apifrom fastapi import FastAPI

from typing import Union

from fastapi import FastAPI

from models.point import Point

from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath("ml_models")
DATA_DIR = Path(BASE_DIR).joinpath("data")

app = FastAPI()


@app.get("/", tags=["into"])
async def index():
    return {"message": "Linear Regression ML"}



@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}
