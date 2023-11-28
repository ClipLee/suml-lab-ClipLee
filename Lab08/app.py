# kod naszego apifrom fastapi import FastAPI
# upewnic sie ze jestes w katalogu i odpalic za pomoca `uvicorn app:app --reload`

from typing import Union
from libs.task1 import train, get_predict
from libs.task2 import train, get_predict


from fastapi import FastAPI, HTTPException, Form
from typing import Annotated
import uvicorn

from models.point import Point

from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath("ml_models")
DATA_DIR = Path(BASE_DIR).joinpath("data")

app = FastAPI()


@app.get("/", tags=["into"])
async def index():
    return {"message": "Linear Regression ML"}


@app.get("/Lab08/.../point", response_model=Point, status_code=200)
async def poin(x: Annotated[int, Form()], y: Annotated[int, Form()]):
    return Point(x=x, y=y)


@app.post("model/train", tags=["model"], status_code=200)
async def train_model(data: Point, data_name="10_points", model_name="our_model"):
    data_file = Path(DATA_DIR).joinpath(f"{data_name}.csv")
    model_file = Path(MODEL_DIR).joinpath(f"{model_name}.pkl")

    data = data.dict()
    x = data["x"]
    y = data["y"]

    train(x, y, data_file, model_file)

    response_object = {"model_fit": "OK", "model_saved": "OK"}
    return response_object


@app.post("model/predict", tags=["model"], response_model=Point, status_code=200)
async def get_prediction(data: Point, model_name="our_model"):
    model_file = Path(MODEL_DIR).joinpath(f"{model_name}.pkl")

    if not model_file.exists():
        raise HTTPException(status_code=400, detail="Model not found")

    data = data.dict()
    x = data["X"]

    y_pred = get_predict(x=x, ml_model=model_file)
    data["y"] = y_pred[0][0]

    return Point(x=x, y=y_pred[0][0])

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)
