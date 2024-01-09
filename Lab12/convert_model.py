import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

inital_type = {
    ("X", FloatTensorType([None, 1]))
}

Model = pickle.load(open("Lab12/ml_models/our_model.pkl", "rb"))
converted_model = convert_sklearn(Model, initial_types=inital_type)

with open("Lab12/ml_models/our_model.onnx", "wb") as f:
    f.write(converted_model.SerializeToString())

    # odpalac konwersje za pomoca python -m onnxruntime.tools.convert_onnx_models_to_ort Lab12/ml_models/our_model.onnx