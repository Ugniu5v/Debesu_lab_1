from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

app = FastAPI()

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()
preprocess = weights.transforms()


@app.get("/")
async def root():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.post("/predict")
async def predict(image: UploadFile):
    batch = preprocess(Image.open(image.file).convert("RGB")).unsqueeze(0)

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    return {
        "category": category_name,
        "confidence": f"{100 * score:.1f}%",
    }
