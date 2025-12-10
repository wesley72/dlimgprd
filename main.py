from fastapi import FastAPI, UploadFile, File
from torchvision import models, transforms
from PIL import Image
import torch
import torch.nn.functional as F
import io

app = FastAPI(title="Dog-Cat Classifier API")
'''
# Load model once at startup
class_names = ["cat", "dog"]
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()
'''
class_names = ["cat", "dog"]

# ✅ Load pretrained ResNet18 backbone
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Replace final layer for 2 classes
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

# ✅ Fix: handle checkpoints saved with "model." prefix
state_dict = torch.load("model.pth", map_location="cpu")
if "model" in state_dict:  # if saved as {"model": state_dict}
    state_dict = state_dict["model"]

# Strip "model." prefix if present
new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict, strict=False)

model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #transforms.Normalize(mean=weights.meta["mean"], std=weights.meta["std"])
])

@app.get("/")
def root():
    return {"message": "Dog-Cat Prediction API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # Run inference
    outputs = model(img_tensor)
    probs = F.softmax(outputs, dim=1)

    # Format response
    predictions = {class_names[i]: float(probs[0][i]) for i in range(len(class_names))}
    predicted_class = class_names[torch.argmax(probs)]
    confidence = float(torch.max(probs))

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": predictions
    }