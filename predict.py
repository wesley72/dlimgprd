import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import DogCatResNet
#from model import DogCatCNN
import torch.nn.functional as F


# Parse command line argument
parser = argparse.ArgumentParser()
parser.add_argument("--img_path", type=str, required=True)
args = parser.parse_args()

# Define preprocessing (resize to 128x128, convert to tensor, normalize)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load image
print(f"🔍 Trying to load image from: {args.img_path}")
img = Image.open(args.img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # add batch dimension

# Load model
model = DogCatResNet()
model.load_state_dict(torch.load("models/cnn_dog_cat.pth"))  # adjust filename if different
model.eval()

# Run prediction
with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)
    probs = F.softmax(output, dim=1)
confidence = probs[0][predicted.item()].item()

print(f"Confidence: {confidence:.2f}")


# Print result
classes = ["cat", "dog"]
print("Predicted class:", classes[predicted.item()])