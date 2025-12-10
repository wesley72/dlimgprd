import torch
import torch.nn as nn
from model import DogCatResNet   # ✅ correct
#from model import DogCatCNN
from dataset import get_loaders   # returns (train_loader, val_loader, test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model():
    # ✅ Load test loader
    _, _, test_loader = get_loaders()

    # ✅ Load trained model
    model = DogCatResNet().to(device)
    model.load_state_dict(torch.load("models/cnn_dog_cat.pth", map_location=device))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total

    print(f"📊 Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    evaluate_model()