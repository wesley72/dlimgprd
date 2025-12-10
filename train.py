import os
import torch
import torch.optim as optim
import torch.nn as nn
from model import DogCatResNet
#from model import DogCatCNN
from dataset import get_loaders   # returns (train_loader, val_loader, test_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(epochs=5):
    # ✅ Loaders
    train_loader, val_loader, test_loader = get_loaders()
    model = DogCatResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # weight decay for regularization
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # ---- Training Loop ----
        total_loss = 0
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ---- Validation Loop ----
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Validation Accuracy: {val_acc:.2f}")

    # ---- Save Model ----
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/cnn_dog_cat.pth")
    print("✅ Model saved at models/cnn_dog_cat.pth")

    # ---- Final Test Evaluation ----
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_acc = correct / total
    print(f"📊 Final Test Accuracy: {test_acc:.2f}")

    return model

if __name__ == "__main__":
    train_model(epochs=10)   # train longer for better generalization