import os
import torch
from torchvision import datasets, transforms, models
from torch import nn, optim


DATASET_DIR = "data/plantvillage"
MODEL_OUTPUT = "models/mobilenetv3_crop_disease.pth"

NUM_CLASSES = 10
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(
        DATASET_DIR,
        transform=transform
    )

    print("Classes:", dataset.classes)
    print("Images:", len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        NUM_CLASSES
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Loss: {running_loss / len(dataloader):.4f}"
        )

    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_OUTPUT
    )

    print("Model saved to:", MODEL_OUTPUT)


if __name__ == "__main__":
    main()
