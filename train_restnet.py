import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import multiprocessing


# Thêm hỗ trợ multiprocessing cho Windows
def train_resnet():
    # ====== Config ======
    data_dir = 'data/processe'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Kiểm tra thư mục dữ liệu
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Thư mục dữ liệu '{data_dir}' không tồn tại!")

    num_classes = len(os.listdir(data_dir))

    # ====== Dataloader ======
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Chuẩn hóa nhẹ
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    class_names = dataset.classes

    # ====== Model: ResNet-50 ======
    model = models.resnet50(pretrained=True)

    # Thay thế tầng FC cuối cùng
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # ====== Loss và Optimizer ======
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ====== Training loop ======
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"[Epoch {epoch + 1}] Loss: {running_loss:.3f} | Accuracy: {correct / total:.2%}")

    # ====== Save model ======
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/resnet50_face_classifier.pth")
    print("✅ Đã lưu mô hình ResNet-50.")


# Thêm cấu trúc bảo vệ cho multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support()  # Cần thiết cho Windows
    train_resnet()

#     print("❌ Nhấn 'q' để thoát.")

