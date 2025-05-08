from facenet_pytorch import InceptionResnetV1
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import multiprocessing

# Thêm dòng này để hỗ trợ multiprocessing trên Windows
multiprocessing.freeze_support()


def train_model():
    # ==== 1. Setup ====
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/processed'

    # Kiểm tra thư mục dữ liệu
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Thư mục dữ liệu '{data_dir}' không tồn tại!")

    # Kiểm tra xem có thư mục lớp nào không
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"Không tìm thấy thư mục lớp nào trong '{data_dir}'. "
                                f"Mỗi người cần được đặt trong thư mục riêng.")

    print(f"Tìm thấy {len(subdirs)} thư mục lớp: {', '.join(subdirs)}")

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    class_names = dataset.classes
    print(f" Có {len(class_names)} người trong tập huấn luyện.")

    # ==== 2. Mô hình ====
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ==== 3. Train ====
    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print(f"[Epoch {epoch + 1}] Loss: {running_loss:.3f} | Accuracy: {correct / total:.2%}")

    # ==== 4. Lưu model ====
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/facenet_classifier.pth")
    print("✅ Đã lưu mô hình.")


# Thêm cấu trúc bảo vệ cho multiprocessing
if __name__ == '__main__':
    multiprocessing.freeze_support() 
    train_model()