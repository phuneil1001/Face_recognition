from PIL import Image
import cv2
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
import mediapipe as mp
import numpy as np
import os

# ====== Config ======
model_path = "models/facenet_classifier.pth"
class_names = os.listdir("data/processed")  # Thư mục theo tên người
class_names.sort()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====== Load model ======
model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=len(class_names)).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ====== MediaPipe Face Detector ======
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# ====== Transform ảnh ======
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ====== Hàm crop mặt từ frame ======
def get_face(frame):
    h, w, _ = frame.shape
    results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        bbox = results.detections[0].location_data.relative_bounding_box
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        x2 = min(w, x + int(bbox.width * w))
        y2 = min(h, y + int(bbox.height * h))
        face = frame[y:y2, x:x2]
        return face, (x, y, x2, y2)
    return None, None

# ====== Webcam ======
cap = cv2.VideoCapture(0)
print("🎥 Webcam đang chạy... Nhấn Q để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face_img, bbox = get_face(frame)
    if face_img is not None:
        try:
            # Tiền xử lý - Chuyển đổi từ NumPy array sang PIL Image
            face_resized = cv2.resize(face_img, (160, 160))
            face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            # Dự đoán
            with torch.no_grad():
                output = model(face_tensor)
                pred = torch.argmax(output, dim=1).item()
                name = class_names[pred]
                prob = torch.softmax(output, dim=1)[0][pred].item()

            # Vẽ kết quả lên frame
            x, y, x2, y2 = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({prob:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print("⚠️ Lỗi xử lý khuôn mặt:", e)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()