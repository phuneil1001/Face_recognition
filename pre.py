import cv2
import numpy as np
import face_recognition
from pathlib import Path

# ======== SETUP =========
TARGET_SIZE = (160, 160)

# ======== HÀM AUGMENTATION =========
def augment_face(face_img):
    augmented_faces = [face_img]

    # 1. Flip ngang
    flipped = cv2.flip(face_img, 1)
    augmented_faces.append(flipped)

    # 2. Tăng/giảm sáng
    def adjust_brightness(img, factor):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    for factor in [0.8, 1.2]:
        augmented_faces.append(adjust_brightness(face_img, factor))

    # 3. Làm mờ
    blurred = cv2.GaussianBlur(face_img, (5, 5), 0)
    augmented_faces.append(blurred)

    # 4. Xoay ±10 độ
    def rotate_image(img, angle):
        M = cv2.getRotationMatrix2D((TARGET_SIZE[0]//2, TARGET_SIZE[1]//2), angle, 1)
        return cv2.warpAffine(img, M, TARGET_SIZE)

    for angle in [-10, 10]:
        augmented_faces.append(rotate_image(face_img, angle))

    return augmented_faces

# ======== DÒ MẶT VÀ CẮT =========
def detect_and_crop_face(image):
    # Chuyển ảnh sang RGB vì face_recognition yêu cầu định dạng này
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Phát hiện khuôn mặt
    face_locations = face_recognition.face_locations(rgb_image)

    if not face_locations:
        return None

    # Lấy khuôn mặt đầu tiên (top, right, bottom, left)
    top, right, bottom, left = face_locations[0]
    # Cắt khuôn mặt
    face_img = image[top:bottom, left:right]
    # Resize về kích thước mục tiêu
    return cv2.resize(face_img, TARGET_SIZE)

# ======== TIỀN XỬ LÝ MỘT FILE ẢNH =========
def preprocess_single_image(input_path, output_dir):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"❌ File đầu vào {input_path} không tồn tại")
        return

    print(f"Đang xử lý: {input_path}")
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"❌ Không thể đọc ảnh: {input_path}")
        return

    face = detect_and_crop_face(image)
    if face is None:
        print(f"❌ Không tìm thấy mặt trong: {input_path.name}")
        return

    aug_faces = augment_face(face)
    for i, face_img in enumerate(aug_faces):
        save_path = output_dir / f"{input_path.stem}_aug{i}.jpg"
        success = cv2.imwrite(str(save_path), face_img)
        if success:
            print(f"✅ Đã lưu: {save_path}")
        else:
            print(f"❌ Lỗi khi lưu: {save_path}")

def preprocess_all_images_in_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"❌ Thư mục đầu vào {input_dir} không tồn tại")
        return

    for img_path in input_dir.glob("*.[jJ][pP][gG]"):  # Lấy tất cả các file .jpg hoặc .JPG
        print(f"Đang xử lý: {img_path}")
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"❌ Không thể đọc ảnh: {img_path}")
            continue

        face = detect_and_crop_face(image)
        if face is None:
            print(f"❌ Không tìm thấy mặt trong: {img_path.name}")
            continue

        aug_faces = augment_face(face)
        for i, face_img in enumerate(aug_faces):
            save_path = output_dir / f"{img_path.stem}_aug{i}.jpg"
            success = cv2.imwrite(str(save_path), face_img)
            if success:
                print(f"✅ Đã lưu: {save_path}")
            else:
                print(f"❌ Lỗi khi lưu: {save_path}")

if __name__ == "__main__":
    # Đường dẫn file ảnh đầu vào và thư mục đầu ra
    # preprocess_single_image("data_input/ducanh/image1.jpg", "data_output/processed")
    preprocess_all_images_in_folder("data_input/ducanh", "data_output_2/processed")