import cv2
import face_recognition
import numpy as np
import glob  

# Tải hình ảnh mẫu chứa khuôn mặt cần nhận diện
# Lấy tất cả các file .jpg trong thư mục data_img
image_paths = glob.glob('data_img/*.jpg')  # Sử dụng dấu / cho đường dẫn
reference_encodings = []

for img_path in image_paths:
    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        reference_encodings.append(encodings[0])

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Kiểm tra xem webcam có mở được không
if not cap.isOpened():
    print("Không thể mở webcam")
    exit()

while True:
    # Đọc frame từ webcam
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận frame")
        break

    # Chuyển frame sang RGB (face_recognition yêu cầu RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tìm tất cả các khuôn mặt và mã hóa chúng
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Duyệt qua từng khuôn mặt được phát hiện
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # So sánh khuôn mặt với khuôn mặt mẫu
        matches = face_recognition.face_distance(reference_encodings, face_encoding)
        name = "Unknown"

        # Nếu khoảng cách nhỏ hơn ngưỡng (0.3 là ngưỡng mới), coi như khớp
        if matches[0] < 0.4:
            name = "Known"

        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Ghi nhãn tên
        cv2.putText(frame, name, (left, top - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow('Face Recognition', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()