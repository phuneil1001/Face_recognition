import cv2
import face_recognition
import numpy as np

# Tải hình ảnh mẫu chứa khuôn mặt cần nhận diện
reference_image = face_recognition.load_image_file(r"C:\Users\ADMIN\Desktop\face_recognition\data_img\WIN_20250507_18_34_52_Pro.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

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
        matches = face_recognition.face_distance([reference_encoding], face_encoding)
        name = "Unknown"

        # Nếu khoảng cách nhỏ hơn ngưỡng (0.6 là ngưỡng mặc định), coi như khớp
        if matches[0] < 0.6:
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