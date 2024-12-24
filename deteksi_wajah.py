import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os

# Muat model YOLOv11 (pastikan file model YOLOv11 sudah ada di direktori yang benar)
model = torch.load('yolo11n.pt')  # Ganti dengan path file model .pt Anda
model.eval()  # Set model ke mode evaluasi

# Muat classifier Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi Streamlit
st.title("Deteksi Wajah dan Objek (Handphone) dari Foto, Video, dan Kamera Langsung")
st.write("Unggah foto/video atau pilih kamera untuk mendeteksi objek seperti handphone dan wajah.")

# Fungsi untuk deteksi objek dan wajah menggunakan YOLOv11
def detect_objects(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Deteksi objek menggunakan YOLOv11
    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()  # Convert frame to tensor
    img_tensor /= 255.0  # Normalisasi gambar
    img_tensor = img_tensor.unsqueeze(0)  # Menambah dimensi batch (1, C, H, W)
    
    with torch.no_grad():
        outputs = model(img_tensor)  # Deteksi objek

    # Proses hasil deteksi
    results = outputs[0]
    pred_classes = results[:, -1].cpu().numpy().astype(int)
    pred_boxes = results[:, :-1].cpu().numpy()
    
    height, width, _ = frame.shape
    boxes = []
    class_ids = []
    confidences = []

    for i in range(len(pred_classes)):
        if pred_classes[i] == 0:  # Misalnya, '0' untuk handphone (sesuaikan dengan ID kelas yang diinginkan)
            x1, y1, x2, y2 = pred_boxes[i]
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            confidences.append(results[i][4].cpu().item())
            class_ids.append(pred_classes[i])

    # NMS untuk mengurangi deteksi berlebih
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Gambar kotak dan label pada objek yang terdeteksi
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(pred_classes[i])
            confidence_text = f"{label} ({confidences[i]*100:.2f}%)"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Deteksi wajah
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Wajah Terdeteksi", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return frame

# Fungsi untuk deteksi pada foto
def detect_from_image(uploaded_image):
    image = Image.open(uploaded_image)
    frame = np.array(image)
    detected_frame = detect_objects(frame)
    return detected_frame

# Fungsi untuk streaming video dari file
def process_video(video_file):
    # Membuat file sementara untuk video yang diunggah
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(video_file.read())
        temp_file.close()  # Simpan file sementara

    # Membaca video dari file sementara
    cap = cv2.VideoCapture(temp_file.name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi objek dan wajah pada frame
        detected_frame = detect_objects(frame)

        # Convert frame ke format Image agar bisa ditampilkan oleh Streamlit
        img = Image.fromarray(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB))

        # Tampilkan gambar pada Streamlit
        st.image(img, channels="RGB", use_column_width=True)

        # Tunggu sejenak agar tampilan bisa di-update
        if st.button("Stop Video"):
            break

    cap.release()

# Fungsi untuk deteksi kamera langsung
def detect_from_camera():
    cap = cv2.VideoCapture(0)  # Buka kamera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Deteksi objek dan wajah pada frame
        detected_frame = detect_objects(frame)

        # Convert frame ke format Image agar bisa ditampilkan oleh Streamlit
        img = Image.fromarray(cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB))

        # Tampilkan gambar pada Streamlit
        st.image(img, channels="RGB", use_column_width=True)

        # Tunggu sejenak agar tampilan bisa di-update
        if st.button("Stop Kamera"):
            break

    cap.release()

# Bagian untuk memilih input (Foto, Video, atau Kamera Langsung)
uploaded_image = st.file_uploader("Pilih foto untuk diunggah", type=["jpg", "jpeg", "png"])
uploaded_video = st.file_uploader("Pilih video untuk diunggah", type=["mp4", "avi", "mov"])

# Menjalankan deteksi sesuai input yang dipilih
if uploaded_image is not None:
    detected_image = detect_from_image(uploaded_image)
    st.image(detected_image, caption="Hasil Deteksi Foto", channels="RGB", use_column_width=True)

elif uploaded_video is not None:
    process_video(uploaded_video)

else:
    if st.button("Deteksi Kamera Langsung"):
        detect_from_camera()
    else:
        st.write("Unggah foto atau video, atau pilih untuk menggunakan kamera langsung.")
        
if __name__ == "_main_":
    main()