from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter
import base64  # Untuk encoding gambar ke base64
import pygame
import random

st.markdown("""
<style>
.navbar {
    background-color: #303b32; /* Warna latar belakang */
    padding: 20px; /* Memberikan lebih banyak ruang di dalam navbar */
    text-align: center; /* Pusatkan elemen di dalam navbar */
    font-size: 18px; /* Ukuran font navbar */
    height: 60px; /* Tinggi navbar */
    line-height: 20px; /* Untuk mengatur jarak vertikal teks */
}
.navbar a {
    color: white; /* Warna teks */
    text-decoration: none; /* Hapus garis bawah pada teks */
    padding: 15px; /* Tambahkan ruang di sekitar teks */
    font-weight: bold; /* Membuat teks lebih tebal */
    font-size: 36px; /* Ukuran font tautan */
}
.navbar a:hover {
    background-color: #008000; /* Warna latar saat hover */
    border-radius: 5px; /* Sudut membulat */
}
.sinyal-container {
    position: fixed;
    bottom: 20px;
    left: 20px;
    background-color: #303b32; /* Warna latar belakang sinyal */
    color: white;
    padding: 10px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: bold;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); /* Tambahkan efek bayangan */
    z-index: 1000;
    display: flex;
    align-items: center;
}
.sinyal-container img {
    width: 24px;
    height: 24px;
    margin-right: 8px;
}
</style>
<div class="navbar">
    <a href="#section1">🍀 DETEKSI OBJEK ADITIYA 🍀</a>
</div>
<div class="sinyal-container">
    <img src="https://cdn-icons-png.flaticon.com/512/107/107802.png" alt="Signal Icon">
    Sinyal Bagus Cuy
</div>
""", unsafe_allow_html=True)

# Fungsi untuk menambahkan latar belakang
def set_background(image_path, size="cover"):
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode()
    
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image;base64,{base64_image}");
        background-size: {size};  /* Menentukan ukuran gambar latar belakang */
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)

# Fungsi untuk menampilkan GIF di sidebar dengan bentuk lingkaran dan digeser atas-bawah
def display_animation(gif_path):
    with open(gif_path, "rb") as file:
        gif_data = file.read()
    base64_gif = base64.b64encode(gif_data).decode()
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; height: 250px; overflow-y: auto;">
            <img src="data:image/gif;base64,{base64_gif}" alt="Animation" style="width: 200px; height: 200px; border-radius: 50%; object-fit: cover;">
        </div>
        """,
        unsafe_allow_html=True
    )


# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Process and display the detection results
def display_results(image, results):
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results.boxes.conf.cpu().numpy()  # Confidence scores
    labels = results.boxes.cls.cpu().numpy()  # Class indices
    names = results.names  # Class names
    
    detected_objects = []
    
    for i in range(len(boxes)):
        if scores[i] > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(labels[i])]
            score = scores[i]
            detected_objects.append(label)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, detected_objects

# Main Streamlit app
def main():
    # Menambahkan CSS untuk kotak pada sidebar
    st.markdown("""
    <style>
    .sidebar-title {
        background-color: #70ff8f;
        padding: 6px;
        border-radius: 6px;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

    # Tampilkan animasi di sidebar
    display_animation("asta-liebe.gif")  # Path ke file GIF

    # Menampilkan judul dalam kotak
    st.sidebar.markdown('<div class="sidebar-title">MENU</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 style="color: white;">HASIL DETEKSI</h1>',
        unsafe_allow_html=True
    )

    # Atur background
    set_background("Blackbull.png", size="100% 100%")  # Path ke gambar latar belakang

    model_path = "yolo11n.pt"  # Path to your YOLO model
    model = load_model(model_path)

    # Gunakan radio button sebagai kontrol
    detection_status = st.sidebar.radio(
        "",
        options=["🔴 Lom Di Idop Cuy", "🟢 La Di Idop Cuy"],
        index=0,
        key="detection_control"
    )
    
    # Jalankan deteksi hanya jika radio button diatur ke "Aktif"
    if detection_status == "🟢 La Di Idop Cuy":
        cap = cv2.VideoCapture(0)
        st_frame = st.empty()  # Placeholder untuk tampilan video
        st_detection_info = st.empty()  # Placeholder untuk informasi deteksi

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Gagal menangkap gambar dari kamera.")
                break

            # Jalankan deteksi dengan YOLO
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert ke RGB
            results = model.predict(frame, imgsz=460)  # Lakukan prediksi
            
            # Tampilkan hasil dan daftar objek yang terdeteksi
            frame, detected_objects = display_results(frame, results[0])
            
            # Tampilkan video
            st_frame.image(frame, channels="RGB", use_column_width=True)
            
            # Tampilkan informasi objek yang terdeteksi
            if detected_objects:
                object_counts = Counter(detected_objects)
                detection_info = "\n".join([f"{obj}: {count}" for obj, count in object_counts.items()])
            else:
                detection_info = "Tidak ada objek terdeteksi."

            st_detection_info.text(detection_info)  # Perbarui teks informasi deteksi

        cap.release()

if __name__ == "__main__":
    main()