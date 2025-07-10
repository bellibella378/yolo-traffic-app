import cv2
import numpy as np
import onnxruntime as rt

# --- KONFIGURASI ---
# Ganti dengan path ke file model ONNX Anda
MODEL_PATH = 'best.onnx' 
# Ganti dengan path ke file gambar yang ingin diuji
IMAGE_PATH = 'image.jpg'
# Ganti dengan nama file untuk menyimpan hasil
OUTPUT_IMAGE_PATH = 'hasil_deteksi_diperbaiki.jpg'

# --- PENTING ---
# Sesuaikan daftar nama kelas ini AGAR SAMA PERSIS dengan urutan pada saat training model Anda.
# Jumlah item di sini HARUS SAMA dengan jumlah kelas keluaran model.
CLASS_NAMES = [
    'Stop', 'Batas Kecepatan 30', 'Batas Kecepatan 60', 'Dilarang Parkir', 
    'Dilarang Berhenti', 'Hati-hati', 'Belok Kiri', 'Belok Kanan', 'Lampu Merah'
]
NUM_CLASSES = len(CLASS_NAMES)

# Ambang batas kepercayaan (confidence threshold) dan NMS
CONF_THRESHOLD = 0.5 
NMS_THRESHOLD = 0.45 # Ambang batas untuk Non-Maximum Suppression

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    """
    Fungsi untuk menggambar bounding box dan label pada gambar.
    """
    # Beri label dengan nama kelas dan skor kepercayaan
    label = f'{CLASS_NAMES[class_id]} ({confidence:.2f})'
    # Warna acak untuk setiap kelas agar mudah dibedakan
    # PERBAIKAN: Konversi setiap nilai warna ke integer standar Python untuk menghindari error tipe data
    color = ( int((class_id * 50) % 255), int((class_id * 100) % 255), int((class_id * 25) % 255) )
    
    # Gambar persegi panjang (bounding box)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    # Tulis teks label
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def main():
    """
    Fungsi utama untuk memuat model, melakukan inferensi, dan menampilkan hasil.
    """
    try:
        # Muat sesi inferensi ONNX
        session = rt.InferenceSession(MODEL_PATH)
        print("Model ONNX berhasil dimuat.")
    except Exception as e:
        print(f"Error saat memuat model ONNX: {e}")
        return

    # Dapatkan properti input dari model
    model_inputs = session.get_inputs()
    input_name = model_inputs[0].name
    input_shape = model_inputs[0].shape # Contoh: [1, 3, 640, 640]
    input_height = input_shape[2]
    input_width = input_shape[3]
    
    # Baca gambar dari file
    original_image = cv2.imread(IMAGE_PATH)
    if original_image is None:
        print(f"Error: Gagal membaca gambar dari path: {IMAGE_PATH}")
        return

    original_height, original_width, _ = original_image.shape
    
    # --- PRE-PROCESSING GAMBAR ---
    image_resized = cv2.resize(original_image, (input_width, input_height))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    input_tensor = np.expand_dims(image_transposed, axis=0).astype(np.float32)

    # --- INFERENSI / DETEKSI ---
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {input_name: input_tensor})
    
    # --- POST-PROCESSING HASIL (Kompatibel dengan YOLO) ---
    # Asumsi output model adalah [batch, 4 + num_classes, num_detections]
    # Contoh: [1, 13, 8400] untuk 9 kelas
    output_data = outputs[0][0]
    detections = output_data.T  # Transpose menjadi [num_detections, 4 + num_classes]

    print(f"Deteksi selesai. Memproses {detections.shape[0]} potensi deteksi...")
    
    boxes = []
    scores = []
    class_ids = []

    # Hitung faktor skala
    x_scale = original_width / input_width
    y_scale = original_height / input_height

    for detection in detections:
        # Kolom 4 dan seterusnya berisi skor untuk setiap kelas
        class_scores = detection[4:]
        
        # Cari kelas dengan skor tertinggi dari deteksi ini
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id] # Skor kepercayaan adalah skor dari kelas yang terdeteksi

        # Abaikan deteksi jika di bawah ambang batas
        if confidence < CONF_THRESHOLD:
            continue
        
        # Ekstrak koordinat bounding box [center_x, center_y, width, height]
        cx, cy, w, h = detection[0:4]
        
        # Konversi ke format [x_top_left, y_top_left, width, height] dan sesuaikan skala
        x1 = int((cx - w / 2) * x_scale)
        y1 = int((cy - h / 2) * y_scale)
        width = int(w * x_scale)
        height = int(h * y_scale)
        
        boxes.append([x1, y1, width, height])
        scores.append(float(confidence))
        class_ids.append(class_id)

    # Terapkan Non-Maximum Suppression untuk menghilangkan bounding box yang tumpang tindih
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        print(f"Ditemukan {len(indices)} objek setelah NMS.")
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            class_id = class_ids[i]
            confidence = scores[i]
            
            # --- Safety Check ---
            # Periksa apakah class_id valid sebelum mengakses CLASS_NAMES
            if class_id < NUM_CLASSES:
                draw_bounding_box(original_image, class_id, confidence, x, y, x + w, y + h)
            else:
                print(f"Peringatan: Terdeteksi class_id={class_id} yang berada di luar jangkauan CLASS_NAMES (jumlah: {NUM_CLASSES}). Harap periksa daftar kelas Anda.")
    else:
        print("Tidak ada objek yang terdeteksi di atas ambang batas kepercayaan.")

    # --- TAMPILKAN DAN SIMPAN HASIL ---
    print(f"Hasil deteksi disimpan ke: {OUTPUT_IMAGE_PATH}")
    cv2.imwrite(OUTPUT_IMAGE_PATH, original_image)
    
    cv2.imshow('Deteksi Rambu Lalu Lintas (Diperbaiki)', original_image)
    print("Tekan tombol apa saja untuk keluar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
