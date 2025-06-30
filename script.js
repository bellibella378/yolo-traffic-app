const imageUpload = document.getElementById('imageUpload');
const detectButton = document.getElementById('detectButton');
const loadingMessage = document.getElementById('loadingMessage');
const inputImage = document.getElementById('inputImage');
const detectionCanvas = document.getElementById('detectionCanvas');
const ctx = detectionCanvas.getContext('2d');
const detectionResults = document.getElementById('detectionResults');

let model; // Variabel global untuk menyimpan model YOLO
const MODEL_PATH = './model/model.json'; // Pastikan path ini benar!

// Ganti dengan label kelas rambu lalu lintas Anda sesuai urutan indeks model Anda
const CLASS_LABELS = [
    'stop_sign', 'yield_sign', 'speed_limit_30', 'speed_limit_50',
    'no_entry', 'pedestrian_crossing', 'traffic_light', 'parking_sign',
    // Tambahkan kelas lain yang didukung model Anda
];

// Sesuaikan ukuran input yang diharapkan oleh model YOLO Anda
// Contoh umum: 320, 416, 608
const INPUT_SIZE = 416;

// Ambang batas kepercayaan (confidence threshold) untuk menampilkan deteksi
const DETECTION_THRESHOLD = 0.5;
// Ambang batas Intersection over Union (IoU) untuk Non-Maximum Suppression (NMS)
const IOU_THRESHOLD = 0.4;
// Jumlah maksimum deteksi yang akan disimpan setelah NMS
const MAX_DETECTIONS = 100;

// --- Fungsi untuk memuat model TensorFlow.js ---
async function loadModel() {
    loadingMessage.style.display = 'block';
    loadingMessage.innerText = 'Memuat model... Harap tunggu.';
    try {
        console.log('Memuat model dari:', MODEL_PATH);
        // tf.loadGraphModel digunakan untuk model yang dikonversi dari Keras/Darknet
        model = await tf.loadGraphModel(MODEL_PATH);
        console.log('Model berhasil dimuat!');
        loadingMessage.innerText = 'Model berhasil dimuat. Pilih gambar untuk deteksi.';
        detectButton.disabled = false; // Aktifkan tombol deteksi setelah model dimuat
    } catch (error) {
        console.error('Gagal memuat model:', error);
        loadingMessage.innerText = `Gagal memuat model: ${error.message}. Pastikan file model ada di '${MODEL_PATH}' dan benar. Periksa konsol untuk detail.`;
    } finally {
        // Sembunyikan pesan loading setelah beberapa saat
        setTimeout(() => {
            loadingMessage.style.display = 'none';
        }, 3000);
    }
}

// Panggil fungsi loadModel saat halaman dimuat
window.onload = loadModel;

// --- Event listener untuk input gambar ---
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files?.[0]; // Menggunakan optional chaining untuk keamanan
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            inputImage.src = e.target.result;
            inputImage.onload = () => {
                // Atur ukuran kanvas sesuai gambar input
                detectionCanvas.width = inputImage.width;
                detectionCanvas.height = inputImage.height;
                // Bersihkan kanvas dan gambar ulang gambar input
                ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
                ctx.drawImage(inputImage, 0, 0, detectionCanvas.width, detectionCanvas.height);
                // Reset area hasil deteksi
                detectionResults.innerHTML = '<p>Tidak ada deteksi.</p>';
                // Aktifkan tombol deteksi hanya jika model sudah dimuat
                detectButton.disabled = !model;
            };
        };
        reader.readAsDataURL(file);
    } else {
        // Jika tidak ada file yang dipilih
        inputImage.src = '';
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        detectionResults.innerHTML = '<p>Tidak ada deteksi.</p>';
        detectButton.disabled = true;
    }
});

// --- Event listener untuk tombol deteksi ---
detectButton.addEventListener('click', async () => {
    if (!model) {
        alert('Model belum dimuat. Mohon tunggu atau periksa konsol untuk kesalahan.');
        return;
    }
    if (!inputImage.src) {
        alert('Mohon pilih gambar terlebih dahulu.');
        return;
    }

    detectButton.disabled = true; // Nonaktifkan tombol selama proses
    loadingMessage.innerText = 'Menganalisis gambar...';
    loadingMessage.style.display = 'block';
    detectionResults.innerHTML = ''; // Bersihkan hasil sebelumnya

    try {
        // 1. Preprocess gambar input untuk model YOLO
        const imgTensor = tf.browser.fromPixels(inputImage);
        // Ubah ukuran, konversi ke float, normalisasi (0-1), dan tambahkan dimensi batch
        const resizedNormalizedImg = tf.image.resizeBilinear(imgTensor, [INPUT_SIZE, INPUT_SIZE])
                                          .toFloat()
                                          .div(255.0)
                                          .expandDims(0);

        // 2. Lakukan inferensi model
        console.log('Melakukan inferensi...');
        // Pastikan Anda tahu struktur output 'predictions' dari model Anda
        const predictions = await model.executeAsync(resizedNormalizedImg);
        console.log('Inferensi selesai. Output model:', predictions);

        // 3. Proses output YOLO untuk mendapatkan bounding box, skor, dan kelas
        const detections = await processYoloOutput(predictions, inputImage.width, inputImage.height);

        // 4. Gambar bounding box dan tampilkan hasil
        // Bersihkan kanvas dan gambar ulang gambar asli
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        ctx.drawImage(inputImage, 0, 0, detectionCanvas.width, detectionCanvas.height);

        if (detections.length > 0) {
            detections.forEach((detection, index) => {
                const [x1, y1, x2, y2] = detection.bbox; // Koordinat dalam piksel gambar asli
                const score = detection.score;
                const classId = detection.classId;
                const className = CLASS_LABELS?.[classId] || `Unknown Class ${classId}`;

                // Gambar bounding box
                ctx.beginPath();
                ctx.rect(x1, y1, x2 - x1, y2 - y1);
                ctx.lineWidth = 3;
                ctx.strokeStyle = '#FF0000'; // Warna merah
                ctx.stroke();

                // Gambar label teks (kelas dan probabilitas)
                ctx.fillStyle = '#FF0000';
                ctx.font = '16px Arial';
                const labelText = `${className} (${(score * 100).toFixed(2)}%)`;
                // Posisikan teks di atas kotak atau di dalamnya jika tidak ada ruang di atas
                ctx.fillText(labelText, x1 + 5, y1 > 20 ? y1 - 10 : y1 + 20);

                // Tampilkan hasil di kotak teks
                detectionResults.innerHTML += `
                    <div class="detection-item">
                        <p><span>Objek ${index + 1}:</span> ${className}</p>
                        <p><span>Confidence:</span> ${(score * 100).toFixed(2)}%</p>
                        <p><span>Bounding Box (x1, y1, x2, y2):</span> [${x1.toFixed(0)}, ${y1.toFixed(0)}, ${x2.toFixed(0)}, ${y2.toFixed(0)}]</p>
                    </div>
                `;
            });
        } else {
            detectionResults.innerHTML = '<p>Tidak ada rambu lalu lintas terdeteksi.</p>';
        }

        // Bersihkan tensor dari memori untuk menghindari kebocoran memori
        imgTensor.dispose();
        resizedNormalizedImg.dispose();
        if (Array.isArray(predictions)) {
            predictions.forEach(tensor => tensor.dispose());
        } else if (predictions) { // Jika hanya ada satu tensor output
            predictions.dispose();
        }

    } catch (error) {
        console.error('Terjadi kesalahan selama deteksi:', error);
        detectionResults.innerHTML = `<p style="color: red;">Error saat deteksi: ${error.message}. Periksa konsol untuk detail.</p>`;
    } finally {
        loadingMessage.style.display = 'none';
        detectButton.disabled = false; // Aktifkan kembali tombol
    }
});

// --- FUNGSI PENTING: Memproses Output YOLO ---
// Fungsi ini adalah inti dari deteksi YOLO. Anda HARUS menyesuaikannya
// berdasarkan bagaimana model YOLO Anda mengeluarkan prediksinya.
// Kode ini adalah placeholder SANGAT SEDERHANA yang mengasumsikan format tertentu.
async function processYoloOutput(predictions, originalImgWidth, originalImgHeight) {
    let boxes = [];
    let scores = [];
    let classIds = [];

    // --- Contoh Implementasi PLACEHOLDER ---
    // Asumsi: model mengeluarkan satu tensor prediksi dengan bentuk:
    // [1, num_predictions, (x, y, width, height, confidence, class_prob_1, class_prob_2, ...)]
    // Atau jika model Anda menggunakan YOLOv3/v4 dengan beberapa output feature map,
    // maka 'predictions' akan menjadi array dari beberapa tensor.
    // Anda perlu mengiterasi setiap tensor output dan memprosesnya.

    // Contoh ini mengasumsikan output sederhana dari sebuah model TFJS yang telah diproses
    // untuk menghasilkan daftar bounding box, skor, dan kelas.
    // Jika Anda mengonversi model Darknet/Keras/PyTorch YOLO, Anda biasanya harus
    // melakukan decoding anchor box, sigmoid activation, dan non-maximum suppression (NMS) secara manual di sini.

    // placeholder: ambil tensor pertama dari array prediksi
    const outputTensor = Array.isArray(predictions) ? predictions[0] : predictions;
    if (!outputTensor || outputTensor.shape.length < 2) {
        console.error("Format output tensor tidak sesuai harapan. Periksa struktur model Anda.");
        return [];
    }

    // Jika output tensor memiliki format [num_detections, (x1, y1, x2, y2, score, class_id)]
    // (ini sering terjadi setelah post-processing di Python sebelum konversi)
    const rawDetections = await outputTensor.array(); // Dapatkan data sebagai array JS

    // Iterasi melalui setiap deteksi mentah
    for (const detection of rawDetections) {
        // Asumsi format 'detection': [x1_norm, y1_norm, x2_norm, y2_norm, score, class_id]
        // (koordinat dinormalisasi 0-1)
        const [x1_norm, y1_norm, x2_norm, y2_norm, score, class_id] = detection;

        if (score >= DETECTION_THRESHOLD) {
            // Konversi koordinat normalisasi ke koordinat piksel gambar asli
            const x1_px = x1_norm * originalImgWidth;
            const y1_px = y1_norm * originalImgHeight;
            const x2_px = x2_norm * originalImgWidth;
            const y2_px = y2_norm * originalImgHeight;

            // Simpan bounding box dalam format [y_min, x_min, y_max, x_max] untuk tf.image.nonMaxSuppressionAsync
            boxes.push([y1_px, x1_px, y2_px, x2_px]);
            scores.push(score);
            classIds.push(Math.round(class_id)); // Pastikan class_id adalah integer
        }
    }

    // Lakukan Non-Maximum Suppression (NMS)
    const nmsResult = await tf.image.nonMaxSuppressionAsync(
        tf.tensor2d(boxes),
        tf.tensor1d(scores),
        MAX_DETECTIONS,
        IOU_THRESHOLD,
        DETECTION_THRESHOLD // Gunakan threshold yang sama untuk filtering awal dan NMS
    );

    const pickedIndices = await nmsResult.data();
    const finalDetections = [];

    // Filter deteksi berdasarkan hasil NMS
    for (let i = 0; i < pickedIndices.length; i++) {
        const index = pickedIndices[i];
        const [y1, x1, y2, x2] = boxes[index]; // Koordinat sudah dalam piksel
        const score = scores[index];
        const classId = classIds[index];

        finalDetections.push({
            bbox: [x1, y1, x2, y2],
            score: score,
            classId: classId
        });
    }

    // Penting: Bersihkan tensor yang tidak lagi digunakan untuk menghemat memori
    outputTensor.dispose();
    nmsResult.dispose();

    // --- AKHIR Implementasi PLACEHOLDER ---

    return finalDetections;
}
