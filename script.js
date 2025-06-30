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
const CLASS_LABELS = ['stop_sign', 'yield_sign', 'speed_limit_30', 'speed_limit_50', 'no_entry', 'pedestrian_crossing', 'traffic_light'];
// Sesuaikan ukuran input yang diharapkan oleh model YOLO Anda
const INPUT_SIZE = 416; // Contoh: 416x416

// Fungsi untuk memuat model TensorFlow.js
async function loadModel() {
    loadingMessage.style.display = 'block';
    loadingMessage.innerText = 'Memuat model... Harap tunggu.';
    try {
        console.log('Memuat model dari:', MODEL_PATH);
        model = await tf.loadGraphModel(MODEL_PATH);
        console.log('Model berhasil dimuat!');
        loadingMessage.innerText = 'Model berhasil dimuat. Siap untuk deteksi.';
        detectButton.disabled = false; // Aktifkan tombol deteksi setelah model dimuat
    } catch (error) {
        console.error('Gagal memuat model:', error);
        loadingMessage.innerText = `Gagal memuat model: ${error.message}. Pastikan file model ada di '${MODEL_PATH}' dan benar.`;
    } finally {
        setTimeout(() => {
            loadingMessage.style.display = 'none';
        }, 3000);
    }
}

// Panggil fungsi loadModel saat halaman dimuat
window.onload = loadModel;

// Event listener untuk input gambar
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            inputImage.src = e.target.result;
            inputImage.onload = () => {
                detectionCanvas.width = inputImage.width;
                detectionCanvas.height = inputImage.height;
                ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
                ctx.drawImage(inputImage, 0, 0, detectionCanvas.width, detectionCanvas.height);
                detectionResults.innerHTML = '<p>Tidak ada deteksi.</p>';
                detectButton.disabled = !model;
            };
        };
        reader.readAsDataURL(file);
    } else {
        inputImage.src = '';
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        detectionResults.innerHTML = '<p>Tidak ada deteksi.</p>';
        detectButton.disabled = true;
    }
});

// Event listener untuk tombol deteksi
detectButton.addEventListener('click', async () => {
    if (!model) {
        alert('Model belum dimuat. Mohon tunggu atau periksa konsol untuk error.');
        return;
    }
    if (!inputImage.src) {
        alert('Mohon pilih gambar terlebih dahulu.');
        return;
    }

    detectButton.disabled = true;
    loadingMessage.innerText = 'Menganalisis gambar...';
    loadingMessage.style.display = 'block';
    detectionResults.innerHTML = '';

    try {
        // 1. Preprocess gambar input
        const img = tf.browser.fromPixels(inputImage);
        const resizedImg = tf.image.resizeBilinear(img, [INPUT_SIZE, INPUT_SIZE]).toFloat();
        const normalizedImg = resizedImg.div(255.0).expandDims(0);

        // 2. Lakukan inferensi model
        console.log('Melakukan inferensi...');
        const predictions = await model.executeAsync(normalizedImg);
        console.log('Inferensi selesai.');

        // 3. Proses output YOLO untuk mendapatkan bounding box, skor, dan kelas
        const detections = await processYoloOutput(predictions, inputImage.width, inputImage.height);

        // 4. Gambar bounding box dan tampilkan hasil
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        ctx.drawImage(inputImage, 0, 0, detectionCanvas.width, detectionCanvas.height);
        detectionResults.innerHTML = '';

        if (detections.length > 0) {
            detections.forEach(detection => {
                const [x1, y1, x2, y2] = detection.bbox;
                const score = detection.score;
                const classId = detection.classId;
                const className = CLASS_LABELS?.[classId] || `Unknown (${classId})`;

                // Gambar bounding box
                ctx.beginPath();
                ctx.rect(x1, y1, x2 - x1, y2 - y1);
                ctx.lineWidth = 3;
                ctx.strokeStyle = '#FF0000';
                ctx.stroke();

                // Gambar label
                ctx.fillStyle = '#FF0000';
                ctx.font = '16px Arial';
                const labelText = `${className} (${(score * 100).toFixed(2)}%)`;
                ctx.fillText(labelText, x1 + 5, y1 > 20 ? y1 - 10 : y1 + 20);

                // Tampilkan detail di hasil
                detectionResults.innerHTML += `
                    <div class="detection-item">
                        <p><span>Objek:</span> ${className}</p>
                        <p><span>Confidence:</span> ${(score * 100).toFixed(2)}%</p>
                        <p><span>Bounding Box (x1, y1, x2, y2):</span> [${x1.toFixed(0)}, ${y1.toFixed(0)}, ${x2.toFixed(0)}, ${y2.toFixed(0)}]</p>
                    </div>
                `;
            });
        } else {
            detectionResults.innerHTML = '<p>Tidak ada rambu lalu lintas terdeteksi.</p>';
        }

        // Bersihkan tensor dari memori
        normalizedImg.dispose();
        if (Array.isArray(predictions)) {
            predictions.forEach(tensor => tensor.dispose());
        } else if (predictions) {
            predictions.dispose();
        }

    } catch (error) {
        console.error('Error during detection:', error);
        detectionResults.innerHTML = `<p style="color: red;">Error saat deteksi: ${error.message}</p>`;
    } finally {
        loadingMessage.style.display = 'none';
        detectButton.disabled = false;
    }
});

// *** FUNGSI INI SANGAT PENTING DAN HARUS DIIMPLEMENTASIKAN SESUAI OUTPUT MODEL YOLO ANDA ***
async function processYoloOutput(predictions, imgWidth, imgHeight) {
    // Struktur output 'predictions' akan bergantung pada model YOLO Anda.
    // Anda perlu memahami bagaimana model Anda mengeluarkan bounding box,
    // confidence score, dan class probabilities.

    // Contoh umum (TAPI INI MUNGKIN BERBEDA UNTUK MODEL ANDA):
    // Asumsikan 'predictions' adalah array tensor, dan salah satunya
    // memiliki bentuk [1, grid_size, grid_size, num_anchors * (5 + num_classes)]
    // di mana 5 adalah (tx, ty, tw, th, object_confidence).

    if (!Array.isArray(predictions) || predictions.length === 0) {
        console.warn('Tidak ada output prediksi yang diterima dari model.');
        return [];
    }

    // *** IMPLEMENTASI SPESIFIK ANDA DIMULAI DI SINI ***
    // Anda perlu mengurai output tensor, menerapkan thresholding,
    // melakukan non-maximum suppression (NMS), dan mengonversi
    // koordinat bounding box ke skala gambar asli.

    // --- Contoh SANGAT SEDERHANA (MUNGKIN TIDAK BERFUNGSI UNTUK MODEL ANDA) ---
    // Ini hanya ilustrasi dan perlu disesuaikan.
    const detectionThreshold = 0.5;
    const nmsThreshold = 0.4;
    const numClasses = CLASS_LABELS.length;

    let boxes = [];
    let scores = [];
    let classIds = [];

    for (const prediction of predictions) {
        const data = await prediction.array();
        // Iterasi melalui output grid dan anchor box (struktur spesifik model Anda)
        // Ekstrak bounding box, confidence, dan class probabilities
        // Lakukan filtering berdasarkan threshold
        // Simpan kotak, skor, dan ID kelas yang relevan
    }

    // Lakukan Non-Maximum Suppression (NMS) untuk menghilangkan kotak yang tumpang tindih
    const selectedIndices = await tf.image.nonMaxSuppressionAsync(
        boxes.map(b => [b.yMin, b.xMin, b.yMax, b.xMax]), // Format [yMin, xMin, yMax, xMax]
        scores,
        50, // max_detections
        nmsThreshold
    );

    const finalDetections = [];
    const indices = await selectedIndices.data();
    for (let i = 0; i < indices.length; ++i) {
        const index = indices?.[i];
        if (index !== undefined) {
            const bbox = boxes?.[index];
            const score = scores?.[index];
            const classId = classIds?.[index];
            if (bbox && score !== undefined && classId !== undefined) {
                // Konversikan kembali ke skala gambar asli jika perlu
                const x1 = bbox.xMin * imgWidth;
                const y1 = bbox.yMin * imgHeight;
                const x2 = bbox.xMax * imgWidth;
                const y2 = bbox.yMax * imgHeight;
                finalDetections.push({ bbox: [x1, y1, x2, y2], score, classId });
            }
        }
    }

    tf.dispose(selectedIndices);
    // *** AKHIR IMPLEMENTASI SPESIFIK ANDA ***

    return finalDetections;
}
