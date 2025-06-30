const imageUpload = document.getElementById('imageUpload');
const detectButton = document.getElementById('detectButton');
const loadingMessage = document.getElementById('loadingMessage');
const inputImage = document.getElementById('inputImage');
const detectionCanvas = document.getElementById('detectionCanvas');
const ctx = detectionCanvas.getContext('2d');
const detectionResults = document.getElementById('detectionResults');

let model; // Variabel global untuk menyimpan model YOLO
const MODEL_PATH = './model/model.json'; // Pastikan path ini benar!
const CLASS_LABELS = ['stop_sign', 'yield_sign', 'speed_limit_30', 'speed_limit_50', 'no_entry', 'pedestrian_crossing', 'traffic_light']; // Ganti dengan label kelas rambu lalu lintas Anda

// Fungsi untuk memuat model TensorFlow.js
async function loadModel() {
    loadingMessage.style.display = 'block';
    try {
        // Asumsi model YOLO Anda telah dikonversi ke format tfjs_graph_model
        model = await tf.loadGraphModel(MODEL_PATH);
        console.log('Model loaded successfully!');
        loadingMessage.innerText = 'Model berhasil dimuat. Siap untuk deteksi.';
        detectButton.disabled = false; // Aktifkan tombol deteksi setelah model dimuat
    } catch (error) {
        console.error('Error loading model:', error);
        loadingMessage.innerText = `Gagal memuat model: ${error.message}. Pastikan file model ada di '${MODEL_PATH}' dan benar.`;
    } finally {
        // Sembunyikan pesan loading setelah beberapa saat
        setTimeout(() => {
            loadingMessage.style.display = 'none';
        }, 3000);
    }
}

// Panggil fungsi loadModel saat halaman dimuat
window.onload = loadModel;

// Event listener untuk input gambar
imageUpload.addEventListener('change', (event) => {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            inputImage.src = e.target.result;
            inputImage.onload = () => {
                // Atur ukuran kanvas sesuai gambar
                detectionCanvas.width = inputImage.width;
                detectionCanvas.height = inputImage.height;
                ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height); // Bersihkan kanvas
                ctx.drawImage(inputImage, 0, 0, detectionCanvas.width, detectionCanvas.height); // Gambar ulang gambar input
                detectionResults.innerHTML = '<p>Tidak ada deteksi.</p>'; // Reset hasil
                detectButton.disabled = !model; // Hanya aktifkan jika model sudah dimuat
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
    detectionResults.innerHTML = ''; // Bersihkan hasil sebelumnya

    try {
        const tensor = tf.browser.fromPixels(inputImage).resizeBilinear([416, 416]).expandDims(0).toFloat().div(255.0); // Sesuaikan ukuran input YOLO Anda (misal: 416x416)

        // Melakukan inferensi
        const predictions = await model.executeAsync(tensor);

        // TODO: Implement YOLO inference logic here
        // Bagian ini sangat bergantung pada arsitektur model YOLO Anda dan bagaimana outputnya.
        // Umumnya, output YOLO akan berupa tensor yang berisi:
        // - Bounding box (x, y, width, height) atau (x_min, y_min, x_max, y_max)
        // - Confidence score untuk objek
        // - Class probabilities

        // Contoh dummy output (Anda harus menggantinya dengan logika nyata)
        // Asumsi output `predictions` adalah array tensor, dan tensor pertama adalah boxData, kedua adalah classData
        // const [boxData, classData] = await Promise.all(predictions.map(t => t.data()));

        // --- Contoh pseudo-code untuk memproses output YOLO ---
        // Biasanya melibatkan:
        // 1. Memproses output tensor ke dalam format yang lebih mudah (misal array JS)
        // 2. Menerapkan NMS (Non-Maximum Suppression) untuk menghilangkan bounding box yang tumpang tindih
        // 3. Filtering berdasarkan confidence threshold

        // Untuk demonstrasi, kita akan membuat beberapa bounding box dummy:
        const detectedObjects = [];
        // Ganti ini dengan logika pemrosesan output model YOLO Anda yang sebenarnya
        // Misalnya, jika model Anda mengeluarkan [batch_size, num_boxes, 4+1+num_classes]
        // Anda perlu memparse tensor tersebut.
        // Contoh sederhana jika output adalah [x, y, width, height, confidence, class_id]
        // Misalnya, asumsi output Anda sudah dalam bentuk array flat [x1, y1, w1, h1, conf1, class1, x2, y2, w2, h2, conf2, class2, ...]
        // Atau jika model Anda mengembalikan tensor yang sudah diproses (misal dari tf.js-models seperti coco-ssd, tapi ini YOLO)

        // *** PENTING: Anda harus mengganti bagian ini dengan logika parsing output YOLO Anda ***
        // Ini adalah tempat Anda akan mengimplementasikan post-processing YOLO,
        // seperti decoding box, NMS, dll.
        // Jika Anda menggunakan model yang telah dikonversi secara otomatis, mungkin ada cara untuk mendapatkan kotak, skor, dan kelas secara langsung.
        
        // Contoh placeholder output. Anda akan mendapatkan ini dari hasil prediksi model Anda
        // Format: [x_min, y_min, x_max, y_max, score, class_id] (normalisasi 0-1)
        const dummyDetections = [
            // Contoh 1: Rambu Stop
            { x: 0.1, y: 0.2, width: 0.2, height: 0.3, score: 0.95, classId: 0 },
            // Contoh 2: Rambu Batas Kecepatan
            { x: 0.5, y: 0.4, width: 0.15, height: 0.2, score: 0.88, classId: 2 },
        ];

        // Konversi koordinat normalisasi ke koordinat piksel
        dummyDetections.forEach(d => {
            const x_min = d.x * detectionCanvas.width;
            const y_min = d.y * detectionCanvas.height;
            const width = d.width * detectionCanvas.width;
            const height = d.height * detectionCanvas.height;
            const x_max = x_min + width;
            const y_max = y_min + height;

            detectedObjects.push({
                box: [x_min, y_min, x_max, y_max],
                score: d.score,
                class: CLASS_LABELS[d.classId] || `unknown_class_${d.classId}`,
                classId: d.classId
            });
        });

        // Akhir dari bagian placeholder ***

        // Bersihkan kanvas dan gambar ulang gambar asli
        ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
        ctx.drawImage(inputImage, 0, 0, detectionCanvas.width, detectionCanvas.height);

        detectionResults.innerHTML = '';
        if (detectedObjects.length > 0) {
            detectedObjects.forEach((obj, index) => {
                const [x_min, y_min, x_max, y_max] = obj.box;

                // Gambar bounding box
                ctx.beginPath();
                ctx.rect(x_min, y_min, x_max - x_min, y_max - y_min);
                ctx.lineWidth = 3;
                ctx.strokeStyle = '#FF0000'; // Warna merah
                ctx.stroke();

                // Gambar label teks (kelas dan probabilitas)
                ctx.fillStyle = '#FF0000';
                ctx.font = '16px Arial';
                const text = `${obj.class} (${(obj.score * 100).toFixed(2)}%)`;
                ctx.fillText(text, x_min + 5, y_min > 20 ? y_min - 10 : y_min + 20); // Posisikan teks

                // Tampilkan hasil di kotak teks
                detectionResults.innerHTML += `
                    <div class="detection-item">
                        <p><span>Objek ${index + 1}:</span> ${obj.class}</p>
                        <p><span>Confidence:</span> ${(obj.score * 100).toFixed(2)}%</p>
                        <p><span>Bounding Box (x_min, y_min, x_max, y_max):</span> [${x_min.toFixed(0)}, ${y_min.toFixed(0)}, ${x_max.toFixed(0)}, ${y_max.toFixed(0)}]</p>
                    </div>
                `;
            });
        } else {
            detectionResults.innerHTML = '<p>Tidak ada rambu lalu lintas terdeteksi.</p>';
        }

    } catch (error) {
        console.error('Error during detection:', error);
        detectionResults.innerHTML = `<p style="color: red;">Error saat deteksi: ${error.message}</p>`;
    } finally {
        loadingMessage.style.display = 'none';
        detectButton.disabled = false;
    }
});
