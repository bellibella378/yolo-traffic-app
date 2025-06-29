const upload = document.getElementById("upload");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const hasilList = document.getElementById("hasil-list");

let currentImage = null;

upload.addEventListener("change", function () {
  const file = this.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function (e) {
    const img = new Image();
    img.onload = function () {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      currentImage = img;
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
});

async function deteksi() {
  if (!upload.files[0]) {
    alert("Silakan unggah gambar terlebih dahulu.");
    return;
  }

  const formData = new FormData();
  formData.append("image", upload.files[0]);

  const response = await fetch("http://localhost:5000/deteksi", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  // Reset tampilan
  ctx.drawImage(currentImage, 0, 0);
  hasilList.innerHTML = "";

  data.forEach((deteksi) => {
    const { label, confidence, bbox } = deteksi;

    // Gambar bounding box
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

    // Gambar label
    ctx.fillStyle = "red";
    ctx.font = "16px Arial";
    ctx.fillText(`${label} (${(confidence * 100).toFixed(1)}%)`, bbox.x, bbox.y - 5);

    // Tambahkan ke daftar hasil
    const li = document.createElement("li");
    li.textContent = `Rambu: ${label}, Confidence: ${(confidence * 100).toFixed(1)}%, Lokasi: (${bbox.x}, ${bbox.y})`;
    hasilList.appendChild(li);
  });
}
