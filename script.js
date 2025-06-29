const preview = document.getElementById("preview");
const upload = document.getElementById("upload");
const hasil = document.getElementById("hasil");

upload.addEventListener("change", function () {
  const file = this.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function (e) {
      preview.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }
});

function deteksi() {
  if (preview.src === "") {
    alert("Silakan unggah gambar terlebih dahulu.");
    return;
  }

  // Simulasi deteksi bounding box
  const box = document.createElement("div");
  box.style.position = "absolute";
  box.style.border = "2px solid red";
  box.style.left = "80px";
  box.style.top = "60px";
  box.style.width = "150px";
  box.style.height = "150px";

  // Tambahkan bounding box ke preview
  const container = document.getElementById("preview-container");
  container.style.position = "relative";
  container.appendChild(box);

  // Update hasil deteksi
  hasil.innerHTML = `
    <h2>Hasil Deteksi</h2>
    <p>Model: YOLOv8m</p>
    <ul>
      <li>Rambu: Stop</li>
      <li>Confidence: 0.91</li>
      <li>Koordinat: (x=80, y=60, w=150, h=150)</li>
    </ul>
  `;
}
