const preview = document.getElementById("preview");
const upload = document.getElementById("upload");

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
  alert("Simulasi deteksi YOLOv8 berhasil dijalankan!");
}