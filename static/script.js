let mode = "upload";

function setMode(selected) {
  mode = selected;
  if (mode === "camera") {
    document.getElementById("upload-container").style.display = "none";
    document.getElementById("camera-container").style.display = "block";
    startCamera();
  } else {
    document.getElementById("camera-container").style.display = "none";
    document.getElementById("upload-container").style.display = "block";
    stopCamera();
  }
}

function startCamera() {
  const video = document.getElementById("video");
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((err) => console.error("Kamera gagal:", err));
}

function stopCamera() {
  const video = document.getElementById("video");
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((track) => track.stop());
    video.srcObject = null;
  }
}

async function captureAndSend() {
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

  const image_data = canvas.toDataURL("image/jpeg");
  const response = await fetch("/detect-frame", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: image_data }),
  });

  const result = await response.json();
  if (result.image_url) {
    document.getElementById("result-image").src = result.image_url;
  }
}

async function uploadFile() {
  const input = document.getElementById("uploadInput");
  const formData = new FormData();
  formData.append("file", input.files[0]);

  const response = await fetch("/upload", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  if (result.image_url) {
    document.getElementById("result-image").src = result.image_url;
  }
}
