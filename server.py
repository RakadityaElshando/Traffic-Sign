from flask import Flask, render_template, request, jsonify, send_from_directory
from ultralytics import YOLO
import torch
import torchvision
import cv2
import os
import uuid
import base64
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ========== Load Models ==========
yolo_model = YOLO("best.pt")

def load_fastrcnn_model(path="model_fastrcnn.pth", num_classes=29):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

fast_model = load_fastrcnn_model()
CONFIDENCE_THRESHOLD = 0.5

# ========== Label Descriptions ==========
label_descriptions = {
    "dilarangParkir": "Rambu yang menunjukkan larangan untuk parkir di area tersebut.",
    "dilarangMasuk": "Rambu yang menunjukkan bahwa kendaraan dilarang masuk.",
    "hatiHati": "Rambu peringatan agar pengendara lebih waspada.",
    "jalanBergelombang": "Menandakan bahwa jalan di depan bergelombang.",
    "jalanLicin": "Rambu peringatan bahwa jalan licin.",
    "jalanMenyempit": "Jalan menyempit di depan.",
    "mobilDilarangMasuk": "Larangan masuk untuk kendaraan roda empat.",
    "parkir": "Area parkir yang diizinkan.",
    "pejalanKaki": "Menandakan lintasan atau keberadaan pejalan kaki.",
    "pemeliharaanJalan": "Terdapat perbaikan jalan di depan.",
    "perempatan": "Terdapat simpang empat di depan.",
    "rambuTikunganKanan": "Jalan di depan berbelok ke kanan.",
    "rambuTikunganKiri": "Jalan di depan berbelok ke kiri.",
    "rambuTikunganTajamKanan": "Tikungan tajam ke kanan di depan.",
    "rambuTikunganTajamKiri": "Tikungan tajam ke kiri di depan.",
    "rambuTengkorak": "Peringatan area berbahaya atau rawan kecelakaan.",
    "sepeda": "Rambu yang menunjukkan lintasan khusus sepeda.",
    "stop": "Rambu STOP: wajib berhenti sebelum melanjutkan.",
    "trafficLight": "Lampu lalu lintas di depan.",
    "zebraCross": "Lintasan penyeberangan pejalan kaki.",
    "anakSekolah": "Hati-hati, ada anak sekolah di sekitar.",
    "anjing": "Peringatan kemungkinan ada hewan melintas.",
    "awasKereta": "Lintasan kereta api di depan.",
    "awan": "Rambu cuaca: kemungkinan kabut/awan tebal.",
    "jalanTurun": "Tanjakan menurun tajam di depan.",
    "jalanTanjak": "Tanjakan curam di depan.",
    "rambuBerhenti": "Kendaraan wajib berhenti sementara.",
    "rambuPutarBalik": "Lokasi untuk putar balik kendaraan.",
    "persimpanganEmpat": "Simpang empat, hati-hati dan beri prioritas.",
    "persimpanganTigaKiri": "Simpang tiga, hati-hati dan beri prioritas."
}

# Index to label mapping for Faster R-CNN (should match training order)
idx_to_label = list(label_descriptions.keys())

def get_description(label):
    return label_descriptions.get(label, f"Tidak ada deskripsi untuk '{label}'")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    model_type = request.args.get("model", "yolo")

    if not file:
        return "No file uploaded", 400

    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    label = "Tidak terdeteksi"
    annotated_frame = None

    if model_type == "yolo":
        result = yolo_model.predict(source=filepath, save=False, imgsz=640)[0]
        annotated_frame = result.plot()
        if len(result.boxes.cls) > 0:
            label = yolo_model.names[int(result.boxes.cls[0])]

    elif model_type == "fast_rcnn":
        image = Image.open(filepath).convert("RGB")
        image_np = np.array(image)
        img_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)

        with torch.no_grad():
            output = fast_model(img_tensor)[0]

        annotated_frame = image_np.copy()
        for box, score, lbl in zip(output["boxes"], output["scores"], output["labels"]):
            if score >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.int().tolist()
                label_idx = int(lbl)
                if 0 <= label_idx < len(idx_to_label):
                    label = idx_to_label[label_idx]
                else:
                    label = f"Label {label_idx}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                break
    else:
        return "Model tidak dikenali", 400

    output_path = os.path.join(UPLOAD_FOLDER, f"result_{filename}")
    cv2.imwrite(output_path, annotated_frame)

    return jsonify({
        "image_url": f"/{output_path}",
        "description": get_description(label)
    })

@app.route("/detect-frame", methods=["POST"])
def detect_frame():
    data = request.get_json()
    model_type = request.args.get("model", "yolo")
    image_data = data["image"].split(",")[1]
    image_bytes = BytesIO(base64.b64decode(image_data))
    image = Image.open(image_bytes).convert("RGB")
    image_np = np.array(image)

    label = "Tidak terdeteksi"
    annotated_frame = None

    if model_type == "yolo":
        result = yolo_model.predict(source=image_np, save=False, imgsz=640)[0]
        annotated_frame = result.plot()
        if len(result.boxes.cls) > 0:
            label = yolo_model.names[int(result.boxes.cls[0])]

    elif model_type == "fast_rcnn":
        img_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)
        with torch.no_grad():
            output = fast_model(img_tensor)[0]

        annotated_frame = image_np.copy()
        for box, score, lbl in zip(output["boxes"], output["scores"], output["labels"]):
            if score >= CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = box.int().tolist()
                label_idx = int(lbl)
                if 0 <= label_idx < len(idx_to_label):
                    label = idx_to_label[label_idx]
                else:
                    label = f"Label {label_idx}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                break
    else:
        return "Model tidak dikenali", 400

    output_filename = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    cv2.imwrite(output_path, annotated_frame)

    return jsonify({
        "image_url": f"/{output_path}",
        "description": get_description(label)
    })

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory('.', path)

if __name__ == "__main__":
    app.run(debug=True)