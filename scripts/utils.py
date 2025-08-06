# utils.py
import cv2
import os
from PIL import Image
from fpdf import FPDF
import numpy as np
import os 

REQUIRED_CLASSES = {}

CLASS_NAMES = {
    0: "person",
    1: "ear_plugs",
    2: "entry_exit_line",
    3: "helmet",
    4: "safety_goggles",
    5: "safety_hand_gloves",
    6: "safety_jacket",
    7: "safety_shoes"
}

def set_required_classes(gear_csv):
    global REQUIRED_CLASSES
    names = [g.strip().lower() for g in gear_csv.split(",")]
    REQUIRED_CLASSES = {k: v for k, v in CLASS_NAMES.items() if v.lower() in names and k != 0 and k != 2}
    print("[INFO] Required safety gear classes set to:", REQUIRED_CLASSES)

def extract_detections(results):
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0]) if box.id is not None else -1  # <-- ADD THIS
        detections.append({
            "class": cls_id,
            "label": CLASS_NAMES[cls_id],
            "conf": conf,
            "box": (x1, y1, x2, y2),
            "track_id": track_id  # <-- ADD THIS
        })
    return detections


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = max(1, (boxA[2] - boxA[0] + 1)) * max(1, (boxA[3] - boxA[1] + 1))
    boxBArea = max(1, (boxB[2] - boxB[0] + 1)) * max(1, (boxB[3] - boxB[1] + 1))

    return interArea / float(boxAArea + boxBArea - interArea)

def detect_violators(detections, frame):
    persons = [d for d in detections if d['class'] == 0]
    violators = []

    for person in persons:
        x1, y1, x2, y2 = person['box']
        gear_found = set()

        for d in detections:
            if d['class'] != 0 and iou(person['box'], d['box']) > 0.3:
                gear_found.add(d['class'])

        missing = [cls for cls in REQUIRED_CLASSES if cls not in gear_found]

        if missing:
            cropped = frame[y1:y2, x1:x2]
            violators.append({
                "image": cropped,
                "missing": [REQUIRED_CLASSES[cls] for cls in missing],
                "track_id":person.get("track_id", -1)
            })
    return violators

def save_violation_image(img, missing_items, output_dir):
    label = "_".join(missing_items)
    filename = f"{label}_{np.random.randint(10000)}.jpg"
    path = os.path.join(output_dir, "incomplete_gears", filename)
    cv2.imwrite(path, img)
    return path

def generate_pdf(image_paths, pdf_path):
    pdf = FPDF()
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Resize image to fit A4 width if too large
            if img.width > 1000:
                img = img.resize((800, int(img.height * 800 / img.width)))

            temp_path = img_path.replace(".jpg", "_resized.jpg")
            img.save(temp_path)

            pdf.add_page()
            pdf.image(temp_path, x=10, y=10, w=190)

            if os.path.exists(temp_path):
                os.remove(temp_path)  # delete resized image
        except Exception as e:
            print(f"Error adding image to PDF: {img_path} -> {e}")

    pdf.output(pdf_path)

def draw_annotations(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det['box']
        track_text = f" ID:{det['track_id']}" if det.get("track_id", -1) != -1 else ""
        label = f"{det['label']}{track_text} {det['conf']:.2f}"
        color = (0, 255, 0) if det['class'] != 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame
