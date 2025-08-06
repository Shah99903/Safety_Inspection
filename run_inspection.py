# run_inspection.py
import os
import cv2
from datetime import datetime
from ultralytics import YOLO
from scripts.utils import (
    extract_detections,
    detect_violators,
    draw_annotations,
    save_violation_image,
    generate_pdf
)
from scripts.email_alerts import send_alert
from scripts.tracker.byte_tracker import BYTETracker

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

model_path = os.getenv("MODEL_PATH")
input_video = os.getenv("VIDEO_SOURCE")
required_gear = os.getenv("REQUIRED_GEAR", "")
from scripts.utils import set_required_classes
set_required_classes(required_gear)

# Set up folders
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"results/{timestamp}"
os.makedirs(f"{output_dir}/incomplete_gears", exist_ok=True)
os.makedirs(f"{output_dir}/frames", exist_ok=True)

# Load YOLOv8 model
model = YOLO(model_path)
cap = cv2.VideoCapture(0 if input_video == "" else input_video)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out_video = cv2.VideoWriter(f"{output_dir}/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Initialize ByteTrack
tracker = BYTETracker()

violator_images = []
saved_track_ids = set()  # To avoid duplicates

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = extract_detections(results)

    # Format detections for tracker [x1, y1, x2, y2, score, class]
    dets_for_tracker = [
        det["box"] + (det["conf"], det["class"])
        for det in detections if det["class"] == 0  # Track only persons
    ]
    tracks = tracker.update_tracks(dets_for_tracker, frame)

    violators = detect_violators(detections, frame)

    for track in tracks:
        track_id = track["track_id"]
        bbox = track["bbox"]
        x, y = track["center"]  # fixed: use the center from your tracker
        x1, y1, x2, y2 = int(x - 50), int(y - 50), int(x + 50), int(y + 50)
        bbox = (x1, y1, x2, y2)

        for violator in violators:
            if track_id not in saved_track_ids:
                person_img = violator["image"]
                missing_items = violator["missing"]
                path = save_violation_image(person_img, missing_items, output_dir)
                violator_images.append(path)
                saved_track_ids.add(track_id)



    annotated = draw_annotations(frame, detections)
    out_video.write(annotated)
    cv2.imshow("Safety Inspection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()

# Generate summary PDF
pdf_path = f"{output_dir}/summary.pdf"
if violator_images:
    generate_pdf(violator_images, pdf_path)
    send_alert(
        subject="Safety Gear Violation Alert",
        body="Violations detected. See the attached image.",
        attachments=violator_images[:1]
    )

print(f"Processing complete. Results saved to {output_dir}")
