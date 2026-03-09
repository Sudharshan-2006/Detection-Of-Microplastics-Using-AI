"""
main.py
Microplastic detection + measurement + CSV + annotated image (no color).
"""

import os
import math
import csv
import cv2
import numpy as np
import datetime

# ---------- SETTINGS ----------
USE_TFLITE = True
TFLITE_MODEL = "model_unquant.tflite"   # per-particle Teachable Machine model
LABELS_FILE = "labels.txt"

IMAGE_PATH = "test.jpg"
OUTPUT_IMAGE = "annotated_output.jpg"
CSV_OUTPUT = "particle_results.csv"     # will be overwritten with timestamped name

MIN_AREA_PX = 30
GAUSSIAN_BLUR = (7, 7)
MORPH_KERNEL = (5, 5)

# ---------- TFLITE LOADING ----------
tflite_interpreter = None
tflite_io = None  # (input_details, output_details, labels)


def load_tflite_if_available(path, labels_path=None):
    global tflite_interpreter, tflite_io
    if not path or not os.path.exists(path):
        print("TFLite model not found; per-particle classification disabled.")
        return None
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        labels = []
        if labels_path and os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                labels = [line.strip() for line in f.readlines()]
        print("Loaded TFLite model:", path)
        tflite_interpreter = interpreter
        tflite_io = (input_details, output_details, labels)
        return interpreter
    except Exception as e:
        print("Could not load TFLite model (will continue without per-particle ML).")
        print("Error:", e)
        return None


def run_tflite_on_crop(interpreter, io_details, crop):
    in_det, out_det, labels = io_details
    shape = in_det[0]["shape"]  # (1, h, w, 3)
    target_h, target_w = int(shape[1]), int(shape[2])
    img = cv2.resize(crop, (target_w, target_h))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    interpreter.set_tensor(in_det[0]["index"], img)
    interpreter.invoke()
    out = interpreter.get_tensor(out_det[0]["index"])[0]
    idx = int(np.argmax(out))
    score = float(out[idx])
    label = labels[idx] if labels else str(idx)
    return label, score


# ---------- DETECTION + ANALYSIS ----------
def detect_and_analyze(image_path):
    global CSV_OUTPUT

    img = cv2.imread(image_path)
    if img is None:
        print("Error: image not found:", image_path)
        return

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # bowl detection (mask)
    bowl_mask = None
    bowl_circle = None
    try:
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=100, param2=35, minRadius=30, maxRadius=max(H, W)
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            circles = sorted(circles, key=lambda c: c[2], reverse=True)
            x, y, r = circles[0]
            bowl_mask = np.zeros((H, W), dtype=np.uint8)
            cv2.circle(bowl_mask, (x, y), r, 255, -1)
            bowl_circle = (x, y, r)
    except Exception:
        bowl_mask = None

    if bowl_mask is None:
        bowl_mask = np.ones((H, W), dtype=np.uint8) * 255

    bowl_area_px = np.count_nonzero(bowl_mask)

    # preprocessing
    blur = cv2.GaussianBlur(gray, GAUSSIAN_BLUR, 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mean_in = cv2.mean(gray, mask=bowl_mask)[0]
    if mean_in < 100:
        binary = cv2.bitwise_not(th) if np.mean(th) > 127 else th
    else:
        binary = th

    binary = cv2.bitwise_and(binary, binary, mask=bowl_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    particles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= MIN_AREA_PX:
            x, y, w, h = cv2.boundingRect(cnt)
            particles.append({"contour": cnt, "area_px": int(area), "bbox": (x, y, w, h)})

    particles = sorted(particles, key=lambda x: -x["area_px"])

    # TFLite
    if USE_TFLITE and os.path.exists(TFLITE_MODEL):
        load_tflite_if_available(TFLITE_MODEL, LABELS_FILE)
    else:
        global tflite_interpreter, tflite_io
        tflite_interpreter = None
        tflite_io = None

    annotated = img.copy()
    total_area = 0
    results_for_csv = []
    table_rows = []
    idx = 0

    for p in particles:
        idx += 1
        area_px = p["area_px"]
        total_area += area_px
        x, y, w, h = p["bbox"]
        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        crop = img[y0:y1, x0:x1]

        equi_d = math.sqrt(4 * area_px / math.pi)

        model_label = None
        model_conf = None
        if tflite_interpreter and tflite_io:
            try:
                model_label, model_conf = run_tflite_on_crop(
                    tflite_interpreter, tflite_io, crop
                )
            except Exception:
                model_label, model_conf = None, None

        cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
        txt = f"#{idx} A:{area_px}"
        cv2.putText(
            annotated,
            txt,
            (x0, max(12, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
        )

        table_rows.append(
            {
                "id": idx,
                "area_px": area_px,
                "equivalent_d_px": round(equi_d, 1),
                "model_label": model_label,
                "model_conf": round(model_conf, 2) if model_conf is not None else None,
            }
        )

        results_for_csv.append(
            [
                idx,
                area_px,
                round(equi_d, 1),
                model_label if model_label is not None else "",
                round(model_conf, 2) if model_conf is not None else "",
            ]
        )

    particle_count = len(table_rows)
    percent_area = (total_area / float(bowl_area_px)) * 100 if bowl_area_px > 0 else None

    if bowl_circle:
        x, y, r = bowl_circle
        cv2.circle(annotated, (x, y), r, (255, 0, 0), 2)

    summary_text = f"Count:{particle_count}  TotalA:{total_area}px"
    if percent_area is not None:
        summary_text += f"  {percent_area:.2f}%"
    cv2.putText(
        annotated,
        summary_text,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )

    cv2.imwrite(OUTPUT_IMAGE, annotated)
    print("Saved annotated image to:", OUTPUT_IMAGE)

    # ---------- CSV with timestamped name ----------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    CSV_OUTPUT = f"particle_results_{ts}.csv"

    with open(CSV_OUTPUT, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Area(px)", "Diameter(px)", "Class", "Confidence"])
        for row in results_for_csv:
            writer.writerow(row)
        writer.writerow([])
        writer.writerow(["Total Particles", particle_count])
        writer.writerow(["Total Area(px)", total_area])
        if percent_area is not None:
            writer.writerow(["Percent Bowl Area", f"{percent_area:.2f}%"])
    print("CSV saved as:", CSV_OUTPUT)

    # also print table to console
    print("\n--- ANALYSIS RESULT (TABLE FORMAT) ---\n")
    header = "ID | Area(px) | Diameter(px) | Class | Conf"
    print(header)
    print("-" * len(header))
    for r in table_rows:
        cls = r["model_label"] if r["model_label"] is not None else "-"
        conf = f"{r['model_conf']:.2f}" if r["model_conf"] is not None else "-"
        print(
            f"{r['id']:<3}| {r['area_px']:<9}| {r['equivalent_d_px']:<13}| "
            f"{cls:<5}| {conf}"
        )
    print("\n--------------------------------------")
    print(f"Total particles: {particle_count}")
    print(f"Total area(px): {total_area}")
    if percent_area is not None:
        print(f"Percent of bowl area: {percent_area:.2f}%")
    print("--------------------------------------\n")


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print("Error: input image not found:", IMAGE_PATH)
    else:
        if USE_TFLITE and os.path.exists(TFLITE_MODEL):
            load_tflite_if_available(TFLITE_MODEL, LABELS_FILE)
        detect_and_analyze(IMAGE_PATH)
