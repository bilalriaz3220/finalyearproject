from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import cv2
from ultralytics import YOLO

app = FastAPI()

# -------------------------
# Configuration
# -------------------------
STATIC_DIR = "userdata"
UPLOAD_FOLDER = "uploads"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------
# Load YOLO models
# -------------------------
model1 = YOLO("best.pt")     # First YOLO model
model2 = YOLO("bestB.pt")    # Second YOLO model

# -------------------------
# Video Detection Endpoint (Segmented)
# -------------------------
@app.post("/detect_video/")
async def detect_video(file: UploadFile = File(...)):
    try:
        uid = uuid.uuid4().hex
        input_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")
        user_output_folder = os.path.join(STATIC_DIR, uid)
        os.makedirs(user_output_folder, exist_ok=True)

        # Save uploaded video
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process and return 1-second chunks
        segment_urls = process_video_by_second(input_path, user_output_folder)

        ec2_ip = "3.86.60.160"  # Replace with your public EC2 IP or domain
        full_urls = [f"http://{ec2_ip}:8000/static/{uid}/{name}" for name in segment_urls]
        return JSONResponse(content={"segments": full_urls})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# Image Detection Endpoint
# -------------------------
@app.post("/detect_image/")
async def detect_image(image: UploadFile = File(...)):
    try:
        uid = uuid.uuid4().hex
        user_output_folder = os.path.join(STATIC_DIR, uid)
        os.makedirs(user_output_folder, exist_ok=True)

        input_path = os.path.join(user_output_folder, image.filename)
        with open(input_path, "wb") as f:
            f.write(await image.read())

        img = cv2.imread(input_path)

        results1 = model1.predict(img, conf=0.5, verbose=False)
        results2 = model2.predict(img, conf=0.5, verbose=False)

        img1 = results1[0].plot()
        img2 = results2[0].plot()

        blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        output_filename = f"output_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(user_output_folder, output_filename)
        cv2.imwrite(output_path, blended)

        ec2_ip = "3.86.60.160"
        image_url = f"http://{ec2_ip}:8000/static/{uid}/{output_filename}"
        return JSONResponse(content={"image_url": image_url})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# Dual-Model Segment-by-Segment Processor
# -------------------------
def process_video_by_second(input_path: str, output_folder: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    segment_urls = []
    frame_count = 0
    segment_index = 0
    frames_per_segment = fps  # Process 1 second at a time

    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_buffer.append(frame)
        frame_count += 1

        if frame_count % frames_per_segment == 0:
            segment_filename = f"segment_{segment_index}.mp4"
            segment_path = os.path.join(output_folder, segment_filename)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))

            for f in frames_buffer:
                results1 = model1.predict(f, conf=0.5, verbose=False)
                results2 = model2.predict(f, conf=0.5, verbose=False)
                img1 = results1[0].plot()
                img2 = results2[0].plot()
                blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
                out.write(blended)

            out.release()
            segment_urls.append(segment_filename)
            segment_index += 1
            frames_buffer.clear()

    # If any leftover frames after loop
    if frames_buffer:
        segment_filename = f"segment_{segment_index}.mp4"
        segment_path = os.path.join(output_folder, segment_filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(segment_path, fourcc, fps, (width, height))

        for f in frames_buffer:
            results1 = model1.predict(f, conf=0.5, verbose=False)
            results2 = model2.predict(f, conf=0.5, verbose=False)
            img1 = results1[0].plot()
            img2 = results2[0].plot()
            blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
            out.write(blended)

        out.release()
        segment_urls.append(segment_filename)

    cap.release()
    return segment_urls

# -------------------------
# Local run (optional)
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
