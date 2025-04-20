from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
import cv2
from pathlib import Path
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
# Load YOLO Model
# -------------------------
model = YOLO("best.pt")  # Ensure 'best.pt' exists in your working directory

# -------------------------
# Detect in Video Endpoint
# -------------------------
@app.post("/detect_video/")
async def detect_video(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")
    output_folder = os.path.join(STATIC_DIR, uid)
    os.makedirs(output_folder, exist_ok=True)
    output_filename = f"output_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(output_folder, output_filename)

    # Save uploaded video
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process the video
    process_video(input_path, output_path)

    # Construct public URL
    ec2_ip = "18.204.199.142"  # Replace with your EC2 IP or domain
    video_url = f"http://{ec2_ip}:8000/static/{uid}/{output_filename}"
    return JSONResponse(content={"video_url": video_url})


# -------------------------
# Detect in Image Endpoint
# -------------------------
@app.post("/detect_image/")
async def detect_image(image: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    output_folder = os.path.join(STATIC_DIR, uid)
    os.makedirs(output_folder, exist_ok=True)

    input_path = os.path.join(output_folder, image.filename)
    with open(input_path, "wb") as f:
        f.write(await image.read())

    # Read and process image
    img = cv2.imread(input_path)
    results = model(img, verbose=False)
    output_img = results[0].plot()

    # Blend both model outputs
    blended_img = cv2.addWeighted(img1,0.5, 0)

    # Save output image
    output_filename = f"output_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, output_img)

    ec2_ip = "18.204.199.142"  # Replace with your EC2 IP or domain
    image_url = f"http://{ec2_ip}:8000/static/{uid}/{output_filename}"
    return JSONResponse(content={"image_url": image_url})


# -------------------------
# Helper: Process Video (1 Frame Per Second)
# -------------------------
def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            results = model(frame, verbose=False)
            result_frame = results[0].plot()

            for _ in range(frame_interval):
                out.write(blended_frame)

        frame_idx += 1

    cap.release()
    out.release()


# -------------------------
# Run with Uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
