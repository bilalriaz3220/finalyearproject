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

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# -------------------------
# Load both YOLO models
# -------------------------
model1 = YOLO("best.pt")
model2 = YOLO("bestB.pt")

# -------------------------
# Detect in Video Endpoint
# -------------------------
@app.post("/detect_video/")
async def detect_video(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")
    user_output_folder = os.path.join(STATIC_DIR, uid)
    os.makedirs(user_output_folder, exist_ok=True)
    output_filename = f"output_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(user_output_folder, output_filename)

    # Save uploaded video
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    # Process video with both models (1 frame/sec)
    process_video(input_path, output_path)

    # Return public URL
    ec2_ip = "18.204.199.142"  # Replace with your EC2 public IP or domain
    video_url = f"http://{ec2_ip}:8000/static/{uid}/{output_filename}"
    return JSONResponse(content={"video_url": video_url})


# -------------------------
# Detect in Image Endpoint
# -------------------------
@app.post("/detect_image/")
async def detect_image(image: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    user_output_folder = os.path.join(STATIC_DIR, uid)
    os.makedirs(user_output_folder, exist_ok=True)

    # Save image
    input_path = os.path.join(user_output_folder, image.filename)
    with open(input_path, "wb") as f:
        f.write(await image.read())

    # Read image using OpenCV
    img = cv2.imread(input_path)

    # Run both models
    results1 = model1(img, verbose=False)
    results2 = model2(img, verbose=False)

    # Get plotted images
    img1 = results1[0].plot()
    img2 = results2[0].plot()

    # Blend both model outputs
    blended_img = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    # Save output image
    output_filename = f"output_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(user_output_folder, output_filename)
    cv2.imwrite(output_path, blended_img)

    # Return public URL
    ec2_ip = "18.204.199.142"  # Replace with your EC2 public IP or domain
    image_url = f"http://{ec2_ip}:8000/static/{uid}/{output_filename}"
    return JSONResponse(content={"image_url": image_url})


# -------------------------
# Process Video Function (1 frame per second)
# -------------------------
def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Process one frame per second

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Run both models
            results1 = model1(frame, verbose=False)
            results2 = model2(frame, verbose=False)

            # Plot and overlay results
            frame1 = results1[0].plot()
            frame2 = results2[0].plot()
            blended_frame = cv2.addWeighted(frame1, 0.5, frame2, 0.5, 0)

            # Repeat the frame to fill in skipped seconds
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
