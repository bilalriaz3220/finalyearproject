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
# Load YOLO model
# -------------------------
model1 = YOLO("best.pt")  # Make sure 'best.pt' exists in your directory

# -------------------------
# Detect in Video Endpoint
# -------------------------
@app.post("/detect_video/")
async def detect_video(file: UploadFile = File(...)):
    try:
        uid = uuid.uuid4().hex
        input_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")
        user_output_folder = os.path.join(STATIC_DIR, uid)
        os.makedirs(user_output_folder, exist_ok=True)
        output_filename = f"output_{uuid.uuid4().hex}.mp4"
        output_path = os.path.join(user_output_folder, output_filename)

        # Save uploaded video
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process video
        process_video(input_path, output_path)

        # Return public URL
        ec2_ip = "18.204.199.142"  # Replace with your EC2 IP or domain
        video_url = f"http://{ec2_ip}:8000/static/{uid}/{output_filename}"
        return JSONResponse(content={"video_url": video_url})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# Detect in Image Endpoint
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

        results1 = model1(img, verbose=False)
        img1 = results1[0].plot()

        output_filename = f"output_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(user_output_folder, output_filename)
        cv2.imwrite(output_path, img1)

        ec2_ip = "18.204.199.142"
        image_url = f"http://{ec2_ip}:8000/static/{uid}/{output_filename}"
        return JSONResponse(content={"image_url": image_url})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# Process Video Function (1 frame per second)
# -------------------------
def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

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
            results1 = model1(frame, verbose=False)
            frame1 = results1[0].plot()

            for _ in range(frame_interval):
                out.write(frame1)

        frame_idx += 1

    cap.release()
    out.release()

# -------------------------
# Run with Uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
