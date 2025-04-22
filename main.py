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

        # Process video with dual model detection
        process_video(input_path, output_path)

        # Return public URL
        ec2_ip = "3.84.34.96"  # Replace with your EC2 IP
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

        results1 = model1.predict(img, conf=0.5, verbose=False)
        results2 = model2.predict(img, conf=0.5, verbose=False)

        img1 = results1[0].plot()
        img2 = results2[0].plot()

        blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        output_filename = f"output_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(user_output_folder, output_filename)
        cv2.imwrite(output_path, blended)

        ec2_ip = "3.84.34.96"
        image_url = f"http://{ec2_ip}:8000/static/{uid}/{output_filename}"
        return JSONResponse(content={"image_url": image_url})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# Dual-Model Frame-by-Frame Video Processor
# -------------------------
def process_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict and visualize with both models
        results1 = model1.predict(frame, conf=0.5, verbose=False)
        results2 = model2.predict(frame, conf=0.5, verbose=False)

        img1 = results1[0].plot()
        img2 = results2[0].plot()

        # Blend both outputs
        blended_frame = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        # Write processed frame
        out.write(blended_frame)

        frame_count += 1

    cap.release()
    out.release()

# -------------------------
# Run with Uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
