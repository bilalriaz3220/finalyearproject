from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
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
# Streaming Detection Endpoint
# -------------------------
@app.post("/stream_video/")
async def stream_video(file: UploadFile = File(...)):
    try:
        # Save the uploaded video
        uid = uuid.uuid4().hex
        input_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())

        # Return a streaming response
        return StreamingResponse(
            process_video_stream(input_path),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# -------------------------
# Dual-Model Frame-by-Frame Generator
# -------------------------
def process_video_stream(input_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict using both models
        results1 = model1.predict(frame, conf=0.5, verbose=False)
        results2 = model2.predict(frame, conf=0.5, verbose=False)

        # Overlay predictions
        img1 = results1[0].plot()
        img2 = results2[0].plot()
        blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', blended)
        frame_bytes = buffer.tobytes()

        # Yield for MJPEG stream
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()

# -------------------------
# Run with Uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
