from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from ultralytics import YOLO
from pathlib import Path

import shutil
import uuid
from pathlib import Path
import shutil
app = FastAPI()
STATIC_DIR = "userdata"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Load YOLO model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Create directories
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./runs/detect/predict/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/detect/")
async def detect_video(file: UploadFile = File(...)):
    """
    Upload a video, process it using the YOLO model, and return the processed video.
    """
    input_path = f"{UPLOAD_FOLDER}/{file.filename}"
    output_path = f"{OUTPUT_FOLDER}/{file.filename.replace("mp4","avi")}"
    
    # Save uploaded file
    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # Process the video
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    process_video(input_path, output_path)
    
    # Return the processed video
    return FileResponse(output_path, media_type="video/avi", filename=f"{file.filename}")

def process_video(input_path: str, output_path: str):
    """
    Detect objects in the video and save the output with bounding boxes.
    """
    model = YOLO(MODEL_PATH)
    results = model(input_path,save=True)


# Upload and output directories
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/detect/image/")
async def detect_image(uid: str = Form(...), image: UploadFile = File(...)):
    """
    Detect objects in an uploaded image and return the processed image URL.
    """
    user_folder = os.path.join(STATIC_DIR, uid)
    os.makedirs(user_folder, exist_ok=True)

    # Save uploaded image
    image_filename = image.filename
    input_path = os.path.join(user_folder, image_filename)
    with open(input_path, "wb") as f:
        f.write(await image.read())

    # Run YOLO detection
    results = model(input_path)
    result_img = results[0].plot()  # image with bounding boxes

    # Save output image
    output_filename = f"output_{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(user_folder, output_filename)
    cv2.imwrite(output_path, result_img)

    # Create URL to access image
    image_url = f"/static/{uid}/{output_filename}"
    return JSONResponse(content={"image_url": image_url})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 