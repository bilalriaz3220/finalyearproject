from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import cv2
from ultralytics import YOLO

app = FastAPI()

STATIC_DIR = "userdata"
UPLOAD_FOLDER = "uploads"
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

model1 = YOLO("best.pt")
model2 = YOLO("bestB.pt")

@app.post("/process_video_json/")
async def process_video_json(file: UploadFile = File(...)):
    try:
        uid = uuid.uuid4().hex
        input_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")
        output_path = os.path.join(STATIC_DIR, f"{uid}_result.mp4")

        # Save uploaded file
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())

        # Process video and get JSON results
        json_results = process_and_save_video(input_path, output_path)

        # Create accessible URL for result video
        video_url = f"/static/{os.path.basename(output_path)}"

        return JSONResponse(content={
            "status": "success",
            "video_url": video_url,
            "results": json_results
        })

    except Exception as e:
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

def process_and_save_video(input_path: str, output_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    results_json = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_entry = {"frame": frame_index, "model1": [], "model2": []}

        results1 = model1.predict(frame, conf=0.5, verbose=False)
        results2 = model2.predict(frame, conf=0.5, verbose=False)

        for r in results1[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            result_entry["model1"].append({
                "class_id": int(cls),
                "confidence": float(conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        for r in results2[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            result_entry["model2"].append({
                "class_id": int(cls),
                "confidence": float(conf),
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

        # Visualize detections on frame
        img1 = results1[0].plot()
        img2 = results2[0].plot()
        blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

        out.write(blended)
        results_json.append(result_entry)
        frame_index += 1

    cap.release()
    out.release()
    return results_json


# -------------------------
# Run with Uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
