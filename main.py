from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import uuid
import cv2
from ultralytics import YOLO
import tempfile
import io

app = FastAPI()

model1 = YOLO("best.pt")
model2 = YOLO("bestB.pt")

@app.post("/stream_video/")
async def stream_video(video: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    output_folder = os.path.join("userdata", uid)
    os.makedirs(output_folder, exist_ok=True)

    # Save uploaded video temporarily
    video_path = os.path.join(output_folder, f"{uid}.mp4")
    with open(video_path, "wb") as f:
        f.write(await video.read())

    def generate_frames():
        cap = cv2.VideoCapture(video_path)

        while True:
            success, frame = cap.read()
            if not success:
                break

            # YOLO detection on the frame
            results1 = model1.predict(frame, conf=0.5, verbose=False)
            results2 = model2.predict(frame, conf=0.5, verbose=False)

            img1 = results1[0].plot()
            img2 = results2[0].plot()
            blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

            # Encode the frame to JPEG
            _, jpeg = cv2.imencode(".jpg", blended)
            frame_bytes = jpeg.tobytes()

            # Yield in multipart format (like MJPEG stream)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        cap.release()

    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


 
# -------------------------
# Run with Uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
