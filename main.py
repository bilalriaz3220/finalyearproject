from fastapi.responses import StreamingResponse
import json
import time

@app.post("/stream_video_detection/")
async def stream_video_detection(file: UploadFile = File(...)):
    uid = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_FOLDER, f"{uid}_{file.filename}")
    output_path = os.path.join(STATIC_DIR, f"{uid}_result.mp4")

    with open(input_path, "wb") as buffer:
        buffer.write(await file.read())

    def generate():
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            yield json.dumps({"error": "Video not opened"}) + "\n"
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_index = 0

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

            # Visualize detections
            img1 = results1[0].plot()
            img2 = results2[0].plot()
            blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
            out.write(blended)

            # Send one frame’s result
            yield f"data: {json.dumps(result_entry)}\n\n"
            frame_index += 1

        cap.release()
        out.release()

        # Final video URL
        video_url = f"/static/{os.path.basename(output_path)}"
        yield f"data: {json.dumps({'done': True, 'video_url': video_url})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# -------------------------
# Run with Uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
