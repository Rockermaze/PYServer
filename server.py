from fastapi import FastAPI, WebSocket
import base64
import numpy as np
import cv2
import json

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 WebSocket connection established!")

    while True:
        try:
            data = await websocket.receive_json()  # ✅ Properly receive JSON
            image_data = data.get("image")

            if not image_data:
                print("⚠️ No image data received.")
                await websocket.send_json({"status": "error", "message": "No image data"})
                continue

            # Decode base64 image
            frame_data = base64.b64decode(image_data)

            # Convert to NumPy array
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                print("✅ Frame received successfully.")
                await websocket.send_json({"status": "received"})
            else:
                print("⚠️ Failed to decode frame.")
                await websocket.send_json({"status": "error", "message": "Failed to decode frame"})

        except json.JSONDecodeError:
            print("❌ Error: Invalid JSON format.")
            await websocket.send_json({"status": "error", "message": "Invalid JSON format"})
        except Exception as e:
            print(f"❌ Unexpected Error: {e}")
            break

    print("🔌 WebSocket connection closed.")
