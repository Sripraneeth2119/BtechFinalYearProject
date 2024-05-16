import cv2
import base64
import time
import json 

## Function to convert frame to Base64 encoded string and dump to JSON file
def frame_to_json(frame, json_file,flag):
    # Convert frame to base64 encoded string
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    frame_base64 = base64.b64encode(frame_bytes).decode()

    # Get current time in a human-readable format
    current_time = time.ctime()
    
    # Write frame JSON data to file
    json_data = {"time": current_time, "frame": frame_base64,"flag":flag}
    json.dump(json_data, json_file, indent=4)
    json_file.write('\n')  # Add newline for readability