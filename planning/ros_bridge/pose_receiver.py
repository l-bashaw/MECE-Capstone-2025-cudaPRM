import json
import threading
import websocket

ROSBRIDGE_SERVER = "ws://localhost:9090"
POSE_TOPIC = "/poses"
NAMES_TOPIC = "/pose_names"

latest_poses = []
latest_names = []

def on_message(ws, message):
    global latest_poses, latest_names
    msg = json.loads(message)
    topic = msg.get("topic")
    if topic == POSE_TOPIC and "msg" in msg:
        poses = msg["msg"]["poses"]
        # Each pose is a dict with position and orientation
        latest_poses = poses
    elif topic == NAMES_TOPIC and "msg" in msg:
        names_str = msg["msg"]["data"]
        latest_names = names_str.split(',')

def on_open(ws):
    subscribe_pose = {
        "op": "subscribe",
        "topic": POSE_TOPIC,
        "type": "geometry_msgs/PoseArray"
    }
    subscribe_names = {
        "op": "subscribe",
        "topic": NAMES_TOPIC,
        "type": "std_msgs/String"
    }
    ws.send(json.dumps(subscribe_pose))
    ws.send(json.dumps(subscribe_names))
    print("Subscribed to /poses and /pose_names")

def rosbridge_listener():
    ws = websocket.WebSocketApp(
        ROSBRIDGE_SERVER,
        on_open=on_open,
        on_message=on_message
    )
    ws.run_forever()

def get_latest_poses_and_names():
    return latest_poses, latest_names

def start_rosbridge_thread():
    thread = threading.Thread(target=rosbridge_listener, daemon=True)
    thread.start()

# Example usage:
if __name__ == "__main__":
    start_rosbridge_thread()
    import time
    while True:
        poses, names = get_latest_poses_and_names()
        if poses and names:
            print("Latest poses and names:")
            for i, (pose, name) in enumerate(zip(poses, names)):
                print(f"{i}: {name} -> {pose}")
                print("x: ", pose['position']['x'], 
                      "y: ", pose['position']['y'], 
                      "z: ", pose['position']['z'])
                print("qx:", pose['orientation']['x'], 
                      "qy:", pose['orientation']['y'], 
                      "qz:", pose['orientation']['z'], 
                      "qw:", pose['orientation']['w'])
        time.sleep(1)