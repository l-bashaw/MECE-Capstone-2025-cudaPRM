import json
import time
import websocket

# ROSBridge WebSocket server address
ROSBRIDGE_SERVER = "ws://localhost:9090"

# Topic and message type
TOPIC = "/trajectory"
MSG_TYPE = "std_msgs/Float32MultiArray"

def publish_trajectory(ws, trajectory):
    """
    Publishes a trajectory to ROS2 via rosbridge.
    trajectory: list of states, each state is a list of 5 floats
    """
    flat_data = [item for state in trajectory for item in state]
    msg = {
        "op": "publish",
        "topic": TOPIC,
        "msg": {
            "data": flat_data,
            "layout": {
                "dim": [
                    {"label": "states", "size": len(trajectory), "stride": 5 * len(trajectory)},
                    {"label": "components", "size": 5, "stride": 5}
                ],
                "data_offset": 0
            }
        }
    }
    ws.send(json.dumps(msg))

# # Example usage
# def main():

#     ws = websocket.create_connection(ROSBRIDGE_SERVER)

#     while True:
#         trajectory = [
#             [1.0, 2.0, 3.0, 4.0, 5.0],
#             [6.0, 7.0, 8.0, 9.0, 10.0],
#             [11.0, 12.0, 13.0, 14.0, 15.0]
#         ]

#         publish_trajectory(ws, trajectory)
#         time.sleep(0.02)
#     ws.close()

# if __name__ == "__main__":
#     main()