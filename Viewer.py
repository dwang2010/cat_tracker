import base64, argparse
import cv2, zmq
import numpy as np

# blueprint for video feed viewing service
class Viewer:
    def __init__(self, addr: str, port: str):
        # bind to addr to receive incoming image feed
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.bind("tcp://" + addr + ":" + port)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(""))
        self.start()

    def start(self) -> None:
        # display image feed
        while True:
            try:
                img_str = self.socket.recv_string()
                buff = base64.b64decode(img_str)
                image = np.fromstring(image, dtype=np.uint8)
                source = cv2.imdecode(image, 1)
                cv2.imshow("Cat Tracker", source)
                cv2.waitKey(1)

            except KeyboardInterrupt:
                # close video feed and cleanup
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    # check for script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--addr', help='server address')
    parser.add_argument('-p', '--port', help='server port')
    args = parser.parse_args()

    print ("Starting viewer...")
    view = Viewer(args.addr, args.port)
    print ("Shutting down...")
