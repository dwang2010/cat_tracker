import threading, base64, time
import cv2, zmq
import numpy as np
import WebcamFrameGrabber as WFG

# blueprint for video feed (source) streaming service
class Streamer:
    def __init__(self, addr: str, port: str):
        # connect to addr for pushing image feed
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind("tcp://" + addr + ":" + port)
        self.camera = WFG.WebcamFrameGrabber(0)

        # camera startup delay
        time.sleep(1)
        self.start()

    def start(self):
        # start video stream, continually pushing updates to clients
        while True:
            try:
                # get frame, encode to memory buffer
                frame = self.camera.read()
                encode_pass, buff = cv2.imencode(".jpg", frame)

                # convert image to text, and publish
                img_str = base64.b64encode(buff)
                self.socket.send(img_str)

            except KeyboardInterrupt:
                # stop video feed and cleanup
                self.camera.stop()
                cv2.destroyAllWindows()
                print ("Closing!")
                break

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
                cv2.imshow("test", source)
                cv2.waitKey(1)

            except KeyboardInterrupt:
                # close video feed and cleanup
                cv2.destroyAllWindows()
                break
