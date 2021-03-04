import base64, argparse
import cv2, zmq, imutils
import numpy as np

class Viewer:
    """
    methods for client-side video feed viewing
    """

    def __init__(self, addr: str, port: str):
        """ connects to target addr:port to receive incoming video feed """
        sock = "tcp://{}:{}".format(addr, port)
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(sock)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(""))

        # start the viewer
        self.start()

    def start(self) -> None:
        """ displays received video feed """
        while True:
            try:
                # receive string from server, convert to image
                img_str = self.socket.recv_string()
                buff = base64.b64decode(img_str)
                image = np.fromstring(buff, dtype=np.uint8)
                frame = cv2.imdecode(image, 1)

                frame = imutils.resize(frame, width=1024)

                # display image
                cv2.imshow("Cat Tracker", frame)
                cv2.waitKey(1)

            except KeyboardInterrupt:
                # close video feed and cleanup
                cv2.destroyAllWindows()
                break

if __name__ == "__main__":
    # check for script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--addr', default="192.168.55.1",
                        help='server address')
    parser.add_argument('-p', '--port', default="5556",
                        help='server port')
    args = parser.parse_args()

    # start client process for viewing video stream
    print ("Starting viewer...")
    view = Viewer(args.addr, args.port)
    print ("Shutting down...")
