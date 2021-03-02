import argparse, base64, time
import cv2, zmq
import FrameGrabber as fg

class Streamer:
    """
    methods for starting cat tracking video server
    """

    def __init__(self, addr: str, port: str):
        """ binds target tcp address for video feed publishing """
        sock = "tcp://{}:{}".format(addr, port)
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(sock)

        # start camera process, with startup delay
        self.camera = fg.FrameGrabber(0)
        time.sleep(1)
        self.start()

    def start(self):
        """ starts video stream, and continually pushes updates to clients """
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

if __name__ == "__main__":
    # check for script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--addr', help='server address')
    parser.add_argument('-p', '--port', help='server port')
    args = parser.parse_args()

    print ("Starting server...")
    serve = Streamer(args.addr, args.port)
    print ("Shutting down...")
