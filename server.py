import argparse, base64, time
import cv2, zmq
import WebcamGrabber as wg

# blueprint for video feed (source) streaming service
class Streamer:
    def __init__(self, addr: str, port: str):
        # bind to target addr:port for pushing video feed
        sock = "tcp://" + addr + ":" + port
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(sock)

        # start camera process, with startup delay
        self.camera = wg.WebcamGrabber(0)
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

if __name__ == "__main__":
    # check for script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--addr', help='server address')
    parser.add_argument('-p', '--port', help='server port')
    args = parser.parse_args()

    print ("Starting server...")
    serve = Streamer(args.addr, args.port)
    print ("Shutting down...")
