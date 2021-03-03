import argparse, base64, time
import cv2, zmq
import FrameGrabber as fg
from os.path import join as path_join

from cat_trainer import ModelClass

class Streamer:
    """
    methods for starting cat tracking video server
    """

    def __init__(self, addr: str, port: str, model_path: str):
        """ binds target tcp address for video feed publishing """
        sock = "tcp://{}:{}".format(addr, port)
        context = zmq.Context()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(sock)

        # create ModelClass object for detectron2 inference
        self.model = ModelClass()
        cfg = path_join(model_path, "cat_frcnn_r50_3x.yaml")
        weights = path_join(model_path, "model_final_shrunk.pth")
        self.model.cfg_load(cfg, weights)

        # start camera process, with startup delay
        self.camera = fg.FrameGrabber(0)
        self.camera.start()
        time.sleep(1)

        # start video stream
        self.start()

    def start(self):
        """ starts video stream, and continually pushes updates to clients """
        while True:
            try:
                # collect video frame and scale down frame size
                frame = self.camera.read()

                scale = 50
                width = int(frame.shape[1] * scale / 100)
                height = int(frame.shape[0] * scale / 100)
                frame = cv2.resize(frame, (width, height),
                                   interpolation = cv2.INTER_AREA)

                # perform inference on collected frame
                output = self.model.infer(frame)

                # encode image, convert to text and publish
                encoded, buff = cv2.imencode(".jpg", output)
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
    parser.add_argument("-a", "--addr", default="192.168.55.1",
                        help="server address")
    parser.add_argument("-p", "--port", default="5556",
                        help="server port")
    parser.add_argument("-m", "--model", default="./model_backup",
                        help="path to model files")
    args = parser.parse_args()

    print ("Starting server...")
    serve = Streamer(args.addr, args.port, args.model)
    print ("Shutting down...")
