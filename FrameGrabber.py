import threading
import cv2

class FrameGrabber:
    """
    methods for grabbing webcam image frames
    """

    def __init__(self, src: int = 0):
        """ inits FrameGrabber class """
        self.cap = cv2.VideoCapture(src)

        # limit image resolution based on camera specs
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        self.ret, self.frame = self.cap.read()
        self.done = False

    def start(self) -> None:
        """ start frame capture thread """
        threading.Thread(target=self._grab_frames).start()

    def _grab_frames(self) -> None:
        """ continually grabs frames from webcam """
        while True:
            if self.done: return
            (self.ret, self.frame) = self.cap.read()

    def read(self) -> None:
        """ returns last collected frame """
        return self.frame

    def stop(self) -> None:
        """ stops collection thread and releases camera """
        self.done = True
        self.cap.release()
