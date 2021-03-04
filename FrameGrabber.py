import threading
import cv2

class FrameGrabber:
    """
    methods for grabbing webcam image frames
    """

    def __init__(self, width: int, height: int, src: int=0) -> None:
        """ inits FrameGrabber class and camera settings """
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.end = False
        self.frame = None

    def start_cap(self) -> None:
        """ start thread grabbing frames to clear buffer """
        t = threading.Thread(target=self._get_frames).start()

    def _get_frames(self):
        """ continually grab frames as fast as possible """
        while True:
            if self.end: return
            ret, self.frame = self.cap.read()

    def read(self) -> None:
        """ returns image frame from camera """
        return self.frame

    def stop(self) -> None:
        """ releases camera and closes frame grabbing thread """
        self.end = True
        self.cap.release()
