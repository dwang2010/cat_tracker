import threading
import cv2

# blueprint for threaded webcam image frame retrieval

class WebcamFrameGrabber:
    def __init__(self, src: int = 0):
        # open webcam for video capture
        # buffer the last frame
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.done = False
        self.grab_frames()

    def grab_frames(self) -> None:
        # continually grab frames from webcam
        while True:
            if self.done: return
            (self.ret, self.frame) = cv2.read()

    def read(self) -> None:
        # return last collected frame
        return self.frame

    def stop(self) -> None:
        # stop collecting / cleanup the process
        self.done = True
        self.cap.release()
