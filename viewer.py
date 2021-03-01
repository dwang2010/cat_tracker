import argparse
import WebcamStream as WS

if __name__ == "__main__":
    # check for script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--addr', help='server address')
    parser.add_argument('-p', '--port', help='server port')
    args = parser.parse_args()

    print ("Starting viewer...")
    view = WS.Viewer(args.addr, args.port)
    print ("Shutting down...")
