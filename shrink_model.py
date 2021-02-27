import torch
import argparse
from os.path import splitext

# remove solver states stored in model to reduce filesize
def shrink_model(model_path: str) -> None:
    model = torch.load(model_path)

    del model["optimizer"]
    del model["scheduler"]
    del model["iteration"]

    path, ext = splitext(model_path)
    trunc_path = path + "_shrunk.pth"

    torch.save(model, trunc_path)

if (__name__ == '__main__'):
    print ("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="location of model file")
    args = parser.parse_args()

    shrink_model(args.path)
