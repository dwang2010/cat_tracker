from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

import cv2

from os import listdir, makedirs
from os.path import basename, isdir, isfile
from os.path import join as path_join

import os, random, subprocess, tarfile, json, shutil, math
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set

class ModelClass:
    """
    methods for model configuration / training / inference
    """

    def __init__(self) -> None:
        """ init ModelClass object with null / default attributes """
        self.cfg = get_cfg()
        self.trainer = None
        self.predictor = None

    def cfg_setup(self,
                  num_images: int,
                  num_classes: int,
                  epochs: int = 20) -> None:
        """ setup model configuration for training

        Args:
            num_images: number of images in target dataset
            num_classes: number of distinct object classes in dataset
            epochs: number of times to cycle model through complete dataset
        """
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        print (self.cfg.MODEL.WEIGHTS)
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

        self.cfg.DATASETS.TRAIN = ("train",)
        self.cfg.DATASETS.TEST = ("val",)
        self.cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
        self.cfg.DATALOADER.NUM_WORKERS = 8 # threads

        self.cfg.SOLVER.IMS_PER_BATCH = 4 # batch size
        self.cfg.SOLVER.BASE_LR = 0.01

        ims_per_epoch = num_images // self.cfg.SOLVER.IMS_PER_BATCH
        self.cfg.SOLVER.MAX_ITER = ims_per_epoch * epochs
        print ("max iterations: {}".format(self.cfg.SOLVER.MAX_ITER))

        # decay learning rate at every epoch (down a decade)
        ttl = self.cfg.SOLVER.MAX_ITER
        lr_deltas = [i for i in range(ttl//epochs, ttl, ttl//epochs)]
        self.cfg.SOLVER.STEPS = lr_deltas
        self.cfg.SOLVER.GAMMA = 0.1
        self.cfg.SOLVER.CHECKPOINT_PERIOD = ims_per_epoch

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

        self.trainer = CustomTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)

    def cfg_load(self, cfg_path: str, model_path: str) -> None:
        """ load existing model configuration and weights for inference """
        if not isfile(cfg_path):
            print ("ERROR! Model file not found: {}".format(cfg_path))
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_path)
        self.cfg.MODEL.WEIGHTS = model_path
        self.predictor = DefaultPredictor(self.cfg)

    def learn(self) -> None:
        """ performs model training with loaded dataset """
        self.trainer.train()

    def infer(self, image):
        """ performs inference on provided images

        for color images, OpenCV imread() reads as NumPy array ndarray of row
        (height) x column (width) x color (3) with color order BGR

        color order flipped to RGB for detectron2 processing, then back to BGR
        afterwards for openCV visualization
        """
        outputs = self.predictor(image)
        v = Visualizer(image[:,:,::-1])
        img_out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return img_out.get_image()[:,:,::-1]

    def dump_cfg(self, path) -> None:
        """ saves model config to specified path """
        out = "cat_frcnn_r50_3x.yaml"
        out_path = path_join(path, out)
        with open(out_path, "w") as f:
            print (self.cfg.dump(), file=f)

class CustomTrainer(DefaultTrainer):
    """
    method to  ensure model validates against validation set
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            path = path_join(cfg.OUTPUT_DIR, "evals")
            makedirs(path, exist_ok=True)
            output_folder = path
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def extract_tar(filepath: str, out_path: str) -> None:
    """ extracts files from compressed tarfile """
    with tarfile.open(filepath, "r:gz") as f:
        f.extractall(out_path)

def split_annotations(xml_path: str) -> Tuple:
    """ divides annotation files into train (80%) and val (20%) groups """
    files = set(f.split(".")[0] for f in listdir(xml_path))
    train_len = math.floor( len(files) * 0.8 )
    val_len = len(files) - train_len

    train = set(random.sample(files, train_len))
    val = files - train
    return (train, val)

def split_move(xml_path: str, img_path: str, files: Set[str], out_path: str) -> None:
    """ moves split annotations + images to associated folder structure

    example:
    ./coco/train/imgs/
    ./coco/train/annos/
    """
    img_dir = path_join(out_path, "imgs")
    anno_dir = path_join(out_path, "annos")
    makedirs(img_dir, exist_ok=True)
    makedirs(anno_dir, exist_ok=True)

    for f in files:
        shutil.copy(path_join(xml_path, f + ".xml"), anno_dir)
        shutil.copy(path_join(img_path, f + ".jpg"), img_dir)

def anno2coco(xml_path: str, img_path: str, out_path: str) -> None:
    """ generates annotations file in coco json format """
    json_dict = defaultdict(list)

    for cls in ["cat"]:
        cat_d = {"id" : len(json_dict["categories"]) + 1,
                 "name" : cls,
                 "supercategory" : "animal"}
        json_dict["categories"].append(cat_d)

    for f in listdir(xml_path):
        cmn = f.split(".")[0]
        img_file = cmn + ".jpg"

        # clunky XML field parsing for known format
        root = ET.parse(path_join(xml_path, f)).getroot()

        for seg in root.findall("size"):
            width  = int(seg.find("width").text)
            height = int(seg.find("height").text)

        for seg in root.findall("object"):
            species = seg.find("name").text
            xmin = int(seg.find("bndbox/xmin").text)
            ymin = int(seg.find("bndbox/ymin").text)
            xmax = int(seg.find("bndbox/xmax").text)
            ymax = int(seg.find("bndbox/ymax").text)

        iid = len(json_dict["images"])
        cat_id = 1 if species == "cat" else 0
        box_w = xmax - xmin
        box_h = ymax - ymin

        img_d = {"id"        : iid,
                 "file_name" : img_file,
                 "height"    : height,
                 "width"     : width}

        anno_d = {"id"           : iid,
                  "image_id"     : iid,
                  "category_id"  : cat_id,
                  "bbox"         : [xmin, ymin, box_w, box_h],
                  "area"         : box_w * box_h,
                  "segmentation" : [],
                  "iscrowd"      : 0}

        json_dict["images"].append(img_d)
        if cat_id: json_dict["annotations"].append(anno_d)

    split = basename(out_path)
    out_file = path_join(out_path, split + ".json")
    with open(out_file, "w") as f:
        print (json.dumps(dict(json_dict)), file=f)

def test_visuals(out_path: str, split_name: str) -> None:
    """ selects sample images and visualizes annotations """
    data = DatasetCatalog.get(split_name)
    meta = MetadataCatalog.get(split_name)

    for dat in random.sample(data, 5):
        iid = dat["file_name"]
        print (iid)
        image = cv2.imread(iid)
        v = Visualizer(image[:,:,::-1], metadata=meta)
        out = v.draw_dataset_dict(dat)
        out_file = path_join(out_path, basename(iid))
        print ("writing", out_file)
        cv2.imwrite(out_file, out.get_image()[:,:,::-1])

if (__name__ == '__main__'):
    # folder locations and archive names
    base = "./"
    temp = path_join(base, "temp")
    makedirs(temp, exist_ok=True)

    arch_annos = "annotations.tar.gz"
    arch_imgs  = "images.tar.gz"

    # extract compressed archive if not previously done
    for arch in [arch_annos, arch_imgs]:
        folder = arch.split(".")[0]
        if not isdir(path_join(temp, folder)):
            print("extracting compressed archive: {}".format(arch))
            extract_tar(path_join(base, arch), temp)

    # split annotations into train / val sets, and move files
    annos = path_join(temp, "annotations/xmls")
    imgs  = path_join(temp, "images")

    coco = path_join(base, "coco")
    train = path_join(coco, "train")
    val = path_join(coco, "val")

    if not isdir(coco):
        print ("copying files to coco folder ...")
        set_t, set_v = split_annotations(annos)
        split_move(annos, imgs, set_t, train)
        split_move(annos, imgs, set_v, val)

    # load datasets into detectron2 registry
    DatasetCatalog.clear()
    MetadataCatalog.clear()

    for dat in [train, val]:
        files = listdir(path_join(dat, "imgs"))
        samples = len(files)
        cat_cnt = sum(f[0].isupper() for f in files)
        print (dat, "({}/{}) cats".format(cat_cnt, samples))

        split = basename(dat)
        anno2coco(path_join(dat, "annos"), path_join(dat, "imgs"), dat)
        register_coco_instances(split, {}, path_join(dat, split + ".json"),
                                path_join(dat, "imgs"))

        # visualize random samples to confirm data loading correctly
        #test_visuals(temp, split)

    # train model on dataset if not previously done
    data = DatasetCatalog.get("train")
    model = ModelClass()
    model.cfg_setup(len(data), 1)
    model.dump_cfg("./model_backup")
    #model.learn()

    # perform inference with the trained model
    model = ModelClass()
    model.cfg_load("./model_backup/cat_frcnn_r50_3x.yaml",
                   "./model_backup/model_final_shrunk.pth")
    test = "./temp/images/Ragdoll_210.jpg"
    image = cv2.imread(test)
    res = model.infer(image)
    cv2.imwrite("./test_img.jpg", res)
    model.dump_cfg("./output")

    print ("done")
