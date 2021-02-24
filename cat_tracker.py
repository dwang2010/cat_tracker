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

from os.path import basename, isdir
from os.path import join as path_join

import os, random, subprocess, tarfile, json, shutil, math
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set

# extract files from compressed tarfile
def extract_tar(filepath: str, out_path: str) -> None:
    with tarfile.open(filepath, "r:gz") as f:
        f.extractall(out_path)

# randomly divide annotation files into train (80%) and val (20%) groups
def split_annotations(xml_path: str) -> Tuple:
    files = set(f.split(".")[0] for f in os.listdir(xml_path))
    train_len = math.floor( len(files) * 0.8 )
    val_len = len(files) - train_len

    train = set(random.sample(files, train_len))
    val = files - train
    return (train, val)

# move annotations + images to target folder, per structure:
# ./coco/train/imgs/
#             /annos/
#             /train_coco.json
def split_move(xml_path: str, img_path: str, files: Set[str], out_path: str) -> None:
    img_dir = path_join(out_path, "imgs")
    anno_dir = path_join(out_path, "annos")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(anno_dir, exist_ok=True)

    for f in files:
        shutil.copy(path_join(xml_path, f + ".xml"), anno_dir)
        shutil.copy(path_join(img_path, f + ".jpg"), img_dir)

def anno2coco(xml_path: str, img_path: str, out_path: str) -> None:
    json_dict = defaultdict(list)

    for cls in ["cat", "dog"]:
        cat_d = {"id" : len(json_dict["categories"]) + 1,
                 "name" : cls,
                 "supercategory" : "animal"}
        json_dict["categories"].append(cat_d)

    for f in os.listdir(xml_path):
        cmn = f.split(".")[0]
        img_file = path_join(img_path, cmn + ".jpg")

        # clunky XML field parsing for known format
        root = ET.parse(path_join(xml_path, f)).getroot()

        for seg in root.findall("size"):
            width = seg.find("width").text
            height = seg.find("height").text

        for seg in root.findall("object"):
            species = seg.find("name").text
            xmin = int(seg.find("bndbox/xmin").text)
            ymin = int(seg.find("bndbox/ymin").text)
            xmax = int(seg.find("bndbox/xmax").text)
            ymax = int(seg.find("bndbox/ymax").text)

        iid = len(json_dict["images"])
        cat_id = 1 if species == "cat" else 2
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
        json_dict["annotations"].append(anno_d)

    split = basename(out_path)
    out_file = path_join(out_path, split + ".json")
    with open(out_file, "w") as f:
        print (json.dumps(dict(json_dict)), file=f)

# visualize the annotations
def test_visuals(out_path: str, split_name: str) -> None:
    data = DatasetCatalog.get(split_name)
    meta = MetadataCatalog.get(split_name)

    for dat in random.sample(data, 5):
        iid = dat["file_name"]
        print (iid)
        image = cv2.imread(iid)
        v = Visualizer(image[:,:,::-1], metadata=meta)
        out = v.draw_dataset_dict(dat)
        out_file = os.path.join(out_path, os.path.basename(iid))
        print ("writing", out_file)
        cv2.imwrite(out_file, out.get_image()[:,:,::-1])

# class Trainer(DefaultTrainer):
@classmethod
def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    if output_folder is None:
        path = os.path.join(cfg.OUTPUT_DIR, "evals")
        os.makedirs(path, exist_ok=True)
        output_folder = path
    return COCOEvaluator(dataset_name, cfg, False, output_folder)

# create model config and perform basic setup
def setup(num_images: int, last_model: str, num_classes: int = 1, epochs: int = 10):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # retrain from source model or previously trained model
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if not os.path.isfile(last_model):
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    else:
        print("loading previous model from: {}".format(last_model))
        cfg.MODEL.WEIGHTS = last_model

    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.DATALOADER.NUM_WORKERS = 8 # threads

    cfg.SOLVER.IMS_PER_BATCH = 4 # batch size
    cfg.SOLVER.BASE_LR = 0.01

    ims_per_epoch = num_images // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.MAX_ITER = ims_per_epoch * epochs
    print ("max iterations: {}".format(cfg.SOLVER.MAX_ITER))

    # decay learning rate at every epoch (down a decade)
    ttl = cfg.SOLVER.MAX_ITER
    lr_deltas = [i for i in range(ttl//epochs, ttl, ttl//epochs)]
    cfg.SOLVER.STEPS = lr_deltas
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.CHECKPOINT_PERIOD = ims_per_epoch

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    return cfg

# train the model
def go_model(cfg):
    print ("CHECKPOINT: TRAINING START")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    print ("CHECKPOINT: TRAINING COMPLETE")

def go_test(cfg):
    print ("CHECKPOINT: VALIDATION START")
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("val", ("bbox",), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, "val")
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
    print ("CHECKPOINT: VALIDATION COMPLETE")

if (__name__ == '__main__'):
    # folder locations and archive names
    base = "./"
    temp = path_join(base, "temp")
    os.makedirs(temp, exist_ok=True)

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
        files = os.listdir(path_join(dat, "imgs"))
        samples = len(files)
        cat_cnt = sum(f[0].isupper() for f in files)
        print (dat, "({}/{}) cats".format(cat_cnt, samples))

        split = basename(dat)
        anno2coco(path_join(dat, "annos"), path_join(dat, "imgs"), dat)
        register_coco_instance(split, {}, path_join(dat, split + ".json"), path_join(dat + "imgs"))

        # visualize random samples to confirm data loading correctly
        test_visuals(temp, split)

        # train model on dataset if not previously done
        if not os.path.isfile(model_file):
            data = DatasetCatalog.get("train")
            cfg = setup(len(data), None)

            go_model(cfg)

    print ("done")
