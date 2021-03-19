from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

from ModelCfgClass import ModelCfgClass

import cv2

from os import listdir, makedirs
from os.path import basename, isdir, isfile
from os.path import join as path_join

import os, random, subprocess, tarfile, json, shutil, math
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set

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
    backup = path_join(base, "model_backup")
    out = path_join(base, "output")
    makedirs(temp, exist_ok=True)
    makedirs(backup, exist_ok=True)
    makedirs(out, exist_ok=True)

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
    if not isfile(path_join(backup, "model_final.pth")):
        data = DatasetCatalog.get("train")
        model = ModelCfgClass()
        model.cfg_init(len(data), 1)
        model.learn()
        model.cfg_dump(backup)

    # perform inference with the trained model
    model = ModelCfgClass()
    model.cfg_load(path_join(backup, "cat_frcnn_r50_3x.yaml"),
                   path_join(out, "model_final.pth"))
    test = "./temp/images/Ragdoll_210.jpg"
    image = cv2.imread(test)
    res = model.infer(image)
    cv2.imwrite("./test_img.jpg", res)

    print ("done")
