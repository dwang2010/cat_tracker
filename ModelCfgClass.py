import os

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultTrainer, DefaultPredictor

class CustomTrainer(DefaultTrainer):
    """
    method to  ensure model validates against validation set
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            path = os.path.join(cfg.OUTPUT_DIR, "evals")
            os.makedirs(path, exist_ok=True)
            output_folder = path
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

class ModelCfgClass:
    """
    methods for detectron2 related model configuration / training / inference
    """

    def __init__(self) -> None:
        """ init ModelClass object with null / default attributes """
        self.cfg = None
        self.trainer = None
        self.predictor = None

    def cfg_init(self,
                 num_images: int,
                 num_classes: int,
                 epochs: int = 10) -> None:
        """ setup model configuration for training

        Args:
            num_images: number of images in target dataset
            num_classes: number of distinct object classes in dataset
            epochs: number of times to cycle model through complete dataset
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

        self.cfg.DATASETS.TRAIN = ("train",)
        self.cfg.DATASETS.TEST = ("val",)
        self.cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
        self.cfg.DATALOADER.NUM_WORKERS = 4 # threads

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

        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    def cfg_update_weights(self, path) -> None:
        """ update config weight from file """
        self.cfg.MODEL.WEIGHTS = path

    def cfg_load(self, cfg_path: str, weight_path: str) -> None:
        """ load existing model configuration and weights for inference """
        if not os.path.isfile(cfg_path):
            print ("ERROR! Model file not found: {}".format(cfg_path))
        self.cfg = get_cfg()
        self.cfg.merge_from_file(cfg_path)
        self.cfg_update_weights(weight_path)

    def cfg_dump(self, path) -> None:
        """ saves model config to specified path """
        out = "cat_frcnn_r50_3x.yaml"
        out_path = os.path.join(path, out)
        with open(out_path, "w") as f:
            print (self.cfg.dump(), file=f)

    def learn(self) -> None:
        """ performs model training with loaded dataset """
        self.trainer = CustomTrainer(self.cfg)
        self.trainer.resume_or_load(resume=False)
        self.trainer.train()

    def infer(self, image):
        """ performs inference on provided images

        for color images, OpenCV imread() reads as NumPy array ndarray of row
        (height) x column (width) x color (3) with color order BGR

        color order flipped to RGB for detectron2 processing, then back to BGR
        afterwards for openCV visualization
        """
        self.predictor = DefaultPredictor(self.cfg)
        outputs = self.predictor(image)
        v = Visualizer(image[:,:,::-1])
        img_out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        print (outputs["instances"])
        return img_out.get_image()[:,:,::-1]
