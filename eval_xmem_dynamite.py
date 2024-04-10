#Adapted by Srijeet Roy from: https://github.com/amitrana001/DynaMITe/blob/main/train_net.py

import csv
import numpy as np
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools
import logging

from typing import Any, Dict, List, Set

import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_setup,
    launch,
)
from dynamite.utils.misc import default_argument_parser

# from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from dynamite import (
    COCOLVISDatasetMapper, EvaluationDatasetMapper
)

from dynamite import (
    add_maskformer2_config,
    add_hrnet_config
)

from dynamite.inference.utils.eval_utils import log_single_instance, log_multi_instance
# MiVOS
# from model.propagation.prop_net import PropagationNetwork
# from model.fusion_net import FusionNet

# XMem
from model.network import XMem
from inference.inference_core import InferenceCore
from inference.data.test_datasets import DAVISTestDataset
from inference.data.mask_mapper import MaskMapper

import os
from collections import defaultdict
import pandas as pd
from PIL import Image
from torchvision import transforms
import json
from metrics.summary import summarize_results,summarize_round_results


_root = "/globalwork/roy/dynamite_video/xmem_dynamite/XMem_DynaMITe/datasets/"
_DATASET_PATH = {
    "davis_2017_val": {
        "annotations": "DAVIS/DAVIS-2017-trainval/Annotations/480p",
        "images": "DAVIS/DAVIS-2017-trainval/JPEGImages/480p",
        "sets": "DAVIS/DAVIS-2017-trainval/ImageSets/2017/val.txt",
    },
    "mose": {},
    "kitti_mots_val": {},
}

class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to Mask2Former.
    """

    @classmethod
    def build_test_loader(cls,cfg,dataset_name):
        mapper = EvaluationDatasetMapper(cfg,False,dataset_name)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)        # d2 call
    
    @classmethod
    def interactive_evaluation(cls, cfg, dynamite_model,xmem_model, args=None, xmem_config=None):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.
        """
        print('[INFO] Interactive Evaluation started...')
        if not args:
            return 

        logger = logging.getLogger(__name__)

        if args and args.eval_only:
            eval_datasets = args.eval_datasets      
            vis_path = args.vis_path                
            eval_strategy = args.eval_strategy      
            seed_id = args.seed_id
            iou_threshold = args.iou_threshold
            max_interactions = args.max_interactions
            max_rounds = args.max_rounds
            save_masks = args.save_masks
        
        if not isinstance(iou_threshold, list):                
            iou_threshold = [iou_threshold]
        if not isinstance(max_interactions, list):                
            max_interactions = [max_interactions]

        for dataset_name in eval_datasets:

            if dataset_name in ["davis_2017_val","mose","sbd_multi_insts","coco_2017_val"]:
                print(f'[INFO] Initiating Multi-Instance Evaluation on {eval_datasets}...')
                
                if eval_strategy in ["random", "best", "worst"]:
                    from dynamite.inference.multi_instance.random_best_worst import evaluate
                elif eval_strategy == "max_dt":
                    from dynamite.inference.multi_instance.max_dt import evaluate
                elif eval_strategy == "wlb":
                    from dynamite.inference.multi_instance.wlb import evaluate
                elif eval_strategy == "round_robin":
                    from dynamite.inference.multi_instance.round_robin import evaluate
                
                print(f'[INFO] Loaded Evaluation routine following {eval_strategy} evaluation strategy!')

                print(f'[INFO] Loading all frames from the disc...')
                all_images = load_images(dataset_name)
                print(f'[INFO] Loading all ground truth masks from the disc...')
                all_gt_masks = load_gt_masks(dataset_name)
                
                print(f'[INFO] Loading test data loader from {dataset_name}...')
                data_loader = cls.build_test_loader(cfg, dataset_name)
                print(f'[INFO] Data loader  preparation complete! length: {len(data_loader)}')
                dataloader_dict = defaultdict(list)
                print(f'[INFO] Iterating through the Data Loader...')
                # iterate through the data_loader, one image at a time
                for idx, inputs in enumerate(data_loader):                     
                    curr_seq_name = inputs[0]["file_name"].split('/')[-2]
                    dataloader_dict[curr_seq_name].append([idx, inputs])

                for interactions, iou in list(itertools.product(max_interactions,iou_threshold)):
                    save_path = os.path.join(vis_path, f'{interactions}_interactions/iou_{int(iou*100)}')
                    #save_path = vis_path
                    os.makedirs(save_path, exist_ok=True) 

                    print(f'[INFO] Starting evaluation...')
                    vis_path_vis = os.path.join(save_path, 'vis')
                    os.makedirs(vis_path_vis, exist_ok=True)
                    results_i = evaluate(dynamite_model, xmem_model, 
                                        dataloader_dict, all_images, all_gt_masks,
                                        iou_threshold = iou,
                                        max_interactions = interactions,
                                        eval_strategy = eval_strategy,
                                        seed_id=seed_id,
                                        vis_path=vis_path_vis,
                                        max_rounds=max_rounds,
                                        xmem_config=xmem_config,
                                        dataset_name=dataset_name,
                                        save_masks = save_masks)
                    
                    print(f'[INFO] Evaluation complete for dataset {dataset_name}: IoU threshold={iou}, Interaction budget={interactions}!')

                    with open(os.path.join(save_path,f'results_{interactions}_interactions_iou_{int(iou*100)}.json'), 'w') as f:
                        json.dump(results_i, f)
                    
                    summary, df = summarize_results(results_i)
                    df.to_csv(os.path.join(save_path, f'round_results_{interactions}_interactions_iou_{int(iou*100)}.csv'))
                    with open(os.path.join(save_path,f'summary_{interactions}_interactions_iou_{int(iou*100)}.json'), 'w') as f:
                        json.dump(summary, f)
                    
                    summary_df = summarize_round_results(df, iou_threshold)
                    summary_df.to_csv(os.path.join(save_path, f'round_summary_{interactions}_interactions_iou_{int(iou*100)}.csv'))

def load_images(dataset_name="davis_2017_val"):
    val_set = os.path.join(_root,_DATASET_PATH[dataset_name]["sets"])

    with open(val_set, 'r') as f:
        seqs = [line.rstrip('\n') for line in f.readlines()]
    all_images = {}
    
    image_path = os.path.join(_root,_DATASET_PATH[dataset_name]["images"] )
    transform = transforms.Compose([transforms.ToTensor()])
    for s in seqs:
        seq_images = []
        seq_path = os.path.join(image_path, s)
        for file in os.listdir(seq_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                im = Image.open(os.path.join(seq_path, file))
                im = transform(im)
                seq_images.append(im)
        seq_images = torch.stack(seq_images)
        all_images[s] = seq_images
    return all_images

def load_gt_masks(dataset_name):
    val_set = os.path.join(_root,_DATASET_PATH[dataset_name]["sets"])
    with open(val_set, 'r') as f:
        seqs = [line.rstrip('\n') for line in f.readlines()]
    all_gt_masks = {}
    mask_path = os.path.join(_root,_DATASET_PATH[dataset_name]["annotations"])
    for s in seqs:
        seq_images = []
        seq_path = os.path.join(mask_path, s)
        for file in os.listdir(seq_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                im = np.asarray(Image.open(os.path.join(seq_path, file)))
                seq_images.append(im)
        seq_images = np.asarray(seq_images)
        all_gt_masks[s] = seq_images
    return all_gt_masks               

def setup(args):
    """
    Create configs and perform basic setups.
    """
    print('[INFO] Setting up DynaMITE...')
    cfg = get_cfg()
    # for poly lr schedule
    #add_deeplab_config(cfg)
    add_maskformer2_config(cfg)                 
    add_hrnet_config(cfg)
    cfg.merge_from_file(args.config_file)       # path to config file
    cfg.merge_from_list(args.opts)
    cfg.freeze()                                # make cfg (and children) immutable
    default_setup(cfg, args)                    # D2 call
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="dynamite")
    return cfg


def main(args):
    
    cfg = setup(args)       # create configs 
    print('[INFO] Setup complete!')


    # for evaluation
    if args.eval_only:
        print('[INFO] DynaMITExXMem Evaluation!')
        torch.autograd.set_grad_enabled(False)

        print('[INFO] Building model...')
        dynamite_model = Trainer.build_model(cfg)                                                # load model (torch.nn.Module)
        print('[INFO] Loading model weights...')                                        
        DetectionCheckpointer(dynamite_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(           # d2 checkpoint load
             cfg.MODEL.WEIGHTS, resume=args.resume
        )
        print('[INFO] DynaMITe loaded!')

        # XMem args -
        xmem_config={}
        xmem_config['model'] = './saves/XMem.pth'           # pre-trained XMem weights
        xmem_config['d16_path'] = '../DAVIS/2016'           # path to DAVIS 2016
        xmem_config['d17_path'] = '../DAVIS/2017'           # path to DAVIS 2017
        xmem_config['y18_path']='../YouTube2018'            # path to YouTube2018
        xmem_config['y19_path']='../YouTube'                # path to YouTube
        xmem_config['lv_path'] ='../long_video_set'         # path to long_video_set
        xmem_config['generic_path'] = None                  # for generic eval - a folder that contains "JPEGImages" and "Annotations"
        xmem_config['dataset'] = 'D17'                      # D16/D17/Y18/Y19/LV1/LV3/G
        xmem_config['split'] = 'val'                        # val/test
        xmem_config['output'] = args.vis_path               
        xmem_config['save_all'] = False                     # save all frames - useful only in YouTubeVOS/long-time video
        xmem_config['benchmark'] = False                    # enable to disable amp for FPS benchmarking
        # long-term memory options
        xmem_config['disable_long_term'] = False
        xmem_config['max_mid_term_frames']= 10              # T_max in paper, decrease to save memory 
        xmem_config['min_mid_term_frames']= 5               # T_min in paper, decrease to save memory
        xmem_config['max_long_term_elements']= 10000        # LT_max in paper, increase if objects disappear for a long time
        xmem_config['num_prototypes']= 128                  # P in paper
        xmem_config['top_k']= 30
        xmem_config['mem_every']= 5                         # r in paper. Increase to improve running speed
        xmem_config['deep_update_every']= -1                # Leave -1 normally to synchronize with mem_every
        # multi-scale options
        xmem_config['save_scores']= False
        xmem_config['flip']= False
        xmem_config['size']= 480                            # Resize the shorter side to this size. -1 to use original resolution
        xmem_config['enable_long_term'] = not xmem_config['disable_long_term']

        # XMem data prep
        #meta_dataset = DAVISTestDataset(os.path.join(xmem_config['d17_path'], 'trainval'), imset='2017/val.txt', size=xmem_config['size'])
        #meta_loader = meta_dataset.get_datasets()

        # XMem load
        torch.autograd.set_grad_enabled(False)
        path_to_weights = xmem_config['model']
        xmem = XMem(xmem_config, path_to_weights).cuda().eval()
        xmem_weights = torch.load(path_to_weights)
        xmem.load_weights(xmem_weights, init_as_zero_if_needed=True)
        print(f'[INFO] XMem loaded!')

        res = Trainer.interactive_evaluation(cfg,dynamite_model, xmem, args, xmem_config)

        return res

    else:
        # for training
        # trainer = Trainer(cfg)
        # trainer.resume_or_load(resume=args.resume)
        # return trainer.train()
        print(f'[INFO] Training routine... Not Implemented')



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("[INFO] Command Line Args:", args)
    launch(                                                                            
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )