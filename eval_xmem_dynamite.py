#Adapted by Srijeet Roy from: https://github.com/amitrana001/DynaMITe/blob/main/train_net.py

import numpy as np
try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
import itertools
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import json
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

from metrics.summary import summarize_results,summarize_round_results
import copy
import gc

_root = "/globalwork/roy/dynamite_video/xmem_dynamite/XMem_DynaMITe/datasets/"
_DATASET_PATH = {
    "davis_2017_val": {
        "annotations": "DAVIS/DAVIS-2017-trainval/Annotations/480p",
        "images": "DAVIS/DAVIS-2017-trainval/JPEGImages/480p",
        "sets": "DAVIS/DAVIS-2017-trainval/ImageSets/2017/val.txt",
    },
    "mose_val": {
        "annotations": "MOSE/valid/Annotations",
        "images":"MOSE/valid/JPEGImages",
        "sets":"",
    },
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
    def interactive_evaluation(cls, cfg, dynamite_model, interactions, iou, 
                                all_images, all_gt_masks, dataloader_dict, 
                                args=None, xmem_config=None):
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
            debug = args.debug
        
        if not isinstance(iou_threshold, list):                
            iou_threshold = [iou_threshold]
        if not isinstance(max_interactions, list):                
            max_interactions = [max_interactions]

        for dataset_name in eval_datasets:

            if dataset_name in ["davis_2017_val","mose_val","sbd_multi_insts","coco_2017_val"]:
                print(f'[INFO] Initiating Multi-Instance Evaluation on {eval_datasets}...')
                
                if eval_strategy in ["random", "best", "worst"]:
                    if dataset_name != "mose_val":
                        from dynamite.inference.multi_instance.random_best_worst import evaluate
                    else:
                        from dynamite.inference.multi_instance.random_best_worst_mose import evaluate
                elif eval_strategy == "max_dt":
                    from dynamite.inference.multi_instance.max_dt import evaluate
                elif eval_strategy == "wlb":
                    from dynamite.inference.multi_instance.wlb import evaluate
                elif eval_strategy == "round_robin":
                    from dynamite.inference.multi_instance.round_robin import evaluate
                
                print(f'[INFO] Loaded Evaluation routine following {eval_strategy} evaluation strategy!')                                

                print(f'Interactions: {interactions}')
                print(f'IoU threshold: {iou}')
                save_path = os.path.join(vis_path, f'{interactions}_interactions/iou_{int(iou*100)}')
                #save_path = vis_path
                os.makedirs(save_path, exist_ok=True) 
                expt_path = os.path.join(save_path, 'xmem_propagation')
                os.makedirs(expt_path, exist_ok=True)

                print(f'[INFO] Starting evaluation...')
                vis_path_vis = os.path.join(save_path, 'vis')
                os.makedirs(vis_path_vis, exist_ok=True)
                results_i, progress_report = evaluate(dynamite_model,
                                    xmem_config,
                                    dataloader_dict, all_images, all_gt_masks,
                                    iou_threshold = iou,
                                    max_interactions = interactions,
                                    eval_strategy = eval_strategy,
                                    seed_id=seed_id,
                                    vis_path=vis_path_vis,
                                    max_rounds=max_rounds,
                                    dataset_name=dataset_name,
                                    save_masks=True,
                                    expt_path=expt_path)
                
                print(f'[INFO] Evaluation complete for dataset {dataset_name}: IoU threshold={iou}, Interaction budget={interactions}!')

                with open(os.path.join(save_path,f'results_{interactions}_interactions_iou_{int(iou*100)}.json'), 'w') as f:
                    json.dump(results_i, f)
                with open(os.path.join(save_path,f'progress_{interactions}_interactions_iou_{int(iou*100)}.json'), 'w') as f:
                    json.dump(progress_report, f)
                
                if dataset_name != "mose_val":
                    summary, df = summarize_results(results_i)
                    df.to_csv(os.path.join(save_path, f'round_results_{interactions}_interactions_iou_{int(iou*100)}.csv'))
                    with open(os.path.join(save_path,f'summary_{interactions}_interactions_iou_{int(iou*100)}.json'), 'w') as f:
                        json.dump(summary, f)
                    
                    summary_df = summarize_round_results(df, iou)
                    summary_df.to_csv(os.path.join(save_path, f'round_summary_{interactions}_interactions_iou_{int(iou*100)}.csv'))
                del results_i

                
def load_images(dataset_name="davis_2017_val", debug_mode=False):
    image_path = os.path.join(_root,_DATASET_PATH[dataset_name]["images"])
    if dataset_name=="mose_val":
        seqs = sorted([f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path,f))])
    else:
        val_set = os.path.join(_root,_DATASET_PATH[dataset_name]["sets"])
        with open(val_set, 'r') as f:
            seqs = [line.rstrip('\n') for line in f.readlines()]
    all_images = {}    
    transform = transforms.Compose([transforms.ToTensor()])
    for s in seqs:
        seq_images = []
        seq_path = os.path.join(image_path, s)
        imagefiles = sorted([f for f in os.listdir(seq_path)])
        for file in imagefiles:
            if file.endswith('.jpg') or file.endswith('.png'):
                im = Image.open(os.path.join(seq_path, file))
                im = transform(im)
                seq_images.append(im)
        seq_images = torch.stack(seq_images)
        all_images[s] = seq_images
        if debug_mode:
            break
    return all_images

def load_gt_masks(dataset_name="davis_2017_val", debug_mode=False):
    mask_path = os.path.join(_root,_DATASET_PATH[dataset_name]["annotations"])
    if dataset_name=="mose_val":
        seqs = sorted([f for f in os.listdir(mask_path) if os.path.isdir(os.path.join(mask_path,f))])
    else:
        val_set = os.path.join(_root,_DATASET_PATH[dataset_name]["sets"])
        with open(val_set, 'r') as f:
            seqs = [line.rstrip('\n') for line in f.readlines()]
    all_gt_masks = {}
    for s in seqs:
        seq_images = []
        seq_path = os.path.join(mask_path, s)
        maskfiles = sorted([f for f in os.listdir(seq_path)])
        for file in maskfiles:
            if file.endswith('.jpg') or file.endswith('.png'):
                im = np.asarray(Image.open(os.path.join(seq_path, file)))
                seq_images.append(im)
        seq_images = np.asarray(seq_images)
        all_gt_masks[s] = seq_images
        if debug_mode:
            break
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

    dataset_name = args.eval_datasets[0]
    # load data
    print(f'[INFO] Loading all ground truth masks from the disc...')
    all_gt_masks = load_gt_masks(dataset_name, args.debug)
    if dataset_name != "mose_val":
        print(f'[INFO] Loading all frames from the disc...')
        all_images = load_images(dataset_name, args.debug)                
        assert len(all_images) == len(all_gt_masks)
        print(f'[INFO] Loaded {len(all_images)} sequences.')
    else:
        all_images = {}
    
    print(f'[INFO] Loading test data loader from {dataset_name}...')
    data_loader = Trainer.build_test_loader(cfg, dataset_name)
    print(f'[INFO] Data loader  preparation complete! length: {len(data_loader)}')
    dataloader_dict = defaultdict(list)
    print(f'[INFO] Iterating through the Data Loader...')
    # iterate through the data_loader, one image at a time
    for idx, inputs in enumerate(data_loader):                     
        curr_seq_name = inputs[0]["file_name"].split('/')[-2]
        if args.debug and curr_seq_name != list(all_images.keys())[0]:
            break
        dataloader_dict[curr_seq_name].append([idx, inputs])
    del data_loader

    for interactions, iou in list(itertools.product(args.max_interactions,args.iou_threshold)):
        dataloader_dict_copy = copy.deepcopy(dataloader_dict)
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
            xmem_config['generic_path'] = None                  # for generic eval - a folder that contains "JPEGImages" and "Annotations"
            xmem_config['output'] = None               
            xmem_config['save_all'] = True                     # save all frames - useful only in YouTubeVOS/long-time video
            xmem_config['benchmark'] = False                    # enable to disable amp for FPS benchmarking
            # long-term memory options
            xmem_config['disable_long_term'] = False
            xmem_config['max_mid_term_frames']= 10              # T_max in paper, decrease to save memory 
            xmem_config['min_mid_term_frames']= 5               # T_min in paper, decrease to save memory
            xmem_config['max_long_term_elements']= 10000        # LT_max in paper, increase if objects disappear for a long time (default 10000)
            xmem_config['num_prototypes']= 128                  # P in paper (default 128)
            xmem_config['top_k']= 30
            xmem_config['mem_every']= 5                         # r in paper. Increase to improve running speed
            xmem_config['deep_update_every']= -1                # Leave -1 normally to synchronize with mem_every
            # multi-scale options
            xmem_config['save_scores']= False
            xmem_config['flip']= False
            xmem_config['size']= 480                            # Resize the shorter side to this size (default 480). -1 to use original resolution
            xmem_config['enable_long_term'] = not xmem_config['disable_long_term']

            # bidirectional propagation
            xmem_config['cutoff'] = True        # if True, bidirectional propagation cuts off at nearest DynaMITe-refined frames

            res = Trainer.interactive_evaluation(cfg, dynamite_model, 
                                                interactions, iou, 
                                                all_images, all_gt_masks, dataloader_dict_copy,
                                                args, xmem_config)

            #return res
            print(f'Finished experiment: {interactions} interactions, at IoU threshold {iou}')
            del dynamite_model, res, dataloader_dict_copy
            torch.cuda.empty_cache()
            gc.collect()

    else:
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