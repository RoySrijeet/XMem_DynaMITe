# Copyright (c) Facebook, Inc. and its affiliates.
import logging
logging.basicConfig(level=logging.INFO)
import os
from contextlib import ExitStack, contextmanager
import copy
import numpy as np
import torch
import random
from collections import defaultdict
from torchvision import transforms
from detectron2.utils.comm import get_world_size        # utils.comm -> primitives for multi-gpu communication
from torch import nn
from ..utils.clicker import Clicker
from ..utils.predictor import Predictor
from PIL import Image
from metrics.j_and_f_scores import batched_jaccard,batched_f_measure
from eval import eval_xmem
import gc

import util.xmem_dynamite_helpers as helpers
_DATASET_ROOT = helpers._DATASET_ROOT
_DATASET_PATH = helpers._DATASET_PATH

def evaluate(
    model, xmem_config,
    dataloader_dict, all_images, all_gt_masks,
    iou_threshold = 0.85, max_interactions = 10, sampling_strategy=1,
    eval_strategy = "random", seed_id = 0, vis_path = None, max_rounds=0,
    dataset_name="davis_2017_val", save_masks=False, expt_path=None
):

    # args
    print(f'[EVALUATOR INFO] IoU Threshold: {iou_threshold}')
    print(f'[EVALUATOR INFO] Max Interactions per Frame: {max_interactions}')
    print(f'[EVALUATOR INFO] Max Rounds per Sequence: {max_rounds}')
    print(f'[EVALUATOR INFO] DynaMITe Evaluation Strategy: {eval_strategy}')
    print(f'[EVALUATOR INFO] DynaMITe Point Sampling Strategy: {sampling_strategy}')
    print(f'[EVALUATOR INFO] XMEM CONFIGURATION:\n {xmem_config}')

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(dataloader_dict)))                       # 1999 (davis_2017_val)
    logger.info(f"Using {eval_strategy} evaluation strategy with random seed {seed_id}")    
    
    all_interactions = {}
    all_interactions_per_round = {}
    all_interactions_per_instance = {}
    all_rounds = {}
    all_j_and_f = {}
    all_jaccard = {}
    all_contour = {}
    all_ious = {}
    all_instance_level_iou = {}
    copy_iou_checkpoints = [0.85, 0.90, 0.95, 0.99]
    progress_report = {}

    with ExitStack() as stack:                                           

        if isinstance(model, nn.Module):    
            stack.enter_context(inference_context(model))                           
        stack.enter_context(torch.no_grad())                             
                           
        total_num_interactions = 0                                       # for whole dataset       
        total_num_rounds = 0                                             # for whole dataset

        random.seed(123456+seed_id)

        if dataset_name=="burst_val":
            sequence_list = dataloader_dict
        else:
            sequence_list = list(dataloader_dict.keys())
    
        #### SEQUENCE-LEVEL LOOP ####
        print(f'[EVALUATOR INFO] Sequence-wise evaluation...')
        for seq in sequence_list:
            print(f'\n[SEQUENCE INFO] Sequence: {seq}')
            if vis_path:
                vis_path_seq = os.path.join(vis_path, seq)
            xmem_config['generic_path'] = os.path.join(expt_path,'Annotations')  
            os.makedirs(os.path.join(expt_path,'Annotations'), exist_ok=True)          
            xmem_config['output'] = os.path.join(expt_path,'Annotations',seq)
            os.makedirs(os.path.join(expt_path,'Annotations',seq), exist_ok=True)
            
            if dataset_name=="burst_val":
                dataloader_dict = helpers.burst_video_loader(seq)
            
            if dataset_name not in ["mose_val", "burst_val"]:
                #all_frames = all_images[seq]                            # collect all image frames in the sequence
                all_masks = all_gt_masks[seq]                
            else:
                #all_frames = helpers.load_sequence_images(seq, dataset_name)
                all_masks = helpers.load_sequence_masks(seq, dataset_name)
            num_frames = len(all_masks)
            
            # Initialize propagation module - once per-sequence
            seq_object_ids = set(np.unique(all_gt_masks[seq][0]))       # object ids in the sequence
            seq_num_instances = len(seq_object_ids) - 1                 # remove bg
            #num_frames = len(all_frames)                                # sequence length
            #all_frames = all_frames.unsqueeze(0).float()                            

            # counters and trackers
            lowest_frame_index = 0                                                   # frame with lowest IoU after propagation - initially, first frame
            round_num = 0                                                            # counter for number of propagation rounds
            clicker_dict = {}                                                        # records clickers for interacted frames
            predictor_dict = {}                                                      # records predictors for interacted frames

            num_interactions_for_sequence = [0]*num_frames                           # records #interactions on each frame of the sequence
            out_masks = None                                                         # stores predicted masks by prop module, initially None
            all_interactions_per_round[seq] = []                                     # records round-wise interaction details
            all_j_and_f[seq] = [0]                                                   # records J&F metric score for the sequence
            seq_avg_iou = 0                                                          # records average IoU for the sequence
            seq_avg_jf = 0                                                           # records average J&F for the sequence
            iou_checkpoints = copy.deepcopy(copy_iou_checkpoints)                    # IoU checkpoints
            progress_report[seq] = {}
            
            instance_level_iou = [[]] * num_frames
            dynamite_preds = []
            
            #### ROUND LOOP ####
            while lowest_frame_index!=-1:
                dynamite_preds.append(lowest_frame_index)
                round_num += 1      # round start
                loop = 0
                if vis_path:
                    vis_path_round = os.path.join(vis_path_seq, str(round_num))                                

                print(f'\n\n[DynaMITe INFO][SEQ:{seq}][ROUND:{round_num}] DynaMITe refining frame {lowest_frame_index}...')
                idx, inputs = dataloader_dict[seq][lowest_frame_index]                              # load frame with lowest IoU

                # (re)load Clicker and Predictor (DynaMITe)
                if num_interactions_for_sequence[lowest_frame_index] == 0:                           # if frame has been previously interacted with 
                    clicker = Clicker(inputs, sampling_strategy, dataset_name)
                    predictor = Predictor(model)
                    repeat = False
                else:                                                  
                    clicker = clicker_dict[lowest_frame_index]
                    predictor = predictor_dict[lowest_frame_index]
                    repeat = True
                
                
                #check for missing objects
                object_ids = set(np.unique(all_gt_masks[seq][lowest_frame_index]))                  # objects present in the frame                
                num_instances = clicker.num_instances
                missing_obj_ids = None
                if num_instances!=seq_num_instances:                    
                    missing_obj_ids = seq_object_ids - object_ids
                
                # if available, pred_masks from prev round is set to Clicker
                if out_masks is not None:                                                           # out_masks = [num_frames, H, W]
                    mask_H,mask_W = out_masks[lowest_frame_index].shape
                    pred_masks = np.zeros((seq_num_instances,mask_H,mask_W))                        # pred_masks = [num_instances, H, W]
                    for i in range(seq_num_instances):
                        if missing_obj_ids is not None:                                             # take into account missing objects
                            if i+1 in missing_obj_ids:
                                continue
                        pred_masks[i][np.where(out_masks[lowest_frame_index]==i+1)] = 1       
                    pred_masks = torch.from_numpy(pred_masks)
                    clicker.set_pred_masks(pred_masks)   
                # otherwise, predict segmentation mask with DynaMITe
                else:
                    pred_masks = predictor.get_prediction(clicker)
                    clicker.set_pred_masks(pred_masks)
                # instance-wise IoU
                ious = clicker.compute_iou()
                instance_level_iou[lowest_frame_index] = [iou.tolist() for iou in ious]
                frame_avg_iou = sum(ious)/len(ious)                
                
                # record the interaction, if the frame has never been interacted with
                if not repeat:
                    total_num_interactions+=(num_instances)                                                       # counter over all dataset
                    num_interactions_for_sequence[lowest_frame_index] += num_instances                                  

                    # round,loop,frame_idx,obj_idx,#interactions,frame_iou,seq_iou,seq_jf
                    all_interactions_per_round[seq].append([round_num, loop, 
                                                           lowest_frame_index, [i-1 for i in list(map(int,object_ids - {0}))], 
                                                           num_interactions_for_sequence[lowest_frame_index],
                                                           frame_avg_iou.item(),
                                                           seq_avg_iou, seq_avg_jf])
                if vis_path:
                   clicker.save_visualization(vis_path_round, ious=ious, 
                                            num_interactions=num_interactions_for_sequence[lowest_frame_index], 
                                            round_num=round_num, save_masks=save_masks)             
                
                # interaction limit
                max_iters_for_image = max_interactions * num_instances +1
                point_sampled = True
                random_indexes = list(range(len(ious)))

                #### INTERACTIVE REFINEMENT LOOP ####        
                while (num_interactions_for_sequence[lowest_frame_index]<max_iters_for_image):               # 1st stopping criterion - if interaction budget is over
                    
                    while all(iou >= iou_checkpoints[0] for iou in ious):                                    # IoU checkpoints
                        t = iou_checkpoints.pop(0)
                        print(f'[DynaMITe INFO][SEQ:{seq}][ROUND:{round_num}] Frame {lowest_frame_index} reached IoU Checkpoint {t} after {num_interactions_for_sequence[lowest_frame_index]} interactions.')
                    
                    if all(iou >= iou_threshold for iou in ious):                                             # 2nd stopping criterion - if IoU threshold is reached
                        print(f'[DynaMITe INFO][SEQ:{seq}][ROUND:{round_num}] Frame {lowest_frame_index} meets IoU Threshold {iou_threshold} after {num_interactions_for_sequence[lowest_frame_index]} interactions!')
                        break

                    loop += 1             
                    
                    # According to eval strategy, select which instance to interact with
                    if eval_strategy == "worst":
                        indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=False).indices
                    elif eval_strategy == "best":                    
                        indexes = torch.topk(torch.tensor(ious), k = len(ious),largest=True).indices
                    elif eval_strategy == "random":
                        random.shuffle(random_indexes)
                        indexes = random_indexes
                    else:
                        assert eval_strategy in ["worst", "best", "random"]

                    # DynaMITe Interaction
                    point_sampled = False
                    for i in indexes:                        
                        if ious[i]<iou_threshold:                                                                
                            obj_index = clicker.get_next_click(refine_obj_index=i, time_step=num_interactions_for_sequence[lowest_frame_index])
                            # obj_index: -1 for bg, 0,1 and so on for instances
                            point_sampled = True
                            break
                    if point_sampled:
                        print(f'DynaMITe added a click on {obj_index}!')
                        if obj_index == -1:
                            print(f'This is a background click to refine the mask of object {i}')
                        total_num_interactions+=1
                        num_interactions_for_sequence[lowest_frame_index] += 1
                        pred_masks = predictor.get_prediction(clicker)
                        clicker.set_pred_masks(pred_masks)
                        ious = clicker.compute_iou()
                        if vis_path:
                            clicker.save_visualization(vis_path_round, ious=ious, 
                                                    num_interactions=num_interactions_for_sequence[lowest_frame_index], 
                                                    round_num=round_num, save_masks=save_masks)
                        
                        frame_avg_iou = sum(ious)/len(ious)
                        instance_level_iou[lowest_frame_index] = [iou.tolist() for iou in ious]
                        all_interactions_per_round[seq].append([round_num, loop, lowest_frame_index, int(obj_index), num_interactions_for_sequence[lowest_frame_index], frame_avg_iou.item(), seq_avg_iou, seq_avg_jf])
                
                clicker.save_visualization(vis_path_round, ious=ious, 
                                        num_interactions=num_interactions_for_sequence[lowest_frame_index], 
                                        round_num=round_num, save_masks=True, expt_path=xmem_config['output'], seq_name=seq)

                # store clicker and predictor, in case this frame needs to be interacted with again
                clicker_dict[lowest_frame_index] = clicker
                predictor_dict[lowest_frame_index] = predictor                            
                
                # XMEM
                out_masks = eval_xmem(xmem_config, seq, all_gt_masks[seq], dynamite_preds)            

                if dataset_name in ['mose_val', 'burst_val']:
                    lowest_frame_index = -1 #stop round iter
                    break
                else:
                    # metrics (mean: over instances in a frame)
                    jaccard_mean, jaccard_instances = batched_jaccard(all_gt_masks[seq], out_masks, average_over_objects=True, nb_objects=seq_num_instances)
                    contour_mean, contour_instances = batched_f_measure(all_gt_masks[seq], out_masks, average_over_objects=True, nb_objects=seq_num_instances)

                    j_and_f = 0.5*jaccard_mean + 0.5*contour_mean
                    j_and_f = j_and_f.tolist()
                    seq_avg_jf = sum(j_and_f)/len(j_and_f)

                    iou_for_sequence = jaccard_mean.tolist()
                    seq_avg_iou = sum(iou_for_sequence)/len(iou_for_sequence)
                    print(f'[PROPAGATION INFO][SEQ:{seq}][ROUND:{round_num}] Prediction results: Average IoU: {seq_avg_iou}, Average J&F: {seq_avg_jf}')                                        
                    
                    all_interactions_per_round[seq].append([round_num, '-', '-', '-', sum(num_interactions_for_sequence), '-', seq_avg_iou, seq_avg_jf])
                    progress_report[seq][round_num] = {'J_AND_F': seq_avg_jf, 'J': seq_avg_iou, 'J_AND_F_FRAME': j_and_f, 'J_FRAME': iou_for_sequence}
                    
                    # Check stopping criteria
                    frame_list = [i for i in range(num_frames)]
                    while True:
                        min_iou_index = np.unravel_index(np.argmin(jaccard_instances, axis=None), jaccard_instances.shape)
                        min_iou = jaccard_instances[min_iou_index]
                        print(f'[EVALUATOR INFO][SEQ:{seq}][ROUND:{round_num}] Weakest frame (instance): idx: {min_iou_index}, value: {min_iou}')
                        if min_iou < iou_threshold:                                                         # 1. whether all frames meet IoU threshold
                            if round_num == max_rounds:                                                     # 2. whether round budget is over
                                print(f'[STOPPING CRITERIA][SEQ:{seq}][ROUND:{round_num}] Maximum round limit ({max_rounds}) reached!')
                                lowest_frame_index = -1
                                break
                            lowest_frame_index = int(min_iou_index[0])
                            print(f'[EVALUATOR INFO][SEQ:{seq}][ROUND:{round_num}] Next index to refine: {lowest_frame_index}, IoU: {min_iou}')
                            if num_interactions_for_sequence[lowest_frame_index] >= max_iters_for_image:     # if interaction budget is over for a frame, look for another frame             
                                print(f'[STOPPING CRITERIA][SEQ:{seq}][ROUND:{round_num}] Budget over - skipping frame {lowest_frame_index}.')
                                frame_list.remove(lowest_frame_index)
                                jaccard_instances[min_iou_index[0]] = 99.
                                if len(frame_list)==0:                                                         # 3. whether interaction budget is over for all frames
                                    lowest_frame_index = -1
                                    print(f'[STOPPING CRITERIA][SEQ:{seq}][ROUND:{round_num}] Ran out of click budget for all frames!')
                                    break
                            else:
                                break                        
                        else:
                            lowest_frame_index = -1
                            print(f'[STOPPING CRITERIA][SEQ:{seq}][ROUND:{round_num}] All frames meet IoU requirement!')
                            break                                        
                         

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

            total_num_rounds += round_num

            # store #interactions for entire sequence
            all_interactions[seq] = num_interactions_for_sequence
            all_instance_level_iou[seq] = instance_level_iou
            all_rounds[seq] = round_num

            if dataset_name not in ['mose_val', 'burst_val']: 
                # store metrics for entire sequence
                all_j_and_f[seq] = j_and_f                                                       
                all_jaccard[seq] = jaccard_mean.tolist()
                all_contour[seq] = contour_mean.tolist()
                all_ious[seq] = iou_for_sequence 
                del iou_for_sequence, jaccard_instances, jaccard_mean, contour_instances, contour_mean
            
            del clicker_dict, predictor_dict
            del all_masks, num_interactions_for_sequence            
            gc.collect()

    results = {
                'iou_threshold': iou_threshold,
                'max_interactions': max_interactions,
                'max_rounds': max_rounds,
                'iou_checkpoints': copy_iou_checkpoints,
                
                'total_num_interactions': [total_num_interactions],
                'all_interactions': all_interactions,
                'all_interactions_per_instance': all_interactions_per_instance,
                
                'total_num_rounds': [total_num_rounds],
                'all_rounds': all_rounds,
                'all_interactions_per_round': all_interactions_per_round,

                'all_instance_level_iou': all_instance_level_iou,
                'all_j_and_f' : all_j_and_f,
                'all_jaccard' : all_jaccard,
                'all_contour' : all_contour,
                'all_ious': all_ious,
    }

    return results, progress_report


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.
    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)