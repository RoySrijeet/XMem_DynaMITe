import os
from os import path
from argparse import ArgumentParser
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from itertools import chain
from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore
from metrics.j_and_f_scores import compute_score

from progressbar import progressbar

try:
    import hickle as hkl
except ImportError:
    print('[XMEM INFO] Failed to import hickle. Fine if not using multi-scale testing.')


def eval_xmem(config, seq, gt_masks=None, dynamite_preds=None):
    """
    Data preparation
    """
    meta_dataset = LongTestDataset(size=config['size'], dataset_name=config["dataset_name"],mask_dir=path.join(config['generic_path']))
    output = config['output']
    torch.autograd.set_grad_enabled(False)
    palette = Image.open('/globalwork/roy/dynamite_video/mivos_dynamite/MiVOS_DynaMITe/datasets/DAVIS/DAVIS-2017-trainval/Annotations/480p/blackswan/00000.png').getpalette()

    # Set up loader
    #meta_loader = meta_dataset.get_datasets()
    meta_loader = meta_dataset.get_datasets(seq_list=[seq])

    # Load our checkpoint
    network = XMem(config, config['model']).cuda().eval()
    if config['model'] is not None:
        model_weights = torch.load(config['model'])
        network.load_weights(model_weights, init_as_zero_if_needed=True)
    else:
        print('[XMEM INFO] No model loaded.')

    total_process_time = 0
    total_frames = 0
    # Start eval
    #for vid_reader in progressbar(meta_loader, max_value=len(meta_dataset), redirect_stdout=True):
    for vid_reader in meta_loader:
        loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=2)
        vid_name = vid_reader.vid_name
        if vid_name != seq:
            continue
        vid_length = len(loader)
        # no need to count usage for LT if the video is not that long anyway
        config['enable_long_term_count_usage'] = (
            config['enable_long_term'] and
            (vid_length
                / (config['max_mid_term_frames']-config['min_mid_term_frames'])
                * config['num_prototypes'])
            >= config['max_long_term_elements']
        )

        mapper = MaskMapper()
        processor = InferenceCore(network, config=config)
        first_mask_loaded = False
        propagated_outputs=[[]]*vid_length
        count = 0
        
        curr = dynamite_preds[-1] 
        dynamite_preds = sorted(dynamite_preds)
        if config['cutoff'] and max(dynamite_preds)!=curr:            
            ceil = dynamite_preds[dynamite_preds.index(curr)+1]
        else:
            ceil = len(vid_reader)
        if config['cutoff'] and min(dynamite_preds)!=curr:
            floor = dynamite_preds[dynamite_preds.index(curr)-1]
        else:
            floor = -1

        #for ti, data in enumerate(loader):
        for ti in chain(range(curr, ceil), range(curr-1,floor,-1)):
            with torch.cuda.amp.autocast(enabled=not config['benchmark']):
                data = vid_reader[ti]
                rgb = data['rgb'].cuda()#[0]
                msk = data.get('mask')                
                if msk is not None:
                    count += 1       
                    msk = torch.from_numpy(msk).unsqueeze(0)                     
                info = data['info']
                frame = info['frame']#[0]
                shape = info['shape']
                shape = [torch.tensor(shape[0]),torch.tensor(shape[1])]
                need_resize = info['need_resize']#[0]

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

                """
                For timing see https://discuss.pytorch.org/t/how-to-measure-time-in-pytorch/26964
                Seems to be very similar in testing as my previous timing method 
                with two cuda sync + time.time() in STCN though 
                """                    
                if not first_mask_loaded:
                    if msk is not None:
                        first_mask_loaded = True
                    else:
                        # no point to do anything without a mask
                        continue

                if config['flip']:
                    rgb = torch.flip(rgb, dims=[-1])
                    msk = torch.flip(msk, dims=[-1]) if msk is not None else None                

                # Map possibly non-continuous labels to continuous ones
                if msk is not None:
                    msk, labels = mapper.convert_mask(msk[0].numpy(),exhaustive=True)
                    msk = torch.Tensor(msk).cuda()
                    if need_resize:
                        msk = vid_reader.resize_mask(msk.unsqueeze(0))[0]
                    processor.set_all_labels(list(mapper.remappings.values()))
                else:
                    labels = None
                                
                # Run the model on this frame
                prob = processor.step(rgb, msk, labels, end=(ti==vid_length-1))

                # Upsample to original size if needed
                if need_resize:
                    prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]

                end.record()
                torch.cuda.synchronize()
                total_process_time += (start.elapsed_time(end)/1000)
                total_frames += 1

                if config['flip']:
                    prob = torch.flip(prob, dims=[-1])
                
                if config['save_scores']:
                    prob = (prob.detach().cpu().numpy()*255).astype(np.uint8)
                    np_path = path.join(output, 'Scores', vid_name)
                    os.makedirs(np_path, exist_ok=True)
                    if ti==len(loader)-1:
                        hkl.dump(mapper.remappings, path.join(np_path, f'backward.hkl'), mode='w')
                    if config['save_all'] or info['save']:#[0]:
                        hkl.dump(prob, path.join(np_path, f'{frame[:-4]}.hkl'), mode='w', compression='lzf')

                # Probability mask -> index mask
                out_mask = torch.max(prob, dim=0).indices
                out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
                
                propagated_outputs[ti]=out_mask
                
                # Save the mask
                if config['save_all'] or info['save']:#[0]:
                    # this_out_path = path.join(output, vid_name)
                    # os.makedirs(this_out_path, exist_ok=True)
                    this_out_path = output
                    out_mask = mapper.remap_index_mask(out_mask)
                    out_img = Image.fromarray(out_mask)
                    #if vid_reader.get_palette() is not None:
                    #    out_img.putpalette(vid_reader.get_palette())
                    out_img.putpalette(palette)
                    out_img.save(os.path.join(this_out_path, frame[:-4]+'.png'))
        
        if config['cutoff']:
            for ti in chain(range(ceil, vid_length), range(floor, -1,-1)):
                with torch.cuda.amp.autocast(enabled=not config['benchmark']):
                    data = vid_reader[ti]
                    msk = data.get('mask')  
                    propagated_outputs[ti]=msk     
                    
        print(f'[XMEM INFO] Mask found for {count} frames!')
        break

    # print(f'Total processing time: {total_process_time}')
    # print(f'Total processed frames: {total_frames}')
    # print(f'FPS: {total_frames / total_process_time}')
    # print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')
    
    return np.array(propagated_outputs).astype('uint8')


if __name__ =='__main__':
    """
    Arguments loading
    """
    parser = ArgumentParser()
    parser.add_argument('--model', default='./saves/XMem.pth')

    # Data options
    # For generic (G) evaluation, point to a folder that contains "JPEGImages" and "Annotations"
    parser.add_argument('--generic_path')
    parser.add_argument('--output', default=None)
    parser.add_argument('--save_all', action='store_true', 
                help='Save all frames. Useful only in YouTubeVOS/long-time video', )

    parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
            
    # Long-term memory options
    parser.add_argument('--disable_long_term', action='store_true')
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                    type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)

    # Multi-scale options
    parser.add_argument('--save_scores', action='store_true')
    parser.add_argument('--flip', action='store_true')
    parser.add_argument('--size', default=480, type=int, 
                help='Resize the shorter side to this size. -1 to use original resolution. ')

    args = parser.parse_args()
    config = vars(args)
    config['enable_long_term'] = not config['disable_long_term']

    eval_xmem(config)