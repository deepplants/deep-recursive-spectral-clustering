import argparse
from argparse import Namespace
import sys
import logging
import logging.config
import json
import os
from datetime import datetime as dt
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms.functional as FV
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torchvision import models as torchvision_models
from pathlib import Path
from typing import Optional, Tuple, List, Any, Callable, Dict, Union
from PIL import Image
from tqdm import tqdm
import numpy as np
import signal
import glob
from collections import OrderedDict

import misc
import tree

def parse_args() -> Namespace:
    
    torchvision_archs = sorted(name for name in torchvision_models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(torchvision_models.__dict__[name]))

    parser = argparse.ArgumentParser(prog='distill-cuts',description=__doc__,add_help=False)

    # Logging level
    parser.add_argument('--logging', type=str, default='INFO',
                        choices=list(logging.getLevelNamesMapping().keys()),
                        help="""Optional arguments to pass to choose logging level.""")

    # Experiment parameters
    parser.add_argument('--conf', type=str, default=None, help="""Path to config
                        file with stored hyperparameters. Whether specify arguments
                        different from default or provide config file path.""")
    parser.add_argument('--conf_export', action='store_true', help="""Whether
                        to save arguments as json in the exp-dir directory.""")
    parser.add_argument('--output_root', type=str, default="data", help="""Saving
                        root path.""")
    # Data parameters
    parser.add_argument('--dataset', type=str, default="voc12_trainaug",
                        help="""Name of the dataset to load.""")
    parser.add_argument('--split', type=str, default="val",
                        help="""Name of the dataset split to load.""")
    parser.add_argument('--datasets_config', type=str, default="./config/datasets.json",
                        help="""JSON file with dataset configs.""")
    # Model parameters
    parser.add_argument('--model', default='dino_vitb8', type=str,
                        choices=torch.hub.list('facebookresearch/dinov2:main') \
                            + torch.hub.list('facebookresearch/dino:main') \
                            + torch.hub.list("facebookresearch/xcit:main") \
                            + torch.hub.list("facebookresearch/deit:main") \
                            + torchvision_archs + ["mae_vitb16", "mocov3_vits16", "mocov3_vitb16"],
                        help="""Name of model to train. For quick experiments
                        with ViTs, we recommend using dino_vits16""")
    # Task parameters
    parser.add_argument('--tasks', type=json.loads, default='{"task1": "kwargs"}', help="""
                        Task to execute.""")
    parser.add_argument('--schedule', nargs='+', required=False, help="""
                        Task scheduling to execute.""")
    
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")

    parser.add_argument('--check_if_empty', action='store_true',
                        help="""Set this flag for interactive checking before folder overriding.""")
    args = parser.parse_args()

    if args.conf is not None:
        if misc.is_main_process():
            with open(args.conf, 'r') as f:
                parser.set_defaults(**json.load(f))
            # Reload arguments to override config file values with command line values
            args = parser.parse_args()

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    logging.info(f"Output root is: {args.output_root}")

    if args.conf_export:
        if misc.is_main_process():
            tmp_args = vars(args).copy()
            del tmp_args['conf_export']  # Do not dump value of conf_export flag
            del tmp_args['conf']  # Values already loaded
            output_path = Path(args.output_root) / args.dataset
            misc.make_output_dir(output_path, args.check_if_empty)
            with open(output_path / 'config.json', 'w', encoding='utf-8') as f:
                json.dump(tmp_args, f, ensure_ascii=False, indent=4)
            logging.info(f"Saved config file in: {output_path / 'config.json'}")

    return args

def main_only(func):
    def wrapper(*args, **kwargs):
        if misc.is_main_process():
            func(*args, **kwargs)
    return wrapper

def extract_features(args: Namespace) -> None:

    # retrieve task arguments
    this_args = misc.get_task_args(args.tasks, "extract_features")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # create output directory
    output_path = ( Path(args.output_root) / args.dataset /
                   args.model / this_args.output_folder )
    if misc.is_main_process():
        misc.make_output_dir(output_path, args.check_if_empty)
    
    # wait for all processes to synchronize
    if dist.is_initialized():
        dist.barrier()

    # setup logger
    this_args.log_filename += '_' + args.split
    misc.setup_logger(this_args, output_path)

    # start feature extraction
    logging.info(f"Output directory is: {output_path}")
    logging.info('==== Start feature extraction ====')

    # load model and log model architecture
    model, patch_size, embed_dim, num_heads, num_tokens = misc.parse_model(args)
    model = model.to(device)
    model.eval()
    logging.debug(model)

    # Add forward hook to extract features
    hook = {}
    if 'dinov2' in args.model and 'reg' in args.model:
        def hook_vit(module, input, output):
            hook['features'] = output[:,num_tokens:,:]
        model._modules["norm"].register_forward_hook(hook_vit)  # type: ignore
    elif 'dino' in args.model or 'mocov3' in args.model or 'mae' in args.model: 
        if 'resnet50' in args.model:
            def hook_resnet50(module, input, output):
                hook['features'] = torch.flatten(output.permute((0,2,3,1)),start_dim=1,end_dim=2)
            model._modules["layer4"][-1].register_forward_hook(hook_resnet50) # type: ignore
        elif 'vit' in args.model:
            def hook_vit(module, input, output):
                hook['features'] = output[:,num_tokens:]
            model._modules["norm"].register_forward_hook(hook_vit)  # type: ignore
        else:
            raise NotImplementedError(f"Forward hook for {args.model} not implemented yet!")
    elif 'deit' in args.model:
        def hook_vit(module, input, output):
            hook['features'] = output[:,num_tokens:]
        model._modules["norm"].register_forward_hook(hook_vit)  # type: ignore
    else:
        raise NotImplementedError(f"Forward hook for {args.model} not implemented yet!")

    # check dataset config 
    split_args = misc.get_split_args(args.datasets_config, args.dataset, args.split)
    
    # create dataset and dataloader
    dataset = misc.AnyDataset(transform=misc.get_transform(name=args.model, **vars(this_args)), load_func=(lambda x: Image.open(x).convert('RGB')), **split_args.img)
    if this_args.resume:
        # remove already processed images if resume extraction
        list_done = [Path(p).stem for p in glob.glob(str(Path(output_path) / ("*.pth")))]
        dataset.names_list = list(sorted(set(dataset.names_list).difference(set(list_done))))
    sampler = misc.DistributedEvalSampler(dataset, seed=args.seed, shuffle=False) if dist.is_initialized() else None
    dataloader = DataLoader(dataset, batch_size=this_args.batch_size_per_gpu, num_workers=this_args.num_workers, sampler=sampler)

    # log dataset and dataloader size
    logging.info(f'Dataset size: {len(dataset)}')
    logging.info(f'Dataloader size: {len(dataloader)}')
    
    # compute features empirical covariance
    sum = torch.zeros((embed_dim,), dtype=torch.float64, device=device)
    joinsum = torch.zeros((embed_dim,embed_dim), dtype=torch.float64, device=device)
    n_obs = torch.zeros((1,), dtype=torch.int64, device=device)
        
    # run loop
    iterable = tqdm(dataloader, desc="Features extraction")
    for (images, ids, paths) in iterable:
        B, C, H, W = images.shape  
        H_patch, W_patch = H // patch_size, W // patch_size
        images = images.to(device).detach()
        if this_args.unfold:
            assert this_args.kernel is not None and this_args.stride is not None, "Kernel and stride must be specified for unfold."
            kernel = tuple([this_args.kernel]*2) if isinstance(this_args.kernel, int) else tuple(this_args.kernel)
            stride = tuple([this_args.stride]*2) if isinstance(this_args.stride, int) else tuple(this_args.stride)
            images = F.unfold(images, kernel_size=kernel, stride=stride)
            images = images.permute(0,2,1).reshape(-1,C,kernel[0],kernel[1])
        # forward pass
        with torch.no_grad():
            model(images)   
            features = hook['features'].to(torch.float64)
            if this_args.unfold:
                features = features.reshape(B, features.shape[0]//B, -1,features.shape[-1]).permute(0,1,3,2)
                features = features.reshape(B, features.shape[1], -1).permute(0,2,1)
                assert this_args.kernel is not None and this_args.stride is not None, "Kernel and stride must be specified for unfold."
                kernel = tuple([this_args.kernel//patch_size]*2) if isinstance(this_args.kernel, int) else tuple(x//patch_size for x in this_args.kernel)
                stride = tuple([this_args.stride//patch_size]*2) if isinstance(this_args.stride, int) else tuple(x//patch_size for x in this_args.stride)
                overlaps = F.fold(torch.ones_like(features), (H_patch, W_patch), kernel_size=kernel, stride=stride).permute(0,2,3,1)
                features = F.fold(features, (H_patch, W_patch), kernel_size=kernel, stride=stride).permute(0,2,3,1)
                features = (features/overlaps).reshape(B, -1, embed_dim)
            # features = torch.nn.functional.normalize(features, dim=-1, p=2)
            P = features.shape[1]
            n_obs += torch.tensor(B * P, dtype=n_obs.dtype, device=device)
            sum += torch.sum(features, dim=(0,1))
            joinsum += torch.einsum('bpe,bpf->ef', features, features)
        for i in range(B):
            output_dict = {}
            output_dict['data'] = features[i].reshape(H_patch, W_patch, embed_dim).detach().cpu()
            output_dict['shape'] = (H_patch, W_patch, embed_dim)
            output_dict['patch_size'] = patch_size
            output_dict['id'] = ids[i]
            output_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in output_dict.items()}
            # save features
            dest = output_path / ( ids[i] + '.pth')
            dest.parent.absolute().mkdir(parents=True, exist_ok=True)
            torch.save(output_dict, dest)
    
    # synchronize and reduce
    dist.all_reduce(sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(joinsum, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_obs, op=dist.ReduceOp.SUM)
    dist.barrier()
    
    if misc.is_main_process():
        path = output_path / 'covariance'
        misc.make_output_dir(path, args.check_if_empty)
        # save empirical covariance
        # https://it.wikipedia.org/wiki/Covarianza_(probabilit%C3%A0)#:~:text=.-,Statistica,-%5Bmodifica%20%7C
        cov = ((joinsum / (n_obs - 1)) - (sum / (n_obs - 1))[:,None] * (sum / n_obs)[None]).detach().cpu()
        torch.save(cov, path / (f'cov_{args.split}.pth'))
        logging.info(f'Empirical covariance saved in {path / (f"cov_{args.split}.pth")}')

@main_only
def semantic_ncuts(args: Namespace) -> None:

    # retrieve task arguments
    this_args = misc.get_task_args(args.tasks, "semantic_ncuts")

    # create output directory
    output_path = (Path(args.output_root) / args.dataset /
                args.model / this_args.output_folder / args.split /
                this_args.overclustering / this_args.normalization / 
                this_args.distance / this_args.heuristic / 
                (this_args.merging if this_args.merging is not None  else 'unmerged'))
    misc.make_output_dir(output_path,args.check_if_empty)

    # setup logger
    misc.setup_logger(this_args, output_path)
    misc.fix_random_seeds(args.seed)

    # start semantic ncut
    logging.info(f"Output directory is: {output_path}")
    logging.info('==== Start semantic ncut ====')
    
    # loading directory
    extr_args = misc.get_task_args(args.tasks, "extract_features")
    output_folder_name = extr_args.output_folder
    if this_args.distance == "mahalanobis":
        output_folder_name = misc.get_task_args(args.tasks, "density_projection").output_folder     
    source_path = ( Path(args.output_root) / args.dataset /
                args.model / output_folder_name )
    assert Path(source_path).is_dir(), f"Features source directory {source_path} do not exists!"
    logging.info(f"Loading features from {source_path}.")

    # Saving ncuts directory
    ncuts_output_path = output_path / 'trees'
    misc.make_output_dir(ncuts_output_path,args.check_if_empty)
    logging.info(f"Grouping output directory is {ncuts_output_path}")

    # Saving stripes directory
    imgs_output_path = output_path / 'imgs'
    misc.make_output_dir(imgs_output_path,args.check_if_empty)
    logging.info(f"Visive grouping output directory is {imgs_output_path}")

    # check dataset config 
    split_args = misc.get_split_args(args.datasets_config, args.dataset, args.split)
    
    # build datasets
    img_dataset = misc.AnyDataset(transform=None,  load_func=(lambda x: Image.open(x).convert('RGB')), **split_args.img)
    annot_dataset = misc.AnyDataset(transform=None, load_func=Image.open, **split_args.annot) if 'dir' in split_args.annot else None
    split_args.img['dir'], split_args.img['ext'] = source_path, 'pth' # override
    feat_dataset = misc.AnyDataset(transform=None, load_func=torch.load, **split_args.img)
    palette = misc.get_palette(args.dataset, args.split)

    if this_args.resume:
        # remove already processed images if resume extraction
        list_done = [Path(p).stem for p in glob.glob(str(Path(ncuts_output_path) / ("*.pth")))]
        list_todo = sorted(set(img_dataset.names_list).difference(set(list_done)))
        img_dataset.names_list = list_todo
        if annot_dataset:
            annot_dataset.names_list = list_todo
        feat_dataset.names_list = list_todo
    
    # log dataset size
    logging.info(f'Dataset size: {len(feat_dataset)}')
    
    def process(index: int, log_queue, log_level) -> None:
        # configure subprocess logger
        worker_logger = misc.configure_worker_logger(log_queue, log_level)
        
        # pick sample and run spectral clustering
        img, _, *_ = img_dataset.__getitem__(index)
        input_dict, id, *_ = feat_dataset.__getitem__(index)
        
        assert input_dict['id'] == id, f"Image id mismatch {input_dict['id']} vs {id}!"
        worker_logger.debug(f'Start processing image id {id}')
        (H_patch, W_patch, emb_dim) = input_dict['shape']
        if this_args.distance == "mahalanobis":
            this_args.distance = "l2"
        patch_size = input_dict['patch_size']
        if this_args.unfold:
            worker_logger.debug(f'Unfold ncut grouping for image id {id}')
            kernel = tuple([this_args.kernel//patch_size]*2) if isinstance(this_args.kernel, int) else tuple(x//patch_size for x in this_args.kernel)
            stride = tuple([this_args.stride//patch_size]*2) if isinstance(this_args.stride, int) else tuple(x//patch_size for x in this_args.stride)
            feats = input_dict['data'].permute(2,0,1).detach().cpu().to(torch.float64)[None] # add batch dimension
            feats = F.unfold(feats, kernel_size=kernel, stride=stride) # 1 x emb_dim*kernel*kernel x num_patches
            feats = feats.permute(0,2,1).reshape(-1,emb_dim,kernel[0]*kernel[1]).permute(0,2,1).numpy()
            groups, embs, root = misc.ncut_grouping_unfolded(points=feats, logger=worker_logger, **vars(this_args))
            groups = torch.tensor(groups)[None].permute(0,2,1)
            groups = F.fold(groups.float(), (H_patch, W_patch), kernel_size=kernel, stride=stride).permute(0,2,3,1).long().numpy()
        else:
            worker_logger.debug(f'Standard ncut grouping for image id {id}')
            feats = input_dict['data'].detach().cpu().numpy().astype(np.float64)
            groups, embs, root = misc.ncut_grouping(points=feats, img=np.array(img), logger=worker_logger, **vars(this_args))
        worker_logger.debug(f'End processing image id {id}')

        # save output
        groups = np.reshape(groups,(H_patch, W_patch))
        output_dict = {}
        output_dict['shape'] = input_dict['shape']
        output_dict['groups'] = groups
        output_dict['root'] = root
        output_dict['embs'] = embs
        output_dict['patch_size'] = patch_size
        output_dict['id'] = id
        output_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in output_dict.items()}
        dest = ncuts_output_path / (id + '.pth')
        dest.parent.absolute().mkdir(parents=True, exist_ok=True)
        torch.save(output_dict, dest)
        
        # visualization
        worker_logger.debug(root)

        # save stripes
        misc.save_strips(
            img=img,
            groups=groups,
            root=root,
            annot=annot_dataset.__getitem__(index)[0] if annot_dataset else None,
            palette=palette,
            dest=imgs_output_path / (id + '.png'),
            logger=worker_logger
        )

    with torch.no_grad():
        misc.parallel_process(range(len(feat_dataset)), process, this_args.num_workers, desc='semantic_ncuts')


@main_only
def postprocess_crf(
    args: Namespace
    ) -> None:

    
    this_args = misc.get_task_args(args.tasks, "postprocess_crf")

    sem_args = misc.get_task_args(args.tasks, "semantic_ncuts")
    
    post_args = misc.get_task_args(args.tasks, "postprocess_ncuts")
    
    # create output directory
    output_path = ( Path(args.output_root) / args.dataset /
                args.model / this_args.output_folder / args.split /
                sem_args.overclustering / sem_args.normalization /
                sem_args.distance / sem_args.heuristic / 
                (sem_args.merging if sem_args.merging is not None  else 'unmerged'))

    if misc.is_main_process():
        misc.make_output_dir(output_path, args.check_if_empty)
    
    # Saving ncuts directory
    ncuts_output_path = output_path / 'trees'
    if misc.is_main_process():
        misc.make_output_dir(ncuts_output_path,args.check_if_empty)
    logging.info(f"Output directory is: {ncuts_output_path}")

    # Saving img stripes directory
    imgs_output_path = output_path / 'imgs'
    if misc.is_main_process():
        misc.make_output_dir(imgs_output_path,args.check_if_empty)
    logging.info(f"Output directory is: {imgs_output_path}")
    
    # loading directory
    if sem_args.distance == "mahalanobis":
        extr_args = misc.get_task_args(args.tasks, "density_projection")
    else:
        extr_args = misc.get_task_args(args.tasks, "extract_features")
    feats_source_path = ( Path(args.output_root) / args.dataset /
                args.model / extr_args.output_folder )
    assert Path(feats_source_path).is_dir(), f"Features source directory {feats_source_path} do not exists!"
    logging.info(f"Loading features from {feats_source_path}.")
    tree_source_path = ( Path(args.output_root) / args.dataset /
                args.model / post_args.output_folder / args.split / sem_args.overclustering /
                sem_args.normalization / sem_args.distance / sem_args.heuristic /
                Path(sem_args.merging if sem_args.merging is not None  else 'unmerged') / 'trees')
    assert Path(tree_source_path).is_dir(), f"Features source directory {tree_source_path} do not exists!"
    logging.info(f"Loading features from {tree_source_path}.")
    
    # setup logger
    misc.setup_logger(this_args, output_path)
    misc.fix_random_seeds(args.seed)

    # start post processing ncuts
    logging.info(f"Output directory is: {output_path}")
    logging.info('==== Start crf postprocessing ====')

    # check dataset config 
    split_args = misc.get_split_args(args.datasets_config, args.dataset, args.split)
    
    img_dataset = misc.AnyDataset(transform=None, load_func=Image.open, **split_args.img)
    annot_dataset = misc.AnyDataset(transform=None, load_func=Image.open, **split_args.annot)
    palette = misc.get_palette(args.dataset, args.split)

    split_args.img['dir'], split_args.img['ext'] = feats_source_path, 'pth' # override
    feats_dataset = misc.AnyDataset(transform=None, load_func=torch.load, **split_args.img)
    split_args.img['dir'] = tree_source_path # override
    tree_dataset = misc.AnyDataset(transform=None, load_func=torch.load, **split_args.img)
            
    def process(index: int, log_queue, log_level):
        worker_logger = misc.configure_worker_logger(log_queue, log_level)
        # get best subtree that intersects all objects in foreground
        img = img_dataset.__getitem__(index)[0]
        annot = annot_dataset.__getitem__(index)[0]
        data = tree_dataset.__getitem__(index)[0]
        feats = feats_dataset.__getitem__(index)[0]
        root, groups, id, embs = data['root'], data['groups'], data['id'], data['embs']
        h, w = groups.shape
        
        labels = list(embs.keys())
        mapp = dict(zip(range(len(labels)),labels))
        prototypes = torch.tensor(np.array(list(embs.values()),np.float64), dtype=torch.float64).cpu()
        num_class = len(labels)
        
        import warnings
        # suppress joblib warning (to be understood)
        warnings.filterwarnings("ignore")
        worker_logger.debug('[INFO]: Running CRF post-processing..')
        from pydensecrf import utils
        from pydensecrf.densecrf import DenseCRF2D
        # CRF post-processor

        W, H = img.size
        new_min_dim = min((W,H)) #1024
        scale = new_min_dim/min((W,H))
        new_size = (int(W*scale), int(H*scale))
        img = img.resize(new_size).convert('RGB')
        annot = annot.resize(new_size, Image.NEAREST)
        feat = FV.resize(feats['data'].cpu().permute(2, 0, 1), list(new_size)[::-1], InterpolationMode.BICUBIC).permute(1, 2, 0).to(torch.float64)
        feat = F.normalize(feat, dim=-1, p=2)
        prototypes = F.normalize(prototypes, dim=-1, p=2)
        probs = F.softmax((feat @ prototypes.T)/1e-1, dim=-1).permute(2, 0, 1).numpy()
        
        U = utils.unary_from_softmax(probs)
        U = np.ascontiguousarray(U)
        image = np.ascontiguousarray(img)
        d = DenseCRF2D(W, H, num_class)
        d.setUnaryEnergy(U.copy())
        d.addPairwiseGaussian(sxy=1, compat=3)
        d.addPairwiseBilateral(sxy=67, srgb=3, rgbim=image.copy(), compat=4)
        Q = d.inference(10)
        Q = np.array(Q).reshape((num_class, H, W))
        crf_segs = np.argmax(Q, axis=0)
        groups = np.vectorize(mapp.__getitem__)(crf_segs)
        
        # save stripes
        misc.save_strips(
            img=img,
            groups=groups,
            root=root,
            annot=annot,
            palette=palette,
            dest=imgs_output_path / (id + '.png'),
            logger=worker_logger
        )
        
        groups_s = np.array(Image.fromarray(groups.astype(np.uint16)).resize((w,h), Image.NEAREST))
        embs = tree.pool_embeddings(root=root, points=feats['data'].cpu().numpy(), attr='idx', values=groups_s, pooling='average', mapping=dict(),leaves_only=True)
        
        data["root"] = root
        data["groups"] = groups
        data["embs"] = embs
        data = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in data.items()}
        dest = ncuts_output_path / (id + '.pth')
        dest.parent.absolute().mkdir(parents=True, exist_ok=True)
        torch.save(data, dest)
        
        return  None
    
    results = misc.parallel_process(range(len(tree_dataset)), process, this_args.num_workers, desc='CRF post processing')


@main_only
def create_stats(
    args: Namespace
    ) -> None:

    this_args = misc.get_task_args(args.tasks, "create_stats")

    sem_args = misc.get_task_args(args.tasks, "semantic_ncuts")
    
    post_args = misc.get_task_args(args.tasks, "postprocess_ncuts")
    
    output_path =  ( Path(args.output_root) / args.dataset /
                args.model / this_args.output_folder / args.split /
                sem_args.overclustering / sem_args.normalization /
                sem_args.distance / sem_args.heuristic / 
                (sem_args.merging if sem_args.merging is not None  else 'unmerged'))
    
    if misc.is_main_process():
        misc.make_output_dir(output_path,args.check_if_empty)
    
    source_path =  ( Path(args.output_root) / args.dataset /
                args.model / post_args.output_folder / args.split /
                sem_args.overclustering / sem_args.normalization /
                sem_args.distance / sem_args.heuristic / 
                (sem_args.merging if sem_args.merging is not None  else 'unmerged')) / 'trees'

    # Saving img stripes directory
    imgs_output_path = output_path / 'imgs'
    if misc.is_main_process():
        misc.make_output_dir(imgs_output_path,args.check_if_empty)
    logging.info(f"Output directory is: {imgs_output_path}")
    
    # setup logger
    misc.setup_logger(this_args, output_path)
    misc.fix_random_seeds(args.seed)

    logging.info(f"Output directory is: {output_path}")
    logging.info('==== Start background pruning ====')

    # check dataset config 
    split_args = misc.get_split_args(args.datasets_config, args.dataset, args.split)
    
    assert 'dir' in split_args.annot, f"Missing 'dir' attribute in {args.dataset} dataset annot config file!"
    
    img_dataset = misc.AnyDataset(transform=None, load_func=Image.open, **split_args.img)
    annot_dataset = misc.AnyDataset(transform=None, load_func=Image.open, **split_args.annot)
    palette = misc.get_palette(args.dataset, args.split)
    assert isinstance(annot_dataset.labels_txt, Path), f"Missing 'labels_txt' attribute in {args.dataset} dataset annot config file!"
    hier_labels = misc.read_and_parse_hier_labels(name=annot_dataset.labels_txt.stem, labels_txt=annot_dataset.labels_txt)

    split_args.img['dir'], split_args.img['ext'] = source_path, 'pth' # override
    tree_dataset = misc.AnyDataset(transform=None, load_func=torch.load, **split_args.img)    
    
    hier_levels = hier_labels.max_depth()
    depths = hier_labels.depths_dict()
    
    fg_nodes = tree.search(hier_labels,'type', misc.CatType.FOREGROUND)
    bg_ids = [getattr(node,'idx') for node in tree.search(hier_labels, 'type', misc.CatType.BACKGROUND) if hasattr(node,'idx')]
    ign_ids = [getattr(node,'idx') for node in tree.search(hier_labels, 'type', misc.CatType.IGNORE) if hasattr(node,'idx')]
    
    for i in range(0, hier_levels):
        logging.info(f"Start evaluation at depth {hier_levels-i}.")
        
        if i == 0:
            remap = {getattr(node,'idx') : j for j,node in enumerate(fg_nodes) if hasattr(node,'idx')}
            labels = ["--".join(getattr(an,'name') for an in node.dinasty()[1:] if hasattr(an,'name')) for node in fg_nodes]
        else:
            fg_nodes_ancest = {getattr(node,'idx') : node.ancestor(i) if (depths[node]>hier_levels-i and hasattr(node,'idx')) else node for node in fg_nodes}
            fg_ancest_ids = {node: j for j, node in enumerate(OrderedDict.fromkeys(fg_nodes_ancest.values()).keys())}
            remap = {k: fg_ancest_ids[v] for k,v in fg_nodes_ancest.items()}
            labels = ["--".join(getattr(an,'name') for an in node.dinasty()[1:] if hasattr(an,'name')) for node in fg_ancest_ids.keys()]

        num_classes = len(list(set(remap.values())))
        
        if len(bg_ids):
            num_classes += 1
            remap = {k: j+1 for k,j in remap.items()}
            remap.update({e: 0 for e in bg_ids})
            
        hmiou = misc.MatchingIoU(num_classes=num_classes, hung_match_freq="sample")

        def process(index: int, log_queue, log_level,  *arg, **kwarg) -> Dict[str, np.ndarray]:
            worker_logger = misc.configure_worker_logger(log_queue, log_level)
            # get best subtree that intersects all objects in foreground
            img = img_dataset.__getitem__(index)[0]
            annot = annot_dataset.__getitem__(index)[0]
            data = tree_dataset.__getitem__(index)[0]
            root, groups, id = data['root'], data['groups'], data['id']
            h, w = groups.shape
            annot_down_a = np.asarray(annot.resize((w,h), resample=Image.Resampling.NEAREST))
            annot_down = annot_down_a.reshape((h*w,))
            valid = np.logical_not(np.in1d(annot_down, ign_ids))
            annot_down = annot_down[valid]
            if annot_down.size != 0:
                annot_down = np.vectorize(remap.__getitem__)(annot_down)
                sparse_true = np.eye(num_classes)[annot_down]
                # take only possible subtrees to reduce computations
                mapp, sparse_pred = tree.mask_subtree_nodes_leaves_in_values(root, 'idx', groups)
                
                assert sparse_pred.shape[0]>0, "No valid subtree found!"
                assert sparse_pred.shape[1]==h and sparse_pred.shape[2]==w, "shape mismatch!"
                sparse_pred_ = sparse_pred.reshape((-1,h*w)).T[valid]
                # sparse_true = sparse_true[:,1:]
                result = hmiou.update_state(sparse_true, sparse_pred_) 
                sparse_pred  = sparse_pred.astype(int)       
                worker_logger.info(f'[INFO]: image {id} HNECovering {result["hnecovering"]}.')
                return  result
            else:
                return dict()

        results = misc.parallel_process(range(len(tree_dataset)), process, this_args.num_workers, desc='Evaluating',require='sharedmem')
        hnecovering = np.nanmean([e['hnecovering'] for e in results if e is not None])
        nfcovering = np.nanmean([e['nfcovering'] for e in results if e is not None])
        
        result = {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in hmiou.result().items()}
        result['labels'] = (['background'] if len(bg_ids) else []) + labels
        result['dataset'] = args.dataset
        result.update(zip(result.pop('labels'), result.pop('iou')))
        result['hnecovering'] = hnecovering
        result['nfcovering'] = nfcovering
        logging.info(result)
        
        misc.make_output_dir(output_path, False)
        path = output_path / f'results_{args.dataset}_{args.split}_{hier_levels-i}.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logging.info(f"Saved config file in: {path}")
        
    hmiou = misc.MatchingHierarchicalIoU(labels_hierarchy=hier_labels, threshold=0.0, match_type='max')

    def process(index: int, log_queue, log_level,  *arg, **kwarg) -> Dict[str, np.ndarray]:
        worker_logger = misc.configure_worker_logger(log_queue, log_level)
        # get best subtree that intersects all objects in foreground
        annot = annot_dataset.__getitem__(index)[0]
        data = tree_dataset.__getitem__(index)[0]
        root, pred, id = data['root'], data['groups'], data['id']
        h, w = pred.shape
        true = np.asarray(annot.resize((w,h), resample=Image.Resampling.NEAREST))
        worker_logger.info(f'[INFO]: start processing image {id}.')
        result = hmiou.update_state(true, pred, root)        
        worker_logger.info(f'[INFO]: image {id} nfcovering {result["nfcovering"] if "nfcovering" in result else None}.')
        return  result
    
    results = misc.parallel_process(range(len(tree_dataset)), process, this_args.num_workers, desc='Evaluating',require='sharedmem')
    nhcovering = np.nanmean([e['nhcovering'] for e in results if 'nhcovering' in e])
    nmcovering = np.nanmean([e['nmcovering'] for e in results if 'nmcovering' in e])
    
    result = {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in hmiou.result().items()}
    result['dataset'] = args.dataset
    result['nhcovering'] = nhcovering
    result['nmcovering'] = nmcovering
    logging.info(result)
    
    misc.make_output_dir(output_path, False)
    path = output_path / f'results_hier_{args.dataset}_{args.split}_{hmiou.match_type}_{hmiou.threshold}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    logging.info(f"Saved config file in: {path}")

if __name__ == "__main__" :

    args = parse_args()

    for task in args.schedule:
        #init distributed mode and fix random seeds (run on GPU if available)
        if dist.is_available():
            if not dist.is_initialized():
                misc.init_distributed_mode(args)
                misc.fix_random_seeds(args.seed)
                cudnn.benchmark = True
        print(task)
        eval(task)(args) # execute task
