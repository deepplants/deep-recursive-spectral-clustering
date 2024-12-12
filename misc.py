import argparse
from argparse import Namespace
import matplotlib as mpl
mpl.use("Agg")
from matplotlib import pyplot as plt
import logging
from logging.handlers import QueueHandler, QueueListener
from colorlog import ColoredFormatter
import json
import os
import sys
import time
import math
import random
import subprocess
from datetime import datetime as dt
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from pathlib import Path
from typing import Mapping, Optional, Tuple, List, Any, Callable, Iterable, Dict, Literal, Union
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import joblib
from scipy.sparse.linalg import eigsh, eigs
from torch.utils.data import Sampler
from sklearn.cluster import KMeans, MiniBatchKMeans
from copy import deepcopy
from PIL import Image
from collections import ChainMap
from multiprocessing import Manager
import scipy as scp
import glob
import time
from PIL import ImageFile
import cv2
from scipy.spatial.distance import pdist, squareform
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tree import TreeNode
import tree
EPSILON = 1E-12
    
from typing import Callable
from torch import multiprocessing as mp
from scipy.sparse import coo_array

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.transform import resize

from enum import Enum

class CatType(Enum):
    FOREGROUND = 0
    BACKGROUND = 1
    IGNORE = 2
    
def setup_logger(
    args: Namespace,
    save_path: Union[Path,str]
    ) -> None:
    """
    This function setup logger level
    """
    level = (args.logging.upper() if 
        isinstance(args.logging, str) else 
        args.logging)
    root = logging.getLogger()
    root.handlers[0].setFormatter(ColoredFormatter('%(log_color)s%(process)d [%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d]\n%(message)s', datefmt='%a, %d %b %Y %H:%M:%S'))
    root.setLevel(level)

    if args.log_filename:
        log_file = logging.FileHandler(
            Path(save_path) / f"{args.log_filename}.log",
            mode='w' if not hasattr(args,'resume') else 'a' if args.resume else 'w',
            encoding = 'utf-8')
        log_file.setLevel(level)
        log_file.setFormatter(logging.Formatter('%(process)d [%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d]\n%(message)s', datefmt='%a, %d %b %Y %H:%M:%S'))
        root.addHandler(log_file)
    
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.info("git:\n  {}\n".format(get_sha()))
    logging.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    


def configure_worker_logger(log_queue, log_level) -> logging.Logger:
    """Adapted from https://github.com/joblib/joblib/issues/1017#issuecomment-1535983689
    Args:
        log_queue (_type_): _description_
        log_level (_type_): _description_

    Returns:
        _type_: _description_
    """
    worker_logger = logging.getLogger('worker')
    if not worker_logger.hasHandlers():
        h = QueueHandler(log_queue)
        worker_logger.addHandler(h)
    worker_logger.setLevel(log_level)
    return worker_logger


def plt_imshow_list(
    imgs: List[np.ndarray], 
    subtitles: Optional[List[str]] = None, 
    nrows: Optional[int] = None, 
    ncols: Optional[int] = None,
    savefilepath: Optional[Union[Path,str]] = None,
    cmaps: Optional[List[Union[None,str]]] = None,
    legend: Optional[Dict[str,Tuple[int]]] = None,
    **kwargs
    ) -> None:
    
    logger = kwargs.get('logger', None) or logging.getLogger()
        
    if subtitles is not None:
        assert len(imgs) == len(subtitles), "images and names must have same length"
    if cmaps is not None:
        assert len(imgs) == len(cmaps), "images and cmaps must have same length"

    ncols = ncols or len(imgs)
    nrows = nrows or 1

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 4*nrows),
                        subplot_kw={'xticks': [], 'yticks': []})
    i=0
    for ax in axs.flat:
        if subtitles is not None:
            ax.set_title(str(subtitles[i]))
        if cmaps is not None:
            im=ax.imshow(imgs[i],cmap=cmaps[i])
        else:
            im=ax.imshow(imgs[i])
        i += 1
        
    if legend is not None:
        # legend is dict(label_i=color_i)
        import matplotlib.patches as mpatches
        patches = [mpatches.Patch(color=k, label=v ) for k,v in legend.items() ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,prop={'size':8}, ncol=(len(patches)//8)+1 )

    fig.tight_layout()

    if savefilepath is not None:
        Path(savefilepath).parent.absolute().mkdir(parents=True, exist_ok=True)
        fig.savefig(str(savefilepath), dpi=100)
        plt.close()
    else:
        plt.show()
    plt.close('all')

def sugarbeets_colormap():
    # {0: (0, 0, 0), 10000: (0, 255, 0), 2: (255, 0, 0), 20001: (255, 0, 0), 20100: (255, 0, 0)}
    cmap = np.zeros((256, 3), dtype=np.uint8)
    cmap[1,:]=(0, 255, 0)
    cmap[2,:]=(255, 0, 0)
    return cmap

def voc_colormap(N=256, normalized=False):
    ''' map PascalVOC class labels to RGB colors '''
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255 if normalized else cmap
    return cmap

# Create color cmap from color dictionary
def cityscapes_colormap():
    from cityscapesscripts.helpers.labels import labels
    mapp =  { label.id : label.color for label in labels} 
    mapp.pop(-1)
    cmap = []
    for i in np.arange(256):
        if i in mapp:
            cmap.append(mapp[i])
        else:
            cmap.append([0, 0, 0])
    cmap = np.vstack(cmap).astype(np.uint8)
    return cmap

def kitti_step_colormap():
    from cityscapesscripts.helpers.labels import labels
    ids = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    mapp =  { label.id : label.color for label in labels} 
    mapp.pop(-1)
    cmap = []
    for i in ids:
        if i in mapp:
            cmap.append(mapp[i])
        else:
            cmap.append([0, 0, 0])
    cmap = np.vstack(cmap).astype(np.uint8)
    cmap = np.vstack((cmap,np.zeros((256-cmap.shape[0], 3),np.uint8)))
    cmap[-1,:] = 255

    return cmap

def coco_colormap(cmapName='jet'):
    '''
    Create a color map for the classes in the COCO Stuff Segmentation Challenge.
    :param cmapName: (optional) Matlab's name of the color map
    :return: cmap - [c, 3] a color map for c colors where the columns indicate the RGB values
    '''

    # Get jet color map from Matlab
    labelCount = 182 # number of cocostuff categories
    cmapGen = mpl.cm.get_cmap(cmapName, labelCount)
    cmap = cmapGen(np.arange(labelCount))
    cmap = cmap[:, 0:3]

    # Reduce value/brightness of stuff colors (easier in HSV format)
    cmap = cmap.reshape((-1, 1, 3))
    hsv = mpl.colors.rgb_to_hsv(cmap)
    hsv[:, 0, 2] = hsv[:, 0, 2] * 0.7
    cmap = mpl.colors.hsv_to_rgb(hsv)
    cmap = cmap.reshape((-1, 3))

    # Permute entries to avoid classes with similar name having similar colors
    st0 = np.random.get_state()
    np.random.seed(42)
    perm = np.random.permutation(labelCount)
    np.random.set_state(st0)
    cmap = cmap[perm, :]
    cmap = np.vstack((cmap,np.zeros((256-cmap.shape[0], 3))))
    cmap[-1,:] = 1.0
    cmap = (255*cmap).astype(np.uint8)

    return cmap

def get_palette(name: str, split: str) -> np.ndarray:
    if 'pascalpart' in name:
        return voc_colormap(390)
    elif 'voc' in name:
        return voc_colormap()
    elif 'cityscapes' in name:
        return cityscapes_colormap()
    elif 'mapillaryvistas' in name:
        return voc_colormap()
    elif 'kitti' in name:
        if 'step' in split:        
            return kitti_step_colormap()
        return cityscapes_colormap()
    elif 'potsdam' in name or 'vaihingen' in name:
        return potsdam_colormap()   
    elif 'coco' in name:
        return coco_colormap()
    elif 'partimagenet' in name:
        if '158' in split:
            return voc_colormap(601)
        return voc_colormap()
    elif 'sugarbeets' in name:
        return sugarbeets_colormap()
    
    raise ValueError(f"Palette not found for {name} and {split}")

def potsdam_colormap():
    mapp = {0: [255, 255, 255],  # roads
                1: [0, 0, 255],  # buildings
                2: [0, 255, 255],  # vegetation
                3: [0, 255, 0],  # tree
                4: [255, 255, 0],  # car
                5: [255, 0, 0]  # clutter
                }
    cmap = []
    for i in np.arange(256):
        if i in mapp:
            cmap.append(mapp[i])
        else:
            cmap.append([0, 0, 0])
    cmap = np.vstack(cmap).astype(np.uint8)
    return cmap

def npy_rgb_to_index(npy_rgb, idx2rgb):
    id2gray = {k: int(cv2.cvtColor(np.array(v,np.uint8)[None,None], cv2.COLOR_RGB2GRAY)[0,0]) for k,v in idx2rgb.items()}
    lut = np.ones(256, dtype=np.uint8) * 255
    lut[list(id2gray.values())] = np.array(list(id2gray.keys()), dtype=np.uint8)
    npy_idx = cv2.LUT(cv2.cvtColor(npy_rgb, cv2.COLOR_RGB2GRAY), lut)
    return npy_idx

def npy_index_to_rgb(npy_idx, idx2rgb):
    idxs = np.unique(npy_idx)
    h,w = npy_idx.shape
    rgb = np.zeros((h,w,3)).astype(np.uint8)
    for i in idxs:
        rgb[npy_idx == i, :] = idx2rgb[i]
    return rgb   

def parse_model(args) -> Tuple[nn.Module, int, int, int]:

    if 'dinov2' in args.model:
        if 'reg' in args.model:
            model = torch.hub.load('facebookresearch/dinov2', args.model) # with registers, e.g. dinov2_vitb14_reg
        else:
            model = torch.hub.load('facebookresearch/dinov2:main', args.model)
        model.fc = torch.nn.Identity()
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
        embed_dim = model.embed_dim
        num_tokens = model.num_tokens + model.num_register_tokens
    elif 'dino' in args.model:
        model = torch.hub.load('facebookresearch/dino:main', args.model)
        model.fc = torch.nn.Identity()
        if 'vit' in args.model:
            patch_size = model.patch_embed.patch_size
            num_heads = model.blocks[0].attn.num_heads
            embed_dim = model.embed_dim
            num_tokens = 1
        elif 'resnet50' in args.model:
            patch_size = 32
            num_heads = -1   
            embed_dim = 2048
            num_tokens = 0
        else:
            raise ValueError(f'Cannot get model: {args.model}')
    elif 'mae' in args.model:
        model = torch.hub.load('facebookresearch/dino:main', args.model.replace('mae', 'dino'))
        checkpoint_file = {
            'mae_vitb16': 'mae_finetuned_vit_base',
            'mae_vitl16': 'mae_finetuned_vit_large',
            'mae_vith14': 'mae_finetuned_vit_huge',
        }[args.model]
        url = f'https://dl.fbaipublicfiles.com/mae/finetune/{checkpoint_file}.pth'
        checkpoint = torch.hub.load_state_dict_from_url(url)
        model.fc = torch.nn.Identity()
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
        embed_dim = model.embed_dim
        num_tokens = 0
    elif 'mocov3' in args.model:
        model = torch.hub.load('facebookresearch/dino:main', args.model.replace('mocov3', 'dino'))
        checkpoint_file, size_char = {
            'mocov3_vits16': ('vit-s-300ep-timm-format.pth', 's'), 
            'mocov3_vitb16': ('vit-b-300ep-timm-format.pth', 'b'),
        }[args.model]
        url = f'https://dl.fbaipublicfiles.com/moco-v3/vit-{size_char}-300ep/vit-{size_char}-300ep.pth.tar'
        checkpoint = torch.hub.load_state_dict_from_url(url)
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
        embed_dim = model.embed_dim
        model.load_state_dict({k.split('module.momentum_encoder.')[-1]:v for k,v in checkpoint['state_dict'].items()},strict=False)
        model.fc = torch.nn.Identity()
        num_tokens = 1
    elif 'deit' in args.model:
        model = torch.hub.load('facebookresearch/deit:main', args.model)
        model.fc = torch.nn.Identity()
        patch_size = model.patch_embed.patch_size
        num_heads = model.blocks[0].attn.num_heads
        embed_dim = model.embed_dim
        num_tokens = 1
    elif 'convnext' in args.model:
        raise NotImplementedError()
    else:
        raise ValueError(f'Cannot get model: {args.model}')
            
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if isinstance(patch_size, tuple):
        patch_size = patch_size[0]
    elif isinstance(patch_size, int):
        pass
    else: 
        raise ValueError(f"patch_size must be tuple or integer, got {type(patch_size)}")
    
    return model, patch_size, embed_dim, num_heads, num_tokens

def get_transform(name: str = 'dino', **kwargs):
    size = kwargs.get('resize',None)
    transform_list = []
    if size is not None:
        resize = transforms.Resize(
                size, # unique int will map lower size to int, tuple will match the sizes
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True) # type: ignore
        transform_list.append(resize)
    elif 'dinov2' in name: # only 14x14 patches
        resize = transforms.Lambda(
            lambda x: transforms.Resize(
                ((x.size[1]//14)*14*2, (x.size[0]//14)*14*2), 
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True)(x) # type: ignore
            )
        transform_list.append(resize)
    
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_list.extend([transforms.ToTensor(), normalize])
    transform = transforms.Compose(transform_list)
    return transform

def get_inverse_transform(name: str = 'dino'):
    if any(x in name for x in ('dino', 'mocov3', 'convnext', )):
        inv_normalize = transforms.Normalize(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            [1 / 0.229, 1 / 0.224, 1 / 0.225])
        transform = transforms.Compose([transforms.ToTensor(), inv_normalize])
    else:
        raise NotImplementedError()
    return transform

def get_task_args(tasks: Dict[str, Dict[str, Any]], name: str, **kwargs) -> Namespace:
    assert name in tasks, f"{name} task key missing in current config file."
    task_dict = tasks.get(name, None)
    assert task_dict is not None, f"{name} arguments missing in current config file."
    task_args = Namespace(**task_dict)
    return task_args

def get_split_args(config_path: str, name: str, split: str) -> Namespace:
    # check dataset config 
    assert os.path.isfile(config_path), f"File {config_path} not found!"
    datasets_dict = json.load(open(config_path))
    assert name in datasets_dict, f"Dataset {name} not found in {config_path}!"
    dataset_config = datasets_dict[name]
    assert split in dataset_config, f"Split {split} not found in {name}!"
    split_dict = dataset_config[split]
    assert split_dict.get('img', None) is not None, f"Split {split} of {name} misses 'img' flag!"
    assert split_dict.get('annot', None) is not None, f"Split {split} of {name} misses 'annot' flag!"
    split_args = Namespace(**split_dict)
    return split_args

def save_strips(
    img: Image.Image, 
    groups: np.ndarray, 
    root: TreeNode, 
    annot: Optional[Image.Image] = None,
    dest: Optional[Path] = None, 
    logger: Optional[logging.Logger] = None, 
    palette: Optional[np.ndarray] = None) -> None:

        mapp = {getattr(k,'idx'):v for k,v in root.exponential_indexing().items() if hasattr(k,'idx')}
        mapp = {k:v/max(mapp.values()) for k,v in mapp.items()} if max(mapp.values())>0 else mapp
        hierarchy_colour_code = np.vectorize(mapp.__getitem__)(groups).astype('float64')
        hierarchy_colour_code /= hierarchy_colour_code.max() if hierarchy_colour_code.max()>0 else hierarchy_colour_code
        ids = np.unique(hierarchy_colour_code)
        hierarchy_colour_code = resize(hierarchy_colour_code, img.size[::-1], anti_aliasing=True, order=0)
        groups = resize(groups, img.size[::-1], anti_aliasing=True, order=0)

        imgs = [img,hierarchy_colour_code,groups]
        subs = ['input','hierarchy','leafs']
        cmaps = [None, 'inferno', 'viridis']
        if annot is not None:
            if palette is not None:
                if len(palette) == 256:
                    annot.putpalette(palette)
                else:
                    annot = npy_index_to_rgb(np.array(annot), idx2rgb=palette)
            imgs += [annot]
            subs += ['annotation']
            cmaps += [None]     
        plt_imshow_list(
            imgs=imgs,
            subtitles=subs,
            savefilepath=dest,
            cmaps=cmaps,
            logger=logger
        )

def make_output_dir(output_dir, check_if_empty=True) -> bool:
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    if check_if_empty and (len(list(output_dir.iterdir())) > 0):
        print(f'Output dir: {str(output_dir)}')
        if not input(f'Output dir already contains files. Continue? (y/n) >> ').lower() in ['','y','yes']:
            # interrupt process
            sys.exit()
    # continue
    return True    

def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

def read_and_parse_hier_labels(name: str = 'voc12', labels_txt: Union[Path,str] = './labels/voc12.txt') -> TreeNode:
    """
    Read and parse hierarchical labels from a text file. Visual usage:

    from misc import read_and_parse_hier_labels
    a = read_and_parse_hier_labels('cocostuffthing', '/home/simone/workdir/labels/cocostuffthing.txt')
    print(a.__str__(showAttrs=['idx','type']))

    """
    level = -1
    child = TreeNode(name=name)
    root = child
    _CASES = {
        '0' : CatType.FOREGROUND, 
        '1' : CatType.BACKGROUND, 
        '2' : CatType.IGNORE 
    }
    for line in open(labels_txt, 'r').read().splitlines():
        current_level = sum([1 for e in line.split('\t') if e == ''])
        elems = [e for e in line.split('\t') if e != '']
        if current_level < level:
            for _ in range(level - current_level):
                child = child.parent
        if len(elems) == 1:
            _child = TreeNode(name=elems[0])
            child.add_child(_child)
            child = _child
        elif len(elems) == 3:
            assert check_int(elems[1]), f"Index {elems[1]} is not an integer."
            assert check_int(elems[2]) and elems[2] in _CASES, f"Case {elems[2]} not in {_CASES.keys()}"
            _child = TreeNode(name=elems[0], idx=int(elems[1]), type=_CASES[elems[2]])
            child.add_child(_child)
        level = current_level
        
    return root
class AnyDataset(Dataset):
    """A very simple dataset for loading ordered data."""

    def __init__(
        self,
        list_txt: Optional[str] = None,
        load_func: Optional[Callable] = Image.open,
        dir: Optional[str] = None,
        ext: Optional[str] = None,
        transform: Optional[Callable] = None,
        labels_txt: Optional[str] = None,
        **kwargs
        ) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.dir = None if dir is None else Path(dir)
        self.ext = ext
        self.transform = transform
        self.load_func = load_func
        self.list_txt = list_txt
        if list_txt is not None:
            self.names_list = sorted(list(set(open(list_txt, 'r').read().splitlines(keepends=False))))
        elif dir is not None and ext is not None:
            self.names_list = sorted([Path(p).stem for p in glob.glob(str(Path(dir) / ("*." + ext)))])
        else:
            raise ValueError("Need to specify one of list_txt or dir at least.")
        self.labels_txt = Path(labels_txt) if labels_txt is not None else None
        # self.labels = None
        # if labels_txt is not None:
        #     self.labels = read_and_parse_hier_labels(name=self.label_txt.stem, labels_txt=self.label_txt)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        name = str(self.names_list[index])
        path = path if self.ext is None else name + '.' + self.ext
        full_path = str(Path(path) if self.dir is None else self.dir / path)
        assert Path(full_path).is_file(), f'Not a file: {full_path}'
        data = self.load_func(full_path)
        if self.transform is not None:                
            data = self.transform(data)
        return data, name, full_path

    def __len__(self) -> int:
        return len(self.names_list)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def wait_until(somepredicate, timeout, period=0.25, *args, **kwargs):
    mustend = time.time() + timeout
    while time.time() < mustend:
        if somepredicate(*args, **kwargs): return True
        time.sleep(period)
    return False

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch

def parallel_process(
    inputs: Iterable, 
    fn: Callable, 
    num_workers: int = 0, 
    desc: Optional[str] = None,
    require: Optional[Literal['sharedmem']] = None,
    *args,
    **kwargs
    ) -> Iterable:
    
    # override joblib.Parallel to show progress
    # within tqdm bar https://stackoverflow.com/a/61027781
    class ProgressParallel(joblib.Parallel):
        def __call__(self, *arg, **kwarg):
            with tqdm(desc=desc) as self._pbar:
                return joblib.Parallel.__call__(self, *arg, **kwarg)

        def print_progress(self):
            self._pbar.total = self.n_dispatched_tasks
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()
    
    assert isinstance(num_workers, int), f"num_workers must be int, got {type(num_workers)}"
    root_logger = logging.getLogger()
    q = Manager().Queue()
    listener = QueueListener(q, *root_logger.handlers)  # Could also assign the handlers of a different logger here
    listener.start()
    logging.info("Starting parallel processing.")
    if num_workers <= 1:
        results = [fn(index, log_queue=q, log_level=root_logger.getEffectiveLevel(), *args, **kwargs) for index in tqdm(inputs, desc=desc)]
    else:
        results = ProgressParallel(n_jobs=num_workers, verbose=10, pre_dispatch="2*n_jobs",require=require)(
            [joblib.delayed(fn)(index, log_queue=q, log_level=root_logger.getEffectiveLevel(), *args, **kwargs) for index in inputs]
        )        
    logging.info("Finished parallel processing.")
    listener.stop()
    return results

def normalization(a: np.ndarray, p: int = 2, axis: Optional[Union[list,tuple,int]] = -1) -> np.ndarray:
    return a / np.maximum(np.sum(a**p,axis=axis,keepdims=True)**(1./p),EPSILON)


def l2_normalization(a: np.ndarray, axis: Optional[Union[list,tuple,int]] = -1) -> np.ndarray:
    return normalization(a, 2, axis)


def minmax_normalization(a: np.ndarray, axis: Optional[Union[list,tuple,int]] = -1) -> np.ndarray:
    den = a.max(axis=axis,keepdims=True)-a.min(axis=axis,keepdims=True)
    sign = np.sign(den)
    return (a - a.min(axis=axis,keepdims=True)) / (sign * np.maximum(np.abs(den),EPSILON))


def mean_normalization(a: np.ndarray, axis: Optional[Union[list,tuple,int]] = -1) -> np.ndarray:
    return (a - a.mean(axis=axis,keepdims=True)) / np.maximum(a.std(axis=axis,keepdims=True),EPSILON)


def pairwise_cosine_distance(a: np.ndarray) -> np.ndarray:
    a = l2_normalization(a, axis=-1)
    return 1. - (a @ a.T)
    # return squareform(pdist(a, metric="cosine"))

def pairwise_l2_distance(a: np.ndarray) -> np.ndarray:
    return squareform(pdist(a, metric="euclidean"))

def compute_adjacency_from_points(
    points: np.ndarray, 
    distance: Literal['cosine','euclidean','l2'] = 'cosine',
    adjacency: Literal['minmax','exp'] = 'minmax',
    thresh: Optional[float] = None,
    sigma: float = .9,
    *args,
    **kwargs
    ) -> np.ndarray:
    
    logger = kwargs.get('logger', None) or logging.getLogger()

    dist = None
    adj = None
    
    # convert to np.float64
    points = points.astype(np.float64) 
    
    # compute distance matrix
    if distance == 'cosine':
        dist = pairwise_cosine_distance(points)
    elif distance == 'euclidean' or distance == 'l2':
        dist = pairwise_l2_distance(points)
    else:
        raise ValueError(f'{distance} distance not supported.')
    
    dist[dist<=0] = 0. # floating error
    assert np.all(dist>=0), 'distance must be non-negative definite'

    # get 0-1 adjacency
    if adjacency == 'minmax':
        adj = 1. - minmax_normalization(dist, axis=None)
    elif adjacency == 'exp':
        adj = np.exp(- dist / (2*sigma**2) )
    else:
        raise ValueError(f'{adjacency} adjacency not supported.')
    
    assert np.all(adj>=0) and np.all(adj<=1), 'adjacency components must be in 0-1 interval'
    
    # apply thresholding
    if thresh is not None:
        adj = (adj >= thresh) * adj
    
    # ensure diagonal is made of zeros
    np.fill_diagonal(adj, 0.)
    
    # convert to np.float64
    adj = adj.astype(np.float64) 
    
    # scale adj to entire interval 0-1 matrix
    # adj = minmax_normalization(adj, axis=None)
    
    return adj


def compute_laplacian_from_adjacency(
    adj: np.ndarray, 
    normalization: Optional[Literal['symmetrical','randomwalk','standard']] = 'symmetrical',
    ) -> np.ndarray:
    
    # compute laplacian matrix
    lap = np.diag(adj.sum(1)) - adj
    if normalization == 'symmetrical':
        idiag2 = np.diag(1/np.maximum(adj.sum(1),EPSILON)**0.5)
        lap = idiag2 @ lap @ idiag2
    elif normalization == 'randomwalk':
        idiag = np.diag(1/np.maximum(adj.sum(1),EPSILON))
        lap = idiag @ lap
    elif normalization == 'standard':
        pass
    else:
        raise ValueError(f'{normalization} laplacian type not supported.')
        
    # avoid zeros on diagonal
    # lap += np.diag(np.ones_like(np.diag(lap))*EPSILON)
    return lap

def ncut_grouping_unfolded(
    points: np.ndarray, 
    normalization: Optional[Literal['symmetrical','randomwalk','standard']] = 'symmetrical',
    distance: Literal['cosine','euclidean','l2'] = 'cosine',
    adjacency: Literal['minmax','exp'] = 'minmax',
    overclustering: Literal['null','simultaneous','felzenszwalb','slic','quickshift','watershed','pixelkmeans','kmeans'] = 'simultaneous',
    heuristic: Literal['fiedler','maxgap'] = 'maxgap',
    scaling: Optional[Literal['unit','minmax','standard']] = 'unit',
    merging: Optional[Literal['bottomup','topdown']] = None,
    levels: int = 5,
    kover: int = 40,
    thresh: float = .0,
    sigma: float = .9,
    max_kways: int = 10,
    min_points: int = 3,
    eig_stab_th: float = .1,
    mncut_th: float = .8,
    *args,
    **kwargs
    ) -> Tuple[np.ndarray, Dict[int,np.ndarray], TreeNode]:
    
    logger = kwargs.get('logger', None) or logging.getLogger()    
    B, N, D = points.shape
    assert B > 0, f'batch size must be greater then 0, got {B}'
    batch_groups = np.zeros((B,N),dtype=np.int32)
    batch_embs = dict()
    for b in range(B):
        logger.debug(f'Processing batch {b+1}/{B}')
        groups, embs, _ = ncut_grouping(
            points=points[b],
            normalization=normalization,
            distance=distance,
            adjacency=adjacency,
            overclustering=overclustering,
            heuristic=heuristic,
            scaling=scaling,
            merging=None,
            levels=levels,
            kover=kover//5,
            thresh=thresh,
            sigma=sigma,
            max_kways=max_kways,
            min_points=min_points,
            eig_stab_th=eig_stab_th,
            mncut_th=mncut_th,
            *args,
            **kwargs
            )
        batch_embs.update({k+batch_groups.max():v for k,v in embs.items()})
        batch_groups[b]=groups+batch_groups.max()
    
    groups, embs, root = ncut_grouping(
        points=np.vstack(list(batch_embs.values())),
        normalization=normalization,
        distance=distance,
        adjacency=adjacency,
        overclustering=overclustering,
        heuristic=heuristic,
        scaling=scaling,
        merging=merging,
        levels=levels,
        kover=kover,
        thresh=thresh,
        sigma=sigma,
        max_kways=max_kways,
        min_points=min_points,
        eig_stab_th=eig_stab_th,
        mncut_th=mncut_th,
        *args,
        **kwargs
        )
    
    mapp = dict(zip(batch_embs.keys(),groups))
    batch_groups = np.vectorize(mapp.__getitem__)(batch_groups)
    return batch_groups, embs, root

def ncut_grouping(
    points: np.ndarray, 
    img: Optional[np.ndarray] = None,
    normalization: Optional[Literal['symmetrical','randomwalk','standard']] = 'symmetrical',
    distance: Literal['cosine','euclidean','l2'] = 'cosine',
    adjacency: Literal['minmax','exp'] = 'minmax',
    overclustering: Literal['null','simultaneous','felzenszwalb','slic','quickshift','watershed','pixelkmeans','kmeans'] = 'simultaneous',    heuristic: Literal['fiedler','maxgap'] = 'maxgap',
    scaling: Optional[Literal['unit','minmax','standard']] = 'unit',
    merging: Optional[Literal['bottomup','topdown']] = None,
    levels: int = 5,
    kover: int = 40,
    thresh: float = .0,
    sigma: float = .9,
    max_kways: int = 10,
    min_points: int = 3,
    eig_stab_th: float = .1,
    mncut_th: float = .8,
    max_perturb: int = 50,
    unfold: bool = False,
    stride: Optional[int] = None,
    kernel: Optional[int] = None,
    *args,
    **kwargs
    ) -> Tuple[np.ndarray, Dict[int,np.ndarray], TreeNode]:

    logger = kwargs.get('logger', None) or logging.getLogger()    
    assert max_kways > 1 , f'max_kways must be greater then 1, got {max_kways}'
    # if overclustering is not None:
        # assert (overclustering in ['felzenszwalb','slic','quickshift','watershed']) <= isinstance(img, np.ndarray), "img must be valid numpy array when overclustering is one of ['felzenszwalb','slic','quickshift','watershed']"  
    H, W, E = points.shape  
    
    num_points = H * W
    groups, adj = np.zeros(num_points, int), np.zeros((num_points,num_points), int)
    
    if num_points < 2:
        return groups, adj
    
    # flatten points spatial order
    points = np.reshape(points,(H * W, E))

    adj = compute_adjacency_from_points(
        points=points,
        distance=distance,
        adjacency=adjacency,
        thresh=thresh,
        sigma=sigma,
        logger=logger
        )
    
    logger.debug(f"Adjacency computed, now running {overclustering} overclustering..")
    
    if overclustering == 'simultaneous':
        groups = simultaneous_ncut_grouping(
            adj=adj,
            normalization=normalization,
            scaling=scaling,
            kover=kover,
            logger=logger
        )
        segments = groups
    elif overclustering == 'kmeans':
        kmeans = KMeans(n_clusters=kover, random_state=1234, n_init=10, tol=1e-6, max_iter=600, init='k-means++', algorithm='lloyd')
        segments = kmeans.fit_predict(points).astype(np.uint16)
        groups = segments
    elif overclustering == 'pixelkmeans':
        kmeans = KMeans(n_clusters=kover, random_state=1234, n_init=10, tol=1e-6, max_iter=600, init='k-means++', algorithm='lloyd')
        HH, WW, _ = img.shape
        segments = np.reshape(kmeans.fit_predict(np.reshape(img,(-1, 3))).astype(np.uint16),(HH,WW))
        groups = np.reshape(Image.fromarray(segments).resize((W,H),Image.NEAREST),(H * W,))
    elif overclustering == 'felzenszwalb':
        segments = felzenszwalb(img, scale=100, sigma=0.5, min_size=50).astype(np.uint16)
        groups = np.reshape(Image.fromarray(segments).resize((W,H),Image.NEAREST),(H * W,))
    elif overclustering == 'slic':
        segments = slic(img, n_segments=kover, compactness=2, sigma=5, start_label=1).astype(np.uint16)
        groups = np.reshape(Image.fromarray(segments).resize((W,H),Image.NEAREST),(H * W,))
    elif overclustering == 'quickshift':
        segments = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5).astype(np.uint16)
        groups = np.reshape(Image.fromarray(segments).resize((W,H),Image.NEAREST),(H * W,))
    elif overclustering == 'watershed':
        gradient = sobel(rgb2gray(img))
        segments = watershed(gradient, markers=kover, compactness=0.001).astype(np.uint16)
        groups = np.reshape(Image.fromarray(segments).resize((W,H),Image.NEAREST),(H * W,))
    elif overclustering == 'null':
        
        groups, root = recursive_kways_ncut_grouping(
            adj=adj,
            normalization=normalization,
            scaling=scaling,
            heuristic=heuristic,
            levels=levels,
            max_kways=max_kways,
            min_points=min_points,
            eig_stab_th=eig_stab_th,
            mncut_th=mncut_th,
            parent_value=0,
            max_perturb=max_perturb,
            logger=logger
            )
        
        # rename nodes so that each node has unique id
        mapp = {getattr(k,'idx'):v for k,v in root.postorder_indexing().items() if hasattr(k,'idx')}
        root.update_attribute('idx', root.postorder_indexing())
        groups = np.vectorize(mapp.__getitem__)(groups)
        
        # compute embeddings for each group
        embs = tree.pool_embeddings(root=root, points=points, attr='idx', values=groups, pooling='average', mapping=dict(), leaves_only=True)
        
        return groups, embs, root
    else:
        raise ValueError(f'{overclustering} direction not supported.')

    # run grouping algorithm
    if merging == "bottomup":
        raise ValueError(f"merging type {merging} not totally supported yet.")
        groups, root = bottom_up_merging( # only take tree
            groups=groups,
            adj=adj,
            kways=2,
            logger=logger
        )
    elif merging == "topdown":
        groups, root, flat2hier = top_down_merging(
            groups=groups,
            adj=adj,
            normalization=normalization,
            scaling=scaling,
            heuristic=heuristic,
            max_kways=max_kways,
            min_points=min_points,
            levels=levels,
            eig_stab_th=eig_stab_th,
            mncut_th=mncut_th,
            logger=logger
        )
    elif merging == None:
        pass
    else:
        raise ValueError(f"merging type {merging} not found.")
        
    
    # rename nodes so that each node has unique id
    mapp = {getattr(k,'idx'):v for k,v in root.postorder_indexing().items() if hasattr(k,'idx')}
    root.update_attribute('idx', root.postorder_indexing())
    groups = np.vectorize(mapp.__getitem__)(groups)
    
    # compute embeddings for each group
    embs = tree.pool_embeddings(root=root, points=points, attr='idx', values=groups, pooling='average', mapping=dict(), leaves_only=True)

    # groups = np.vectorize(flat2hier.__getitem__)(segments)
    # groups = np.vectorize(mapp.__getitem__)(groups)

    return groups, embs, root

def bins_min_max_ratio(eigenvector: np.ndarray, bins: int) -> float:
    hist, _ = np.histogram(eigenvector, bins=bins)
    min_value = np.min(hist)
    max_value = np.max(hist)
    return min_value / max_value

def eigendecompose_laplacian(
    laplacian: np.ndarray,
    adjacency: np.ndarray,
    topn: int,
    normalization: Optional[Literal['symmetrical','randomwalk','standard']] = None,
    ) -> Tuple[np.ndarray,np.ndarray]:
    
    if normalization == 'symmetrical':
        # cast to normalised laplacian case, standard eigensystem system D^(-1/2)*(D-L)*D^(-1/2)*z=lamb*z with z = D^(1/2)*y
        eigenvalues, eigenvectors = eigsh(laplacian, k=topn, which='SM', M=np.eye(adjacency.shape[0]))
    elif normalization == 'randomwalk':
        # cast to randomwalk laplacian case, right eigenvectors of randomwalk laplacian D^(-1)*(D-L)*y=(1-lamb)*y
        # https://yao-lab.github.io/2020.csic5011/slides/Lecture08_graph.pdf
        eigenvalues, eigenvectors = eigsh(laplacian, k=topn, which='SM')
    else:
        # cast to standard laplacian case, generalized eigenvalue system (D-L)*y=lamb*D*y
        eigenvalues, eigenvectors = eigsh(laplacian, k=topn, which='SM', M=np.diag(adjacency.sum(1)))
    
    return eigenvalues, eigenvectors


def recursive_kways_ncut_grouping_from_points(
    points: np.ndarray, 
    normalization: Optional[Literal['symmetrical','randomwalk','standard']] = None,
    scaling: Optional[Literal['unit','minmax','standard']] = None,
    heuristic: Literal['fiedler','maxgap'] = 'fiedler',
    distance: Literal['cosine','euclidean','l2'] = 'cosine',
    adjacency: Literal['minmax','exp'] = 'minmax',
    levels: int = 5,
    max_kways: Optional[int] = None,
    min_points: int = 3,
    thresh: float = .0,
    sigma: float = .9,
    parent_value: int = 0,
    parent_id: str = 'r',
    eig_stab_th: Optional[float] = None,
    mncut_th: Optional[float] = None,
    max_perturb: int = 50,
    **kwargs
    ) -> Tuple[np.ndarray, TreeNode]:
    
    logger = kwargs.get('logger', None) or logging.getLogger()
    
    num_points = points.shape[0]
    groups, root = np.zeros(num_points, int), TreeNode(id=parent_id, idx=parent_value, ncut=0.)

    if num_points < min_points or levels==0:
        return groups, root
       
    adj = compute_adjacency_from_points(
        points=points,
        distance=distance,
        adjacency=adjacency,
        thresh=thresh,
        sigma=sigma,
        logger=logger
        )
    
    groups, root = recursive_kways_ncut_grouping(
        adj=adj,
        normalization=normalization,
        scaling=scaling,
        heuristic=heuristic,
        levels=levels,
        max_kways=max_kways,
        min_points=min_points,
        eig_stab_th=eig_stab_th,
        mncut_th=mncut_th,
        parent_value=parent_value,
        max_perturb=max_perturb,
        logger=logger
        )
        
    return groups, root
   
   
def get_min_ncut(ev, adj, num_cuts):
    """Threshold an eigenvector evenly, to determine minimum ncut.

    Parameters
    ----------
    ev : array
        The eigenvector to threshold.
    adj : ndarray
        The weight matrix of the graph.
    num_cuts : int
        The number of evenly spaced thresholds to check for.

    Returns
    -------
    mask : array
        The array of booleans which denotes the bi-partition.
    mcut : float
        The value of the minimum ncut.
    """
    mcut = np.inf
    mn = ev.min()
    mx = ev.max()

    # If all values in `ev` are equal, it implies that the graph can't be
    # further sub-divided. In this case the bi-partition is the the graph
    # itself and an empty set.
    min_mask = np.zeros_like(ev, dtype=bool)
    if np.allclose(mn, mx):
        return min_mask, mcut

    # Refer Shi & Malik 2001, Section 3.1.3, Page 892
    # Perform evenly spaced n-cuts and determine the optimal one.
    thresholds = np.append(np.linspace(mn, mx, num_cuts, endpoint=False),[0.0,ev.mean()])
    for t in thresholds:
        mask = ev > t
        ind = np.eye(2)[mask.astype(int)]
        cost = np.mean(np.diag(ind.T @ (np.diag(adj.sum(1)) - adj) @ ind) / 
            np.maximum(np.diag(ind.T @ np.diag(adj.sum(1)) @ ind),EPSILON)) 
        if cost < mcut:
            min_mask = mask
            mcut = cost

    return min_mask  

def recursive_kways_ncut_grouping(
    adj: np.ndarray, 
    normalization: Optional[Literal['symmetrical','randomwalk','standard']] = None,
    scaling: Optional[Literal['unit','minmax','standard']] = None,
    heuristic: Literal['fiedler','maxgap'] = 'fiedler',
    levels: int = 5,
    max_kways: Optional[int] = None,
    min_points: int = 3,
    eig_stab_th: Optional[float] = None,
    mncut_th: Optional[float] = None,
    parent_value: int = 0,
    parent_id: str = 'r',
    max_perturb: float = 50,
    **kwargs
    ) -> Tuple[np.ndarray, TreeNode]:
    
    logger = kwargs.get('logger', None) or logging.getLogger()
    
    num_points = adj.shape[0]
    groups, root = np.zeros(num_points, int), TreeNode(id=parent_id, idx=parent_value, ncut=0.)
            
    if num_points <= min_points or levels==0:
        logger.debug('Stop: maximum depth reached.' if levels==0 else 'Stop: minimum group cardinality reached.')
        return groups, root
    
    if num_points == 2 and min_points == 1:
        ind = np.eye(2)
        ncut = (np.diag(ind.T @ (np.diag(adj.sum(1)) - adj) @ ind) / 
            np.maximum(np.diag(ind.T @ np.diag(adj.sum(1)) @ ind),EPSILON)) 
        if mncut_th:
            if 1/2*np.sum(ncut) >= mncut_th:
                return groups, root
            else:
                return (
                    np.array([0,(max_kways**(levels-1))], int), 
                    TreeNode(id=parent_id, idx=parent_value, ncut=0.,
                            children=[
                                TreeNode(id=parent_id + 'c0', idx=parent_value, ncut=ncut[0]), 
                                TreeNode(id=parent_id + 'c1',  idx=parent_value + (max_kways**(levels-1)), ncut=ncut[1])
                            ]
                        )
                    )

    # choose number of eigenvectors to extract  
    if heuristic == 'fiedler':
        max_kways = 2
    elif heuristic == 'maxgap':
        assert max_kways is not None, f'max_kways must be integer when heuristic is "maxgap", got {max_kways}'
        assert max_kways >= 2, f'max_kways must be greater than 2, got {max_kways}'
        # assume at most max_kways disconnected components
    else:
        raise ValueError(f"heuristic must be 'fiedler' or 'maxgap', got {heuristic}")
        
    lap = compute_laplacian_from_adjacency(
        adj=adj,
        normalization=normalization)

    try:
        eigenvalues, eigenvectors = eigendecompose_laplacian(lap,adj,min(max_kways, num_points - 1),normalization)
    except Exception as e:
        logger.debug('Exception: ' + str(e))
        return groups, root
    
    # choose number of eigenvectors to extract
    if heuristic == 'fiedler':
        kways = 2
    elif heuristic == 'maxgap':
        indices_by_gap = np.argsort(np.diff(eigenvalues))[::-1]
        # if there is no partition exit
        if len(indices_by_gap[indices_by_gap != 0])==0:
            return groups, root
        # remove zero and take the biggest
        index_largest_gap = indices_by_gap[indices_by_gap != 0][0]
        kways = min(max(index_largest_gap + 1, 2), min(max_kways, num_points - 1))
        logger.debug(f'{kways} partitions found with maxgap..')
    
    eigenvectors = eigenvectors[...,:kways]
    
    if eig_stab_th:
        bins = max((num_points//min_points),3)
        ratios = np.fromiter((bins_min_max_ratio(vec, bins) for vec in eigenvectors.T), float)
        if heuristic == "fiedler" and ratios[1] > eig_stab_th:
            logger.debug(f'fiedler vector is unstable at depth {levels}; it has min max {bins} bins ratios of {ratios[1]}')
            return groups, root
        elif heuristic == "maxgap" and np.any(ratios[1:] > eig_stab_th):
            kways = np.argmax(ratios[1:] > eig_stab_th, axis=0) + 1
            eigenvectors = eigenvectors[...,:kways]
            logger.debug(f'eigenvector at index {kways} is unstable at depth {levels}; it has min max {bins} bins ratios of {ratios[kways]}')
            if kways<2:
                return groups, root
            
    # if parent_value==0 and levels==6:
    #     kways=2
        
    if heuristic == 'fiedler':
        logger.debug('Semi-optimal cut partitioning..')
        groups = get_min_ncut(eigenvectors[...,1], adj, 10).astype(int)
        # groups = (eigenvectors[...,1] > 0).astype(int)
    elif heuristic == 'maxgap':
        if kways==2:
            logger.debug('Semi-optimal cut partitioning..')
            groups = get_min_ncut(eigenvectors[...,1], adj, 10).astype(int)
            # groups = (eigenvectors[...,1] > 0).astype(int)
        else:
            if scaling == 'unit':
                eigenvectors = l2_normalization(eigenvectors, axis=1)
            elif scaling == 'minmax':
                eigenvectors = minmax_normalization(eigenvectors, axis=0)
            elif scaling == 'standard':
                eigenvectors = mean_normalization(eigenvectors, axis=0)
            elif scaling is None:
                pass
            else:
                raise ValueError(f'scaling must be "unit", "minmax" or "standard", got {scaling}')

            logger.debug("Running kmeans..")
            kmeans = KMeans(n_clusters=kways, random_state=1234, n_init=10, tol=1e-6, max_iter=600, init='k-means++', algorithm='lloyd')
            groups = kmeans.fit_predict(eigenvectors)    
 
            # compute unperturbed adjacecny and convert to np.float64
            adj_star = (groups[:,None] == groups[None]).astype(np.float64) 
            # ensure diagonal is made of zeros
            np.fill_diagonal(adj_star, 0.)
            lap_star = compute_laplacian_from_adjacency(
                    adj=adj_star,
                    normalization=normalization)
            
            delta = np.diff(eigenvalues)[index_largest_gap]
            perturbed_norm = np.linalg.norm(lap_star-lap, ord='fro')
            lower_bound = np.nan_to_num(perturbed_norm / delta)
            logger.debug(f'{lower_bound} perturbation')
            if lower_bound > max_perturb:
                logger.debug(f'{lower_bound} exceeds max perturbation distance..')
                return np.zeros(num_points, int), root

    if levels > 0:
        
        if mncut_th:
            # compute mean normalized cut
            ind = np.eye(kways)[groups.copy()]
            mncut = 1/kways * np.sum(np.diag(ind.T @ (np.diag(adj.sum(1)) - adj) @ ind) / 
                np.maximum(np.diag(ind.T @ np.diag(adj.sum(1)) @ ind),EPSILON))
                        
            if mncut >= mncut_th:
                if heuristic == 'fiedler':
                    logger.debug(f'Mean ncut of {mncut} exceeded threshold of {mncut_th} for {kways} components.')
                    return np.zeros(num_points, int), root
                elif heuristic == 'maxgap':
                    while (kways-1)>=2:
                        kways -= 1
                        if kways == 2:
                            logger.debug("Running Semi-Optimal partitioning..")
                            groups = get_min_ncut(eigenvectors[...,1], adj, 10).astype(int)
                        elif kways > 2:
                            logger.debug("Running kmeans..")
                            kmeans = KMeans(n_clusters=kways, random_state=1234, n_init=10, tol=1e-6, max_iter=600, init='k-means++', algorithm='lloyd')
                            groups = kmeans.fit_predict(eigenvectors) 
 
                        else:
                            raise ValueError(f'kways must be >= 2, got {kways}')
                        # compute mean normalized cut
                        ind = np.eye(kways)[groups.copy()]
                        mncut = 1/kways * np.sum(np.diag(ind.T @ (np.diag(adj.sum(1)) - adj) @ ind) / 
                            np.maximum(np.diag(ind.T @ np.diag(adj.sum(1)) @ ind),EPSILON))
                        if mncut < mncut_th:
                            break
                        
                    if mncut >= mncut_th:
                        logger.debug(f'Mean ncut of {mncut} exceeded threshold of {mncut_th} for {kways} components.')
                        return np.zeros(num_points, int), root
                else:
                    raise ValueError('heuristic must be one of "fiedler" or "maxgap".')
        else:
            logger.debug(f'Mncut threshold is {mncut_th}.')
        mask = groups.copy()

        
        ind = np.eye(kways)[mask.copy()]
        ncut = (np.diag(ind.T @ (np.diag(adj.sum(1)) - adj) @ ind) / 
            np.maximum(np.diag(ind.T @ np.diag(adj.sum(1)) @ ind),EPSILON)) 
            
        groups = mask * (max_kways**(levels-1))
        
        for i,id in enumerate(np.unique(mask)):
            groups_child, child = recursive_kways_ncut_grouping(
                adj=adj[mask==id,:][:,mask==id],
                normalization=normalization,
                scaling=scaling,
                heuristic=heuristic,
                levels=levels-1,
                max_kways=max_kways,
                min_points=min_points,
                eig_stab_th=eig_stab_th,
                mncut_th=mncut_th,
                parent_value=parent_value + id * (max_kways**(levels-1)),
                parent_id=parent_id + 'c' + str(i),
                max_perturb=max_perturb,
                logger=logger
                )
            groups[mask==id] += groups_child
            setattr(child,'ncut',ncut[i])
            root.add_child(child)
    else:
        logger.debug(f"Reached level {levels}, exit.")

    return groups, root
   
   
    
def simultaneous_ncut_grouping_from_points(
    points: np.ndarray,
    normalization: Optional[Literal['symmetrical','randomwalk','standard']] = None,
    scaling: Optional[Literal['unit','minmax','standard']] = None,
    merging: Optional[Literal['bottomup','topdown']] = None,
    heuristic: Literal['fiedler','maxgap'] = 'fiedler',
    distance: Literal['cosine','euclidean','l2'] = 'cosine',
    adjacency: Literal['minmax','exp'] = 'minmax',
    kover: int = 40,
    max_kways: Optional[int] = None,
    thresh: float = .0,
    sigma: float = .9,
    eig_stab_th: Optional[float] = None,
    mncut_th: Optional[float] = None,
    **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    logger = kwargs.get('logger', None) or logging.getLogger()
    
    num_points = points.shape[0]
    groups, adj = np.zeros(num_points, int), np.zeros((num_points,num_points), int)
    
    if num_points < 2:
        return groups, adj
    
    adj = compute_adjacency_from_points(
        points=points,
        distance=distance,
        adjacency=adjacency,
        thresh=thresh,
        sigma=sigma,
        logger=logger
        )
    
    logger.debug("Adjacency computed, now running ncutk simultaneous overclustering..")
        
    groups = simultaneous_ncut_grouping(
        adj=adj,
        normalization=normalization,
        scaling=scaling,
        kover=kover,
        logger=logger
        )
        
    return groups, adj

def simultaneous_ncut_grouping(
    adj: np.ndarray,
    normalization: Optional[Literal['symmetrical','randomwalk','standard']] = None,
    scaling: Optional[Literal['unit','minmax','standard']] = None,
    kover: int = 40,
    **kwargs
    ) -> np.ndarray:
    
    logger = kwargs.get('logger', None) or logging.getLogger()
    
    num_points = adj.shape[0]
    groups = np.zeros(num_points, int)
    
    if num_points < 2:
        return groups

    lap = compute_laplacian_from_adjacency(
        adj=adj,
        normalization=normalization)

    try:
        _, eigenvectors = eigendecompose_laplacian(lap,adj,kover,normalization)
    except Exception as e:
        logger.debug(e)
        return groups
    
    if scaling == 'unit':
        eigenvectors = l2_normalization(eigenvectors, axis=1)
    elif scaling == 'minmax':
        eigenvectors = minmax_normalization(eigenvectors, axis=0)
    elif scaling == 'standard':
        eigenvectors = mean_normalization(eigenvectors, axis=0)

    logger.debug("Running kmeans..")
    kmeans = KMeans(n_clusters=min(kover, num_points), random_state=1234, n_init=10, tol=1e-6, max_iter=600, init='k-means++', algorithm='lloyd')
    groups = kmeans.fit_predict(eigenvectors)

    return groups

def bottom_up_merging(
    groups: np.ndarray,
    adj: np.ndarray,
    kways: int = 10,
    **kwargs
    ) -> Tuple[np.ndarray, TreeNode]:
    # Greedy Merging
    
    logger = kwargs.get('logger', None) or logging.getLogger()
    logger.debug("Running ncutk greedy merging..")
    
    groups_temp = deepcopy(groups)
    root = TreeNode(idx=0, ncut=0.)
    nodes = {v: TreeNode(idx=v, ncut=0.) for v in np.unique(groups_temp)}
    while len(nodes) > kways:
        # use temporary ordered labels
        size = len(nodes)
        unique = list(sorted(nodes.keys()))
        mapp = dict(zip(unique, range(size)))
        groups_alias = np.vectorize(mapp.__getitem__)(groups_temp)
        
        onehot_groups = np.eye(size)[groups_alias] # nodes x groups
        carry = onehot_groups.T @ adj @ onehot_groups
        cuts = carry.sum(1) - np.diag(carry)
        assoc = carry.sum(1)
        ncuts = cuts / assoc
        sum_ncuts = ncuts.sum()
        
        unique_pairs = [(i,j) for i in range(size) for j in range(i+1,size)]

        best_index = tuple((0,0))
        min_ncutk = float('inf')
        
        for index in unique_pairs:
            # ncutk = sum of ncuts - sum of individual ncuts for i and j +
            #         + new ncut considering i and j in same group
            ncutk = (sum_ncuts - (ncuts[index[0]] + ncuts[index[1]])
                + ((cuts[index[0]] + cuts[index[1]]) - (carry[index[0], index[1]] + carry[index[1], index[0]])) # type: ignore
                    / max(assoc[index[0]] + assoc[index[1]], EPSILON)) # type: ignore
            if ncutk < min_ncutk:
                best_index = index
                min_ncutk = ncutk
        
        imapp = dict(map(tuple,map(reversed, mapp.items()))) # type: ignore
        groups_temp[groups_temp==imapp[best_index[1]]] = imapp[best_index[0]] # type: ignore
        left = deepcopy(nodes.pop(imapp[best_index[0]])) # type: ignore
        right = deepcopy(nodes.pop(imapp[best_index[1]])) # type: ignore
        setattr(left, 'ncut', ncuts[best_index[0]])
        setattr(right, 'ncut', ncuts[best_index[1]])
        nodes[imapp[best_index[0]]] = TreeNode(idx=imapp[best_index[0]], children=[left, right], ncut=0.) # type: ignore
        
    root = TreeNode(idx=0, children=list(nodes.values()), ncut=0.)
    
    logger.debug("ncutk greedy merging done.")

    return groups, root

def top_down_merging(
    groups: np.ndarray,
    adj: np.ndarray,
    normalization: Optional[Literal['symmetrical','randomwalk','standard']] = None,
    scaling: Optional[Literal['unit','minmax','standard']] = None,
    heuristic: Literal['fiedler','maxgap'] = 'fiedler',
    max_kways: Optional[int] = None,
    min_points: int = 1,
    levels: int = 5,
    eig_stab_th: Optional[float] = None,
    mncut_th: Optional[float] = None,
    max_perturb: int = 50,
    **kwargs
    ) -> Tuple[np.ndarray, TreeNode, Dict]:
    # Global Recursive Cut
    
    logger = kwargs.get('logger', None) or logging.getLogger()
    logger.debug("Running global recursive cut merging..")
    
    groups_flat = deepcopy(groups)
    unique = np.unique(groups_flat)
    size = len(unique)
    flat2range = dict(zip(unique, range(size)))
    groups_range = np.vectorize(flat2range.__getitem__)(groups_flat)
    
    groups_onehot = np.eye(size)[groups_range] # nodes x groups
    adj = groups_onehot.T @ adj @ groups_onehot # nodex x nodex
    
    assert np.all(adj>=0), 'sum of weights must be positive'
    
    # ensure diagonal is made of zeros
    np.fill_diagonal(adj, 0.)
    
    # adj must be a 0-1 matrix
    # adj = minmax_normalization(adj, axis=None)
    
    # convert to np.float64
    adj = adj.astype(np.float64) 
    
    groups_hier, root = recursive_kways_ncut_grouping(
        adj=adj,
        normalization=normalization,
        scaling=scaling,
        heuristic=heuristic,
        levels=levels,
        max_kways=max_kways,
        min_points=min_points,
        eig_stab_th=eig_stab_th,
        mncut_th=mncut_th,
        parent_value=0,
        logger=logger,
        max_perturb=max_perturb
    ) # aggregated segments
    range2hier = dict(zip(range(size),groups_hier))
    groups = np.vectorize(range2hier.__getitem__)(groups_range) # original points

    flat2hier = dict(zip(unique,groups_hier))
    logger.debug("global recursive cut merging done.")

    return groups, root, flat2hier


def softmax(logits: np.ndarray, temp: Optional[float] = 1.):
    exp = np.exp(logits / temp)
    return exp/np.sum(exp,axis=-1,keepdims=True)
class Task:
    def __init__(self, func: Callable, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        
    def run(self, device: str):
        if len(self.args):
            self.args = [arg.to(device) for arg in self.args]
        if len(self.kwargs):
            self.kwargs = {k:v.to(device) for k,v in self.kwargs.items()}
        return self.func(*self.args, **self.kwargs).cpu()

class WorkerWithDevice(mp.Process):
    def __init__(self, queue: mp.Queue, return_queue: mp.Queue, device: str):
        self.device = device
        super().__init__(target=self.work, args=(queue, return_queue, ))

    def work(self, queue, return_queue):
        while queue.qsize() > 0:
            task = queue.get()
            return_queue.put(task.run(self.device))
            del task

class MatchingIoU:

    def __init__(
            self, 
            num_classes: int,
            hung_match_freq: Optional[Literal['sample','dataset']] = None,
            *args, **kwargs) -> None:
        self.num_classes = num_classes
        self.hung_match_freq = hung_match_freq

        # Variable for accumulating frequencies
        self.mtp = np.zeros(shape=(self.num_classes,self.num_classes), dtype=np.int64)
        self.mfp = np.zeros(shape=(self.num_classes,self.num_classes), dtype=np.int64)
        self.mfn = np.zeros(shape=(self.num_classes,self.num_classes), dtype=np.int64)

    def compute_stats(self, sparse_true: np.ndarray, sparse_pred: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        true_pos = sparse_true.T @ sparse_pred
        false_pos = (1-sparse_true).T @ sparse_pred
        false_neg = sparse_true.T @ (1-sparse_pred)
        return true_pos, false_pos, false_neg
    
    def compute_multilabel_iou(self, true_pos: np.ndarray, false_pos: np.ndarray, false_neg: np.ndarray) -> np.ndarray:
        return np.nan_to_num(true_pos / (true_pos + false_pos + false_neg), nan=0, posinf=0, neginf=0)

    def max_match(self, true_pos: np.ndarray, false_pos: np.ndarray, false_neg: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        from scipy.optimize import linear_sum_assignment
        n_true, n_pred = true_pos.shape
        cost_matrix = self.compute_multilabel_iou(true_pos, false_pos, false_neg)
        assert cost_matrix.shape[0] == self.num_classes, 'class axis shape mismatch'
        dim = max(n_true, n_pred)
        assert cost_matrix.max() <= 1. and cost_matrix.min() >= 0., f'cost matrix must be in [0,1], got [{cost_matrix.min()},{cost_matrix.max()}]'
        cost_matrix_padded = np.pad(cost_matrix, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
        assert cost_matrix_padded.shape[0] == cost_matrix_padded.shape[1], 'cost matrix must be square'
        # best_rows, best_cols = linear_sum_assignment(cost_matrix=cost_matrix_padded,maximize=True)
        best_rows, best_cols = np.arange(cost_matrix_padded.shape[0]), np.argmax(cost_matrix_padded,1)
        best_rows, best_cols = best_rows[:self.num_classes], best_cols[:self.num_classes]
        assert np.all(best_rows==np.arange(self.num_classes)), 'order mismatch'
        true_pos_padded = np.pad(true_pos, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
        false_pos_padded = np.pad(false_pos, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
        false_neg_padded = np.pad(false_neg, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
        best_true_pos = true_pos_padded[best_rows,:][:, best_cols]
        best_false_pos = false_pos_padded[best_rows,:][:, best_cols]
        best_false_neg = false_neg_padded[best_rows,:][:, best_cols]
        # pick best prediction for ground truth class and 
        # do not consider false positives (all the other predictions that do not intersect)
        present_true_mask = (best_true_pos + best_false_neg) > 0
        best_false_pos *= (present_true_mask + present_true_mask.T)
        # update only present categories
        mapping = np.array(list(zip(best_rows, best_cols)))
        return best_true_pos, best_false_pos, best_false_neg, mapping
        
    def update_state(self, sparse_true: np.ndarray, sparse_pred: np.ndarray) -> Dict[str,np.ndarray]:
        assert sparse_true.shape[-1] == self.num_classes, 'class axis shape mismatch'
        result = {}
        
        # Compute stats for current sample
        mtp, mfp, mfn = self.compute_stats(sparse_true, sparse_pred)
        assert mtp.shape[0] == self.num_classes, 'class axis shape mismatch'
        assert mfp.shape[0] == self.num_classes, 'class axis shape mismatch'
        assert mfn.shape[0] == self.num_classes, 'class axis shape mismatch'

        # Compute per sample best matching 
        if self.hung_match_freq == 'sample':
            # instance-based hungarian matching IHM
            mtp, mfp, mfn, mapping = self.max_match(mtp, mfp, mfn)
            result['iou_ihm'] = np.nan_to_num(np.diag(mtp) / (np.diag(mtp) + np.diag(mfp) + np.diag(mfn)))
            valid = np.nan_to_num((np.diag(mtp) + np.diag(mfn)) / (np.diag(mtp) + np.diag(mfn)).sum()) > 0
            result['hnecovering'] = np.nanmean(result['iou_ihm'][valid])
            result['nfcovering'] = np.nanmean(self.compute_multilabel_iou(mtp, mfp, mfn).max(1)[valid])
            result['mapping'] = mapping
        
        assert mtp.shape[1] == self.num_classes, 'class axis shape mismatch'
        assert mfp.shape[1] == self.num_classes, 'class axis shape mismatch'
        assert mfn.shape[1] == self.num_classes, 'class axis shape mismatch'

        # Accumulate stats
        self.mtp += mtp.astype(self.mtp.dtype)
        self.mfp += mfp.astype(self.mfp.dtype)
        self.mfn += mfn.astype(self.mfn.dtype)

        # Compute sample stats for current full dataset best match
        if self.hung_match_freq == 'dataset':
            *_, mapping = self.max_match(self.mtp, self.mfp, self.mfn)
            n_true, n_pred = mtp.shape
            dim = max(n_true, n_pred)
            best_rows, best_cols = mapping[:,0], mapping[:,1]
            true_pos_padded = np.pad(mtp, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
            false_pos_padded = np.pad(mfp, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
            false_neg_padded = np.pad(mfn, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
            mtp = true_pos_padded[best_rows,:][:, best_cols]
            mfp = false_pos_padded[best_rows,:][:, best_cols]
            mfn = false_neg_padded[best_rows,:][:, best_cols]
            # return miou for current sample
            result['iou_hm'] = np.nan_to_num(np.diag(mtp) / (np.diag(mtp) + np.diag(mfp) + np.diag(mfn)))
            valid = np.nan_to_num((np.diag(mtp) + np.diag(mfn)) / (np.diag(mtp) + np.diag(mfn)).sum()) > 0
            result['miou'] = np.nanmean(result['iou_hm'][valid])
            result['mapping'] = mapping
        elif self.hung_match_freq is None:
            result['iou'] = np.nan_to_num(np.diag(mtp) / (np.diag(mtp) + np.diag(mfp) + np.diag(mfn)))
            valid = np.nan_to_num((np.diag(mtp) + np.diag(mfn)) / (np.diag(mtp) + np.diag(mfn)).sum()) > 0
            result['miou'] = np.nanmean(result['iou'][valid])  
        elif self.hung_match_freq == "sample":
            pass
        else:
            raise ValueError(f'Invalid hung_match_freq {self.hung_match_freq}')
              
        return result

    def result(self) -> Dict[str,np.ndarray]:
        '''adapted from https://github.com/kazuto1011/deeplab-pytorch/blob/master/libs/utils/metric.py'''
        # Compute per sample best matching 
        mtp, mfp, mfn, mapping = self.mtp, self.mfp, self.mfn, {}
        if self.hung_match_freq == 'dataset':
            mtp, mfp, mfn, mapping = self.max_match(mtp, mfp, mfn)

        iou = np.nan_to_num(np.diag(mtp) / (np.diag(mtp) + np.diag(mfp) + np.diag(mfn)))
        freq = np.nan_to_num((np.diag(mtp) + np.diag(mfn)) / (np.diag(mtp) + np.diag(mfn)).sum())
        valid = freq > 0
        return dict(
            mapping = mapping,
            freq = freq,
            #valid = valid,
            #mtp = np.diag(mtp),
            #mfp = np.diag(mfp),
            #mfn = np.diag(mfn),
            iou = iou,
            miou = np.nanmean(iou[valid]),
            pacc = np.nan_to_num((np.diag(mtp))[valid].sum() / (np.diag(mtp)+np.diag(mfn))[valid].sum()),
            macc = np.nanmean(np.nan_to_num((np.diag(mtp)) / (np.diag(mtp)+np.diag(mfn)))[valid]),
            fwiou = (freq[valid] * iou[valid]).sum()
        )

    def reset_state(self) -> None:
        self.mtp = np.zeros(shape=(self.num_classes, self.num_classes), dtype=np.int64)
        self.mfp = np.zeros(shape=(self.num_classes, self.num_classes), dtype=np.int64)
        self.mfn = np.zeros(shape=(self.num_classes, self.num_classes), dtype=np.int64)


class MatchingHierarchicalIoU:

    def __init__(
            self, 
            labels_hierarchy: TreeNode,
            match_type: Literal['max','hungarian'] = 'max',
            threshold: float = 0.5,
            *args, **kwargs) -> None:
        self.labels_hierarchy = labels_hierarchy
        self.match_type = match_type
        self.threshold = threshold
        nodes_fg = tree.subtree_from_nodes_list(tree.search(self.labels_hierarchy,'type', CatType.FOREGROUND)).nodes_list()
        nodes_fg.remove(self.labels_hierarchy) # remove root node
        self.indexing = {n: i for i,n in enumerate(nodes_fg)}
        self.labels = ["--".join(getattr(an,'name') for an in node.dinasty()[1:] if hasattr(an,'name')) for node in nodes_fg]
        leaves_bg = tree.search(self.labels_hierarchy, 'type', CatType.BACKGROUND)
        if len(leaves_bg) > 0:
            nodes_bg = tree.subtree_from_nodes_list(leaves_bg).nodes_list()
            nodes_bg = list(set(nodes_bg).difference(nodes_fg)) # remove intermediate node that are in fg
            self.indexing = {**{n: 0 for n in nodes_bg},**{k: v+1 for k,v in self.indexing.items()}}
            self.labels = ["background"] + self.labels
        nodes_ign = [x.prune() for x in tree.search(self.labels_hierarchy, 'type', CatType.IGNORE)]
        self.ign_idx = [getattr(x,'idx') for x in nodes_ign if hasattr(x,'idx')]
        self.num_classes = len(set(list(self.indexing.values())))
        # Variable for accumulating frequencies
        self.mtp = np.zeros(shape=(self.num_classes,), dtype=np.int64)
        self.mfp = np.zeros(shape=(self.num_classes,), dtype=np.int64)
        self.mfn = np.zeros(shape=(self.num_classes,), dtype=np.int64)

    def compute_stats(self, sparse_true: np.ndarray, sparse_pred: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        true_pos = sparse_true.T @ sparse_pred
        false_pos = (1-sparse_true).T @ sparse_pred
        false_neg = sparse_true.T @ (1-sparse_pred)
        return true_pos, false_pos, false_neg
    
    def compute_multilabel_iou(self, true_pos: np.ndarray, false_pos: np.ndarray, false_neg: np.ndarray) -> np.ndarray:
        return np.nan_to_num(true_pos / (true_pos + false_pos + false_neg), nan=0, posinf=0, neginf=0)

    def max_match(self, true_pos: np.ndarray, false_pos: np.ndarray, false_neg: np.ndarray) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        from scipy.optimize import linear_sum_assignment
        n_true, n_pred = true_pos.shape
        cost_matrix = self.compute_multilabel_iou(true_pos, false_pos, false_neg)
        dim = max(n_true, n_pred)
        assert cost_matrix.max() <= 1. and cost_matrix.min() >= 0., f'cost matrix must be in [0,1], got [{cost_matrix.min()},{cost_matrix.max()}]'
        cost_matrix_padded = np.pad(cost_matrix, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
        assert cost_matrix_padded.shape[0] == cost_matrix_padded.shape[1], 'cost matrix must be square'
        if self.match_type == 'hungarian':
            best_rows, best_cols = linear_sum_assignment(cost_matrix=cost_matrix_padded,maximize=True)
        else:
            best_rows, best_cols = np.arange(cost_matrix_padded.shape[0]), np.argmax(cost_matrix_padded,1)
        best_rows, best_cols = best_rows[:n_true], best_cols[:n_true]
        assert np.all(best_rows==np.arange(n_true)), 'order mismatch'
        true_pos_padded = np.pad(true_pos, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
        false_pos_padded = np.pad(false_pos, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
        false_neg_padded = np.pad(false_neg, pad_width=((0,max(0,dim-n_true)),(0,max(0,dim-n_pred))), mode='constant', constant_values=0)
        best_true_pos = true_pos_padded[best_rows,:][:, best_cols]
        best_false_pos = false_pos_padded[best_rows,:][:, best_cols]
        best_false_neg = false_neg_padded[best_rows,:][:, best_cols]
        # pick best prediction for ground truth class and 
        # do not consider false positives (all the other predictions that do not intersect)
        present_true_mask = (best_true_pos + best_false_neg) > 0
        best_false_pos *= (present_true_mask + present_true_mask.T)
        # update only present categories
        mapping = np.array(list(zip(best_rows, best_cols)))
        return best_true_pos, best_false_pos, best_false_neg, mapping
        
    def update_state(self, true: np.ndarray, pred: np.ndarray, h_pred: TreeNode) -> Dict[str,np.ndarray]:
        
        assert np.all(true.shape==pred.shape), "shape mismatch!"
        h,w = true.shape
        valid = np.reshape(np.logical_not(np.in1d(true, self.ign_idx)),(-1,))
        # use _id attribute for mapping to same indexing as self.indexing

        h_true = tree.subtree_from_nodes_list(tree.search(self.labels_hierarchy,'idx',np.unique(true).tolist(),lambda x,y: x in y))
        back_nodes = tree.search(h_true,'type', CatType.BACKGROUND)
        if len(back_nodes) > 0:
            h_true_back = tree.subtree_from_nodes_list(back_nodes)
            _, sparse_true_back = tree.mask_subtree_nodes_leaves_in_values(h_true_back, 'idx', true)
            sparse_true_back = np.sum(sparse_true_back,axis=0,keepdims=True).astype(bool)

        fore_nodes = tree.search(h_true,'type', CatType.FOREGROUND)
        if len(fore_nodes) == 0: return {}
        h_true_fore = tree.subtree_from_nodes_list(fore_nodes)
        map_true, sparse_true = tree.mask_subtree_nodes_leaves_in_values(h_true_fore, 'idx', true)
        
        if len(back_nodes) > 0:
            map_true[len(sparse_true)] = back_nodes[0].prune()
            h_true_fore.add_child(map_true[len(sparse_true)], override_id=False)
            sparse_true = np.concatenate([sparse_true, sparse_true_back], axis=0)
        # else:
        #     sparse_true = np.concatenate([sparse_true, np.fill(true.shape, False)])
        #     map_true[len(sparse_true)] = back_nodes[0]
                    
        sparse_true = sparse_true.reshape((-1,h*w)).T[valid].astype(int)
        
        # take only possible subtrees to reduce computations
        map_pred, sparse_pred = tree.mask_subtree_nodes_leaves_in_values(h_pred, 'idx', pred)
        sparse_pred = sparse_pred.reshape((-1,h*w)).T[valid].astype(int)
        
        result = {}
        
        # Compute stats for current sample
        mtp, mfp, mfn = self.compute_stats(sparse_true, sparse_pred)

        # Compute per sample best matching 
        # instance-based matching IHM
        mtp, mfp, mfn, mapping = self.max_match(mtp, mfp, mfn)

        map_true_reach, pair_to_reach_true = tree.pairwise_oriented_reachability(h_true_fore)
        map_pred_reach, pair_to_reach_pred = tree.pairwise_oriented_reachability(h_pred)
        
        reach_true=np.eye(len(map_true_reach)).astype(bool)
        reach_true[*np.array(list(pair_to_reach_true.keys())).T]=np.array(list(pair_to_reach_true.values()))
        reach_pred=np.eye(len(map_pred_reach)).astype(bool)
        reach_pred[*np.array(list(pair_to_reach_pred.keys())).T]=np.array(list(pair_to_reach_pred.values()))

        imap_true_reach = dict((v, k) for k, v in map_true_reach.items())
        imap_pred_reach = dict((v, k) for k, v in map_pred_reach.items())

        permute_true = [imap_true_reach[map_true[idx]] for idx in mapping[:,0]]
        permute_pred = [imap_pred_reach[map_pred[idx]] for idx in mapping[:,1]]
        
        # wrong_hier = np.unique(np.where(np.logical_and(np.logical_not(reach_pred[permute_pred][:,permute_pred]),reach_true[permute_true][:,permute_true]))[1])
        hier_weight = (1-(np.logical_and(np.logical_not(reach_pred[permute_pred][:,permute_pred]),reach_true[permute_true][:,permute_true])).sum(0)/reach_true[permute_true][:,permute_true].sum(0))
        mtp, mfp, mfn = np.diag(mtp), np.diag(mfp), np.diag(mfn)
        result['iou_m'] = np.nan_to_num(mtp / (mtp + mfp + mfn))
        result['iou_m'] *= (result['iou_m'] >= self.threshold)
        valid = np.nan_to_num((mtp + mfn) / (mtp + mfn).sum()) > 0
        result['nhcovering'] = np.nanmean((result['iou_m']*hier_weight)[valid])
        result['nmcovering'] = np.nanmean(result['iou_m'][valid])
        result['mapping'] = mapping
        
        # Accumulate stats
        idxs = np.where(valid)[0]
        gidxs = [self.indexing[map_true[idx]] for idx in idxs]
            
        self.mtp[gidxs] += mtp.astype(self.mtp.dtype)[idxs]
        self.mfp[gidxs] += mfp.astype(self.mfp.dtype)[idxs]
        self.mfn[gidxs] += mfn.astype(self.mfn.dtype)[idxs]

        return result

    def result(self) -> Dict[str,np.ndarray]:
        '''adapted from https://github.com/kazuto1011/deeplab-pytorch/blob/master/libs/utils/metric.py'''
        # Compute per sample best matching 
        mtp, mfp, mfn = self.mtp, self.mfp, self.mfn
        iou = np.nan_to_num(mtp / (mtp + mfp + mfn))
        iou *= iou>=self.threshold
        freq = np.nan_to_num((mtp + mfn) / (mtp + mfn).sum())
        valid = freq > 0
        return {**dict(
            thresh=self.threshold,
            freq = freq,
            miou = np.nanmean(iou[valid]),
            pacc = np.nan_to_num(mtp[valid].sum() / (mtp+mfn)[valid].sum()),
            macc = np.nanmean(np.nan_to_num(mtp / (mtp+mfn))[valid]),
            fwiou = (freq[valid] * iou[valid]).sum(),
        ),**dict(zip(self.labels, iou))}

    def reset_state(self) -> None:
        self.mtp = np.zeros(shape=(self.num_classes,), dtype=np.int64)
        self.mfp = np.zeros(shape=(self.num_classes,), dtype=np.int64)
        self.mfn = np.zeros(shape=(self.num_classes,), dtype=np.int64)