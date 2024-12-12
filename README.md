# Unsupervised Hierarchy-Agnostic Segmentation: Parsing Semantic Image Structure
1. install ```requirements.txt``` line by line,
2. setup datasets paths in ```config/datasets.json```, such as image and annotation sources for each dataset set,
3. run experiments in ```config``` directory:
   ```bash
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/voc12.json # run voc12 NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/coco_stuff.json # run coco_stuff NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/coco_things.json # run coco_thing NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/cityscapes.json # run cityscapes NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/partimagenet.json # run partimagenet NMCovering experiment
   torchrun --nnodes 1 --nproc_per_node 4 --conf configs/pascalpart.json # run pascalpart NMCovering experiment
   ```
4. results are stored in the newly created ```data``` folder.
