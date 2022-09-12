# Continual Semi-Supervised Learning through Contrastive Interpolation Consistency

Official code implementation for "Continual Semi-Supervised Learning through Contrastive Interpolation Consistency" - Accepted at [Pattern Recognition Letters 2022](https://www.sciencedirect.com/science/article/pii/S0167865522002458)  

DOI: `10.1016/j.patrec.2022.08.006`  
Volume: 162

## Usage 
To run the experiments:
  - export $PYTHONPATH=\<ROOT DIR OF THIS REPO\>
  - `python utils/main.py` (+ args)
  - argument `lpc` (labels per class) specifies how many labels are not masked (leave it empty for full supervision)

For example:
 - `python utils/main.py --n_epochs=50 --model=ccic --dataset=seq-cifar10 --lr=0.001 --batch_size=32 --buffer_size=500 --minibatch_size=32 --alpha=0.5 --lamda=0.5 --k=3 --memory_penalty=1 --k_aug=3 --sharp_temp=0.5 --mixup_alpha=0.75`

## Cite this work
Please use the following citation if you intend to use this work.

```
@article{boschini2022continual,
  title = {Continual semi-supervised learning through contrastive interpolation consistency},
  journal = {Pattern Recognition Letters},
  volume = {162},
  pages = {9-14},
  year = {2022},
  issn = {0167-8655},
  doi = {https://doi.org/10.1016/j.patrec.2022.08.006},
  url={https://arxiv.org/abs/2108.06552},
  author = {Boschini, Matteo and Buzzega, Pietro and Bonicelli, Lorenzo and Porrello, Angelo and Calderara, Simone},
}
```