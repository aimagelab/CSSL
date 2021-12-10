To run the experiments:
  - export $PYTHONPATH=\<ROOT DIR OF THIS REPO\>
  - `python utils/main.py` (+ args)
  - argument `lpc` (labels per class) specifies how many labels are not masked (leave it empty for full supervision)

For example:
 - `python utils/main.py --n_epochs=50 --model=ccic --dataset=seq-cifar10 --lr=0.001 --batch_size=32 --buffer_size=500 --minibatch_size=32 --alpha=0.5 --lamda=0.5 --k=3 --memory_penalty=1 --k_aug=3 --sharp_temp=0.5 --mixup_alpha=0.75`
  
