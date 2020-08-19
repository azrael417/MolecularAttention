python train.py -pb --rotate --mae -p custom -t 1 \
  -b 512 --epochs 100 --amp O2 -w 1 -r 0 --depth 4 \
  --width 128 --nheads 0 --dropout_rate 0.2  --lr 6e-5 \
  -o 3cl/model.pt -i 3cl/smiles.smi \
  --precomputed_images 3cl/images.npy --precomputed_values 3cl/values.npy \
   --no_pretrain
