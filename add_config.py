import sys
import yaml
import torch

ckpt = sys.argv[1]
config = sys.argv[2]
out_ckpt = sys.argv[3]

ckpt = torch.load(ckpt, map_location='cpu')

with open(config, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

ckpt['config'] = config
torch.save(ckpt, out_ckpt)

