import torch
import numpy as np
import os
import sys
import argparse

from torch.utils.serialization import load_lua
from torch.legacy.nn import SpatialFullConvolution
from PIL import Image

def replace_module(module, check_fn, create_fn):
    if not hasattr(module, 'modules'):
        return
    if module.modules is None:
        return
    for i in range(len(module.modules)):
        m = module.modules[i]
        if check_fn(m):
            module.modules[i] = create_fn(m)
        replace_module(m, check_fn, create_fn)

def fix_full_conv(m):
    m.finput = None
    m.fgradInput = None
    m.bias = None
    return m

def load_torch_model(path):
    model = load_lua(path, unknown_classes=True)
    replace_module(
        model,
        lambda m: isinstance(m, SpatialFullConvolution),
        fix_full_conv
    )
    return model

batch_size = 100
nz = 100

def main():
    parser = argparse.ArgumentParser(description="Decode latents from art-DCGAN's Portrait GAN")
    parser.add_argument('--outfile', default="outfile.png", help='image file to save')
    parser.add_argument('--seed', type=int, default=None, help='optional random seed')
    parser.add_argument('--latent', default=None, help='file with latent vector')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    saved_latent = None
    if args.latent is not None:
        print("reading latent variable from {}".format(args.latent))
        with open(args.latent) as f:
            vector = f.readlines()
        vector = [float(x.strip()) for x in vector]
        saved_latent = torch.FloatTensor(vector).reshape(nz, 1, 1).cuda()

    if not os.path.exists("portrait_584_net_G_cpu.t7"):
        print("Please use download.py to download the model first")
        return

    print("Loading model")
    model = load_torch_model("portrait_584_net_G_cpu.t7").cuda()
    print("Model loading done")

    z_batch = torch.randn(batch_size, nz, 1, 1).cuda()
    if saved_latent is not None:
        z_batch[0] = saved_latent
    print("Running model")
    out_batch = model.forward(z_batch)
    first_image = out_batch[0].detach()
    tarray = ((0.5 + 0.5 * first_image.cpu().view(3,128,128).numpy()) * 256).astype(np.uint8)
    im = Image.fromarray(tarray.transpose(1, 2, 0))
    print("Done, saving file: {}".format(args.outfile))
    im.save(args.outfile)

if __name__ == "__main__":
    main()
