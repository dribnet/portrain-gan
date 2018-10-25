import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse

from torch.utils.serialization import load_lua
from torch.legacy.nn import SpatialFullConvolution
from PIL import Image
import torchvision.utils as vutils
from torch.autograd import Variable

# This example was intentd optimize the latent code
# based on a traget image using optim.LBFGS
#
# however, I couldn't figure out how to enable gradients
# through the lua model. the error is:
#
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
#
# (compare to reverse_from_latent, which works and is near identical)

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

# this was a failed attempt to recursively require grad in the model
def requires_grad(m):
    if type(m) is list:
        for x in m:
            requires_grad(x)
    else:
        m.requires_grad = True

def main():
    parser = argparse.ArgumentParser(description="Decode latents from art-DCGAN's Portrait GAN")
    parser.add_argument('--outfile', default="outfile.png", help='image file to save')
    parser.add_argument('--seed', type=int, default=None, help='optional random seed')
    parser.add_argument('--steps', type=int, default=10, help='number of iterations to optimize')
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    im = np.array(Image.open("belamy128.png"))
    target = torch.from_numpy((im.transpose(2, 0, 1) / 127.5) - 1.0).float().cuda()
    target = Variable(target, requires_grad=False)

    model = load_torch_model("portrait_584_net_G_cpu.t7")

    # note - this doens't seem to work. but was worth a try
    for x in model.parameters():
        requires_grad(x)
    # switch to train mode
    model.training()

    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS([target], lr=0.8)

    nz = 100
    reference_noise = torch.randn(64, nz, 1, 1).cuda()
    fixed_noise = torch.randn(1, nz, 1, 1).cuda()

    for i in range(args.steps):
        print('STEP: ', i)
        def closure():
            optimizer.zero_grad()
            reference_noise[0] = fixed_noise
            out_batch = model.forward(reference_noise)
            out = out_batch[0]
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)

    tarray = ((0.5 + 0.5 * target.detach().cpu().numpy()) * 256).astype(np.uint8)
    im = Image.fromarray(tarray.transpose(1, 2, 0))
    im.save("reversed_image.png")

if __name__ == "__main__":
    main()
