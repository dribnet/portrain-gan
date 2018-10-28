Portrait-GAN
============

![five example image samples](five_samples.png?raw=true "samples taken from model")

Usage:

```bash
python download.py # downlaod the model
python decode.py   # randomly sample model, files saves to outfile.png
python decode.py --latent latent1.txt --outfile latent1.png # generate from saved
```

Bonus: non-working image encoder
================================

I attempted to build a GAN encoder that would go from
image to latent using torch.optim.LBFGS and gradient information.

As a warmup I did implement `reverse_from_latent.py`, which
will read an input image (eg: `belamy128.png`) and then change that
image to match a prexisting latent via LBFGS.

From that version I made `reverse_from_image.py`. The only
change is that disabled gradients back to the image and instead
tried to enable graidents through the model which (I thought)
would optimize the latent to match the image. But each time
I run that version I get:

```bash
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

I'm not sure if the lua model does not have the gradient information
I need or if I am doing something else wrong. If anyone wants to
take a crack at fixing this, it would be great to have this as 
a reference version of a simple GAN image to latent encoder.

