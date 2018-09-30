import os

import chainer
import numpy as np
from PIL import Image


def sample_generate(gen, noise_gen, rows=10, cols=10, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        n_imgs = rows * cols
        xp = noise_gen.xp

        # set seed to generate same noise array
        xp.random.seed(seed)
        # generate noise
        noise = noise_gen.generate_noise(n_imgs)
        c_fake = xp.repeat(xp.arange(noise_gen.n_labels), rows)
        c_fake_one_hot = noise_gen.to_one_hot(c_fake)
        z = xp.concatenate([noise, c_fake_one_hot], axis=1)

        # reset seed
        xp.random.seed()

        with chainer.using_config('train', False), chainer.using_config(
                'enable_backprop', False):
            x = gen(z)
        x = chainer.backends.cuda.to_cpu(x.data)

        x = np.asarray(np.clip(x * 255.0, 0.0, 255.0), dtype=np.uint8)
        _, _, h, w = x.shape
        x = x.reshape((rows, cols, 3, h, w))
        x = x.transpose(0, 3, 1, 4, 2)
        x = x.reshape((rows * h, cols * w, 3))

        preview_dir = '{}/preview'.format(trainer.out)
        preview_path = '{}/image{:0>8}.png'.format(
            preview_dir, trainer.updater.iteration)
        os.makedirs(preview_dir, exist_ok=True)
        Image.fromarray(x).save(preview_path)

    return make_image
