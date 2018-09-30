import os

import chainer
import numpy as np

from utils import make_fake_noise, to_one_hot
from PIL import Image


def sample_generate(gen, rows=10, cols=10, seed=0):
    @chainer.training.make_extension()
    def make_image(trainer):
        np.random.seed(seed)
        n_imgs = rows * cols
        xp = gen.xp

        noise_size = 100
        n_labels = 10

        noise = make_fake_noise(xp, n_imgs, noise_size)
        c_fake = np.repeat(np.arange(n_labels), rows)
        c_fake_one_hot = to_one_hot(xp, n_labels, c_fake)
        z = xp.concatenate([noise, c_fake_one_hot], axis=1)

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
