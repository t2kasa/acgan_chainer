from argparse import ArgumentParser

import chainer
from chainer.optimizers import Adam
from chainer.training import extensions

from acgan_updater import ACGANUpdater
from cifar10_dataset import Cifar10Dataset
from extensions import sample_generate
from models import Generator, Discriminator
from noise_generator import NoiseGenerator


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--n_labels', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--out', type=str, default='results/result_acgan')
    # Adam arguments
    parser.add_argument('--alpha', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # config
    dataset = Cifar10Dataset(split='train')
    train_iter = chainer.iterators.SerialIterator(dataset, args.batch_size)

    gen = Generator()
    dis = Discriminator(args.n_labels)

    opts = {'opt_gen': Adam(args.alpha, args.beta1, args.beta2).setup(gen),
            'opt_dis': Adam(args.alpha, args.beta1, args.beta2).setup(dis)}
    updater_args = {'iterator': {'main': train_iter}, 'device': args.device,
                    'models': [gen, dis], 'optimizer': opts}

    if 0 <= args.device:
        chainer.backends.cuda.get_device_from_id(args.device).use()
        gen.to_gpu()
        dis.to_gpu()

    noise_gen = NoiseGenerator(gen.xp, n_labels=args.n_labels)
    updater = ACGANUpdater(noise_gen, **updater_args)
    trainer = chainer.training.Trainer(updater, (args.max_iter, 'iteration'),
                                       out=args.out)

    # setup logging
    snapshot_interval = (args.max_iter, 'iteration')
    sample_interval = (1000, 'iteration')
    display_interval = (10, 'iteration')

    trainer.extend(extensions.snapshot_object(
        gen, 'gen_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_{.updater.iteration}.npz'), trigger=snapshot_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))
    log_keys = ["iteration", "loss_dis", "loss_gen"]
    trainer.extend(extensions.LogReport(log_keys, trigger=display_interval))
    trainer.extend(extensions.PrintReport(log_keys), trigger=display_interval)
    trainer.extend(sample_generate(gen, noise_gen), trigger=sample_interval)
    trainer.extend(extensions.PlotReport(
        ['loss_gen', 'loss_dis'], 'iteration', file_name='loss.png'))

    trainer.run()


if __name__ == '__main__':
    main()
