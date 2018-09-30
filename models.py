import chainer
import chainer.functions as F
import chainer.links as L


class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__()
        ch = 512
        init = {'initialW': chainer.initializers.Normal(0.02)}
        with self.init_scope():
            # (b, 110) => (b, ch * 4 * 4)
            self.fc1 = chainer.Sequential(
                L.Linear(ch * 4 * 4, **init),
                F.relu
            )
            # (b, ch, 4, 4) => (b, ch // 2, 8, 8)
            self.tconv2 = chainer.Sequential(
                L.Deconvolution2D(ch, ch // 2, 4, 2, 1, **init),
                L.BatchNormalization(ch // 2),
                F.relu
            )
            # (b, ch // 2, 8, 8) => (b, ch // 4, 16, 16)
            self.tconv3 = chainer.Sequential(
                L.Deconvolution2D(ch // 2, ch // 4, 4, 2, 1, **init),
                L.BatchNormalization(ch // 4),
                F.relu
            )
            # (b, ch // 4, 16, 16) => (b, ch // 8, 32, 32)
            self.tconv4 = chainer.Sequential(
                L.Deconvolution2D(ch // 4, ch // 8, 4, 2, 1, **init),
                L.BatchNormalization(ch // 8),
                F.relu
            )
            # (b, ch // 8, 32, 32) => (b, 3, 32, 32)
            self.tconv5 = chainer.Sequential(
                L.Deconvolution2D(ch // 8, 3, 3, 1, 1, **init),
                F.sigmoid
            )

    def __call__(self, z):
        batch_size = z.shape[0]

        h = self.fc1(z)
        h = h.reshape(batch_size, -1, 4, 4)
        for i in range(2, 5 + 1):
            h = self['tconv{}'.format(i)](h)
        return h


class Discriminator(chainer.Chain):
    def __init__(self, n_labels):
        super(Discriminator, self).__init__()
        use_gamma = False
        ch = 512
        init = {'initialW': chainer.initializers.Normal(0.02)}
        with self.init_scope():
            self.conv1 = chainer.Sequential(
                L.Convolution2D(3, ch // 8, 3, 1, 1, **init),
                F.leaky_relu,
            )
            self.conv2 = chainer.Sequential(
                L.Convolution2D(ch // 8, ch // 4, 4, 2, 1, **init),
                L.BatchNormalization(ch // 4, use_gamma=use_gamma),
                F.leaky_relu,
                L.Convolution2D(ch // 4, ch // 4, 3, 1, 1, **init),
                L.BatchNormalization(ch // 4, use_gamma=use_gamma),
                F.leaky_relu,
            )
            self.conv3 = chainer.Sequential(
                L.Convolution2D(ch // 4, ch // 2, 4, 2, 1, **init),
                L.BatchNormalization(ch // 2, use_gamma=use_gamma),
                F.leaky_relu,
                L.Convolution2D(ch // 2, ch // 2, 3, 1, 1, **init),
                L.BatchNormalization(ch // 2, use_gamma=use_gamma),
                F.leaky_relu,
            )
            self.conv4 = chainer.Sequential(
                L.Convolution2D(ch // 2, ch // 1, 4, 2, 1, **init),
                L.BatchNormalization(ch // 1, use_gamma=use_gamma),
                F.leaky_relu,
                L.Convolution2D(ch // 1, ch // 1, 3, 1, 1, **init),
                L.BatchNormalization(ch // 1, use_gamma=use_gamma),
                F.leaky_relu,
            )
            self.fc_dis = L.Linear(1, **init)
            self.fc_aux = L.Linear(n_labels, **init)

    def __call__(self, h):
        for i in range(1, 4 + 1):
            h = self['conv{}'.format(i)](h)
        d = self.fc_dis(h)
        c = self.fc_aux(h)
        return d, c
