import chainer
import chainer.functions as F

from utils import make_fake_noise, make_fake_label, to_one_hot


class ACGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        super(ACGANUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        n_labels = 10
        n_noise = 100
        gen_optimizer = self.get_optimizer('opt_gen')
        dis_optimizer = self.get_optimizer('opt_dis')
        xp = self.gen.xp

        batch = self.get_iterator('main').next()
        x_real, c_real_true = self.converter(batch, device=self.device)
        x_real = chainer.as_variable(x_real)
        c_real_true = chainer.as_variable(c_real_true)
        batch_size = len(batch)

        noise = make_fake_noise(xp, batch_size, n_noise)
        c_fake_true = make_fake_label(xp, batch_size, n_labels)
        c_fake_true_one_hot = to_one_hot(xp, n_labels, c_fake_true)

        z = chainer.as_variable(
            xp.concatenate([noise, c_fake_true_one_hot], axis=1))

        # D(x_data)
        d_real, c_real_pred = self.dis(x_real)
        # D(G(z))
        x_fake = self.gen(z)
        d_fake, c_fake_pred = self.dis(x_fake)

        loss_dis = compute_loss_dis(d_real, d_fake)
        loss_dis += F.softmax_cross_entropy(c_real_pred, c_real_true)
        loss_dis += F.softmax_cross_entropy(c_fake_pred, c_fake_true)

        loss_gen = compute_loss_gen(d_fake)
        loss_gen += F.softmax_cross_entropy(c_fake_pred, c_fake_true)

        # update G
        self.gen.cleargrads()
        loss_gen.backward()
        gen_optimizer.update()
        x_fake.unchain_backward()

        # update D
        self.dis.cleargrads()
        loss_dis.backward()
        dis_optimizer.update()

        chainer.reporter.report({'loss_gen': loss_gen, 'loss_dis': loss_dis})


def compute_loss_dis(y_real, y_fake):
    return F.mean(F.softplus(-y_real)) + F.mean(F.softplus(y_fake))


def compute_loss_gen(y_fake):
    return F.mean(F.softplus(-y_fake))
