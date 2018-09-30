import chainer


class Cifar10Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, split='train'):
        train, test = chainer.datasets.get_cifar10()
        if split == 'train':
            self.examples = train
        elif split == 'test':
            self.examples = test
        else:
            raise ValueError('split must be either "train" or "test".')

    def __len__(self):
        return len(self.examples)

    def get_example(self, i):
        img, label = self.examples[i]
        return img, label
