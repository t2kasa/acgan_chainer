class NoiseGenerator:
    def __init__(self, xp, noise_size=100, n_labels=10):
        self.xp = xp
        self.noise_size = noise_size
        self.n_labels = n_labels

    def generate_noise(self, batch_size):
        xp = self.xp
        noise = xp.random.normal(size=(batch_size, self.noise_size))
        noise = noise.astype(xp.float32)
        return noise

    def generate_label(self, batch_size):
        xp = self.xp
        label = xp.random.randint(self.n_labels, size=(batch_size,))
        label = label.astype(xp.int32)
        return label

    def to_one_hot(self, label, dtype='f'):
        xp = self.xp
        one_hot = xp.eye(self.n_labels, dtype=dtype)[label]
        return one_hot
