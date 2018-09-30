def make_fake_noise(xp, batch_size, noise_size):
    noise = xp.random.normal(size=(batch_size, noise_size))
    noise = noise.astype(xp.float32)
    return noise


def make_fake_label(xp, batch_size, n_labels):
    c_fake_true = xp.random.randint(n_labels, size=(batch_size,))
    c_fake_true = c_fake_true.astype(xp.int32)
    return c_fake_true


def to_one_hot(xp, n_labels, y, dtype='f'):
    one_hot = xp.eye(n_labels, dtype=dtype)[y]
    return one_hot
