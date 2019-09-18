def bilinear_interpolation():
    pass


def centered_scale_simplified(w1, w2, x):
    return (w1 * (2 * x + 1) - w2) / (2 * w2)

def related_scale(w1, w2, x, r=True):
    q = (2 * w1 * (2 * x + 1) - w2) / (2*w2)
    if not r:
        return min(q, w1 - 2)
    return min(round(q), w1 - 2)