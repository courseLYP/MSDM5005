def denorm(x):
    # TANH [-1, 1]
    out = (x + 1) / 2
    return out.clamp(0, 1)