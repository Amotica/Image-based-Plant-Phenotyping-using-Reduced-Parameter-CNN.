import numpy as np


def memory_usage(model, batch_size=32):
    shapes_count = int(np.sum([np.prod(np.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))
    memory = shapes_count * 4  # Take the number of values, multiply by 4 to get the raw number of bytes (since every floating point is 4 bytes, or maybe by 8 for double precision
    gbytes = np.round(((memory / (1024.0 ** 3))*batch_size), 3)
    return gbytes
