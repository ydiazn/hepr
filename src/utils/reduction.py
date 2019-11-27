import numpy as np 

def average_first_eight_coeficients(cover_work, size):
    dimension = cover_work.shape[0] // size
    num_blocks = dimension ** 2
    num_coeficients = size ** 2

    blocks = [
        np.reshape(
            cover_work[
                b // dimension * size: b // dimension * size + size,
                b % dimension * size: b % dimension * size + size
            ],
            (1, num_coeficients)
        ) for b in range(num_blocks)
    ]

    return (np.sum(blocks, axis=0).reshape(-1)[1:size + 1]) / num_blocks