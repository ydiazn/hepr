import fire
import numpy as np


def main(file1, file2, *args, f1_col, f2_col, output, **kwargs):
    data1 = np.loadtxt(file1, usecols=(f1_col))
    data2 = np.loadtxt(file2, usecols=(f2_col))

    data = np.concatenate((data1, data2)).reshape(-1, 1)
    groups = np.concatenate((
        np.zeros(data1.size, dtype=np.uint8),
        np.ones(data2.size, dtype=np.uint8),
    )).reshape(-1, 1)
    ouput_data = np.concatenate((data, groups), axis=1)

    np.savetxt(output, ouput_data)


if __name__ == "__main__":
    fire.Fire(main)
