import numpy as np


LEN_SIGNAl = 1024


def read_file(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            remove_dirst_str = line.replace("[", "")
            remove_next_str = remove_dirst_str.replace("]", "")
            data.append(remove_next_str.split(", "))

    data_float_format = []
    for item in data:
        data_float_format.append([float(x) for x in item])

    new_data = np.asarray(data_float_format)
    data = np.reshape(new_data, (new_data.shape[1] // LEN_SIGNAl, LEN_SIGNAl))
    return data[0]
