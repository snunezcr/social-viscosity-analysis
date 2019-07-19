# University of Illinois at Urbana-Champaign
# Computation of agent viscosity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


def viscous_force(x_arr, y_arr):
    jrk_x = np.diff(x_arr, n=3)
    jrk_y = np.diff(y_arr, n=3)

    return jrk_x**2 + jrk_y**2


def directional_diff(x_arr, y_arr):
    vel_x = np.diff(x_arr, n=1)[:-2]
    acc_x = np.diff(x_arr, n=2)[:-1]
    vel_y = np.diff(y_arr, n=1)[:-2]
    acc_y = np.diff(y_arr, n=2)[:-1]

    return vel_x*acc_x + vel_y*acc_y


def reynolds(x_arr, y_arr):
    # Avoid division by zero
    patched_dir_diff = directional_diff(x_arr, y_arr) + np.full((len(directional_diff(x_arr, y_arr)),), 0.01)
    return viscous_force(x_arr, y_arr) / patched_dir_diff


def process_dataset(filename):
    dataset = pd.read_csv(filename)
    values = dataset.to_numpy().transpose()

    # We obtain the number of instants and create two matrices, one for viscous force magnitudes
    # and one for its Reynolds numbers
    instants = values.shape[1]
    vf_values = np.empty((0, instants - 3))
    rn_values = np.empty((0, instants - 3))

    # First column is always the time instants
    for i in range(1, 40, 2):
        x = values[i]
        y = values[i + 1]
        vf = np.vstack(viscous_force(x, y)).transpose()
        rn = np.vstack(reynolds(x, y)).transpose()
        vf_values = np.vstack((vf_values, vf))
        rn_values = np.vstack((rn_values, rn))

    return np.arange(0, instants-3), np.mean(vf_values, axis=0), np.mean(rn_values, axis=0)


def main(filename):
    t, vf, rn = process_dataset(filename)

    plt.plot(t, vf, 'b-')
    plt.show()

    plt.plot(t, rn, 'r-')
    plt.show()


if __name__ == "__main__":
    main(sys.argv[1])


