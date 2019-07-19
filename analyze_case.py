# Juan Salamanca and Santiago Nunez-Corrales
# Social viscosity
import scipy.optimize as scop
import pandas as pd
import numpy as np
import glob
import sys


def get_reversions(dframe: pd.DataFrame):
    reversions = 0
    for i in range(1, len(dframe[1]) - 1):
        if dframe[1][i] > dframe[1][i - 1]:
            reversions += 1

    return reversions


def jmak(t, k, n):
    return np.exp(-k * np.power(t, n))


def get_curve_fit(dframe: pd.DataFrame):
    maxv = dframe[1].max()
    rescaled = dframe[1]/maxv
    params, cov = scop.curve_fit(jmak, dframe[0], rescaled)
    return params[0], params[1]


def get_run_data(filename:str, m: int):
    dataset = pd.read_csv(filename, header=None)
    firstN = dataset.head(m)
    k, n = get_curve_fit(firstN)
    revs = get_reversions(firstN)

    outcome = np.array([
        firstN[1][0],
        firstN[1][m - 1],
        firstN[1].max(),
        firstN[1].min(),
        k,
        n,
        revs
    ])

    return outcome


def compute_stored_runs(directory:str, m: int, output:str):
    rows = []

    filenames = glob.glob(directory + "/run*.csv")

    print(filenames)

    for file in filenames:
        new_row = get_run_data(file, m)
        rows.append(new_row)

    outcome = pd.DataFrame(rows, columns=['IV', 'FV', 'MaxV', 'MinV', 'K', 'N', 'REV'])
    outcome.to_csv(directory + '/' + output + '.csv', header=None)

    return outcome


def process_case(directory: str, m: int, output: str):
    runs = compute_stored_runs(directory, m, output)
    ivm = runs['IV'].mean()
    ivs = runs['IV'].std()
    fvm = runs['FV'].mean()
    fvs = runs['FV'].std()
    Mvm = runs['MaxV'].mean()
    Mvs = runs['MaxV'].std()
    mvm = runs['MinV'].mean()
    mvs = runs['MinV'].std()
    km = runs['K'].mean()
    ks = runs['K'].std()
    nm = runs['N'].mean()
    ns = runs['N'].std()
    rm = runs['REV'].mean()
    rs = runs['REV'].std()

    print(f'Iv: {ivm:.2f} ({ivs:.2f})\nFv: {fvm:.2f} ({fvs:.2f})\nMv: {Mvm:.2f} ({Mvs:.2f})\nmv: {mvm:.2f} ({mvs:.2f})\nk: {km:.2E} ({ks:.2E})\nn: {nm:.2f} ({ns:.2f})\nR: {rm:.2f} ({rs:.2f})')

    return np.array([
        ivm, ivs,
        fvm, fvs,
        Mvm, Mvs,
        mvm, mvs,
        km, ks,
        nm, ns,
        rm ,rs
    ])


if __name__ == "__main__":
    process_case(sys.argv[1], int(sys.argv[2]), sys.argv[3])
