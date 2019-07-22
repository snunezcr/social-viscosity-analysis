# Juan Salamanca and Santiago Nunez-Corrales
# Social viscosity
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.multivariate.manova import MANOVA
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
import analyze_case as ac
import pandas as pd
import numpy as np
import sys


def run_all(topdir: str, m: int, prfx: str):
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    names = ['IVM', 'IVS', 'FVM', 'FVS', 'MaxVM', 'MaxVS', 'MinVM', 'MinVS',
             'KM', 'KS', 'NM', 'NS', 'RM', 'RS']
    rows = []

    for tol in tolerances:
        casedir = topdir + '/' + 'all' + '_' + tol
        casesum = ac.process_case(casedir, m, prfx + '_' + tol)
        rows.append(casesum)

    df = pd.DataFrame(rows, columns=names)
    df['TOL'] = tolerances

    return df


def plot_ivm_fvm(df: pd.DataFrame):
    plt.errorbar(df['TOL'], df['IVM'], yerr=df['IVS']/6, fmt='-',
                 color= 'blue', ecolor='blue', elinewidth=1, capsize=1, label='Initial')
    plt.errorbar(df['TOL'], df['MinVM'], yerr=df['MinVS']/6, fmt='-',
                 color='red', ecolor='red', elinewidth=1, capsize=1, label='Final')
    plt.xlabel('Tolerance')
    plt.ylabel('Global viscosity')
    plt.legend(loc='center right')
    plt.savefig('all_ivm_fvm.eps', format='eps', dpi=300)
    plt.show()


def plot_func(df: pd.DataFrame, m: int):
    times = np.arange(0.0, 450)
    dfpar = df[['TOL', 'KM','NM']]
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']

    cols = ['black', 'blue', 'green', 'grey', 'gold', 'red']

    for index, row in dfpar.iterrows():
        curve = ac.jmak(times, row['KM'], row['NM'])
        plt.plot(times, curve, label=f"Tol: {tolerances[index]} (k={row['KM']:.2E}, n={row['NM']:.2f})", color=cols[index])

    plt.xlabel('Discrete time')
    plt.ylabel('Global viscosity')
    plt.legend(loc='upper right')
    plt.savefig('all_functions.eps', format='eps', dpi=300)
    plt.show()


def plot_reversions(df: pd.DataFrame):
    plt.errorbar(df['TOL'], df['RM'], yerr=df['RS'] / 6, fmt='-',
                 color='blue', ecolor='blue', elinewidth=1, capsize=1, label='All-to-all')
    plt.xlabel('Tolerance')
    plt.ylabel('Reversions')
    plt.legend(loc='upper right')
    plt.savefig('all_revs.eps', format='eps', dpi=300)
    plt.show()


def compute_anova_rev(topdir: str, m: int):
    # Assemble a large experiment table with all data
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    dfs = []

    for tol in tolerances:
        casedir = topdir + '/' + 'all' + '_' + tol
        casetable = ac.compute_stored_runs(casedir, m, None)
        casetable['TOL'] = [ float(tol) ] * 5
        dfs.append(casetable)

    df = pd.concat(dfs).reset_index(drop=True)

    # Perform a regression with the data
    results = ols('REV ~ C(TOL)', data=df).fit()
    print(results.summary())
    print('\n\n\n')
    aov_table = sm.stats.anova_lm(results, typ=2)
    print(aov_table)
    print('\n\n\n')
    mc = MultiComparison(df['REV'], df['TOL'])
    mc_results = mc.tukeyhsd()
    print(mc_results)


def compute_manova_cvg(topdir: str, m: int):
    # Assemble a large experiment table with all data
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    dfs = []

    for tol in tolerances:
        casedir = topdir + '/' + 'all' + '_' + tol
        casetable = ac.compute_stored_runs(casedir, m, None)
        casetable['TOL'] = [float(tol)] * 5
        dfs.append(casetable)

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv('all_to_manova.csv')

    # Perform a regression with the data
    endog = np.asarray(df[['K', 'N']])
    exog = np.asarray(df[['TOL']])

    mod = MANOVA.from_formula('K + N ~ TOL', data=df)
    print(mod)
    result = mod.mv_test()
    print(result)
    return mod


def main(directory: str, m: int, output: str):
    # Compute anova
    compute_anova_rev(directory, m)
    compute_manova_cvg(directory, m)

    # Obtain the descriptive dataset
    df = run_all(directory, m, output)

    # Plot all graphs
    plot_ivm_fvm(df)
    plot_func(df, m)
    plot_reversions(df)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])

