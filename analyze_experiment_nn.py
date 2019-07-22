# Juan Salamanca and Santiago Nunez-Corrales
# Social viscosity
from statsmodels.stats.multicomp import MultiComparison
from matplotlib.font_manager import FontProperties
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import statsmodels.api as sm
import analyze_case as ac
import pandas as pd
import numpy as np
import sys


def run_all(topdir: str, m: int, prfx: str):
    neighbors = ["5", "10", "15", "20"]
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    names = ['IVM', 'IVS', 'FVM', 'FVS', 'MaxVM', 'MaxVS', 'MinVM', 'MinVS',
             'KM', 'KS', 'NM', 'NS', 'RM', 'RS', 'TOL', 'NNN']
    rows = []

    for n in neighbors:
        for tol in tolerances:
            casedir = topdir + '/' + 'nn' + '_' + tol + '_' + n
            casesum = np.append(ac.process_case(casedir, m, prfx + '_' + n + '_' + tol), [float(tol), float(n)])
            rows.append(casesum)

    df = pd.DataFrame(rows, columns=names)

    return df


def plot_ivm_fvm(df: pd.DataFrame):
    fontP = FontProperties()
    fontP.set_size('small')
    # Plot one line per case
    df_5 = df[df['NNN'] == 5]
    df_10 = df[df['NNN'] == 10]
    df_15 = df[df['NNN'] == 15]
    df_20 = df[df['NNN'] == 20]

    # Initial values
    plt.errorbar(df_5['TOL'], df_5['IVM'], yerr=df_5['IVS']/6, fmt='-', color= 'blue', ecolor='blue', elinewidth=1, capsize=1, label='Initial [NNN = 5]')
    plt.errorbar(df_10['TOL'], df_10['IVM'], yerr=df_10['IVS'] / 6, fmt='--', color='blue', ecolor='blue', elinewidth=1,
                 capsize=1, label='Initial [NNN = 10]')
    plt.errorbar(df_15['TOL'], df_15['IVM'], yerr=df_15['IVS'] / 6, fmt='-.', color='blue', ecolor='blue', elinewidth=1,
                 capsize=1, label='Initial [NNN = 15]')
    plt.errorbar(df_20['TOL'], df_20['IVM'], yerr=df_20['IVS'] / 6, fmt=':', color='blue', ecolor='blue', elinewidth=1,
                capsize=1, label='Initial [NNN = 20]')


    # Final values
    plt.errorbar(df_5['TOL'], df_5['FVM'], yerr=df_5['FVS']/6, fmt='-', color='red', ecolor='red', elinewidth=1, capsize=1, label='Final [NNN = 5]')
    plt.errorbar(df_10['TOL'], df_10['FVM'], yerr=df_10['FVS'] / 6, fmt='--', color='red', ecolor='red', elinewidth=1,
                 capsize=1, label='Final [NNN = 10]')
    plt.errorbar(df_15['TOL'], df_15['FVM'], yerr=df_15['FVS'] / 6, fmt='-.', color='red', ecolor='red', elinewidth=1,
                 capsize=1, label='Final [NNN = 15]')
    plt.errorbar(df_20['TOL'], df_20['FVM'], yerr=df_20['FVS'] / 6, fmt=':', color='red', ecolor='red', elinewidth=1,
                 capsize=1, label='Final [NNN = 20]')

    plt.xlabel('Tolerance')
    plt.ylabel('Global viscosity')
    #plt.legend(loc='center right', prop=fontP)
    plt.savefig('nnn_ivm_fvm.eps', format='eps', dpi=300)
    plt.show()


def plot_Mvm_mvm(df: pd.DataFrame):
    fontP = FontProperties()
    fontP.set_size('small')
    # Plot one line per case
    df_5 = df[df['NNN'] == 5]
    df_10 = df[df['NNN'] == 10]
    df_15 = df[df['NNN'] == 15]
    df_20 = df[df['NNN'] == 20]

    # Initial values
    plt.errorbar(df_5['TOL'], df_5['MaxVM'], yerr=df_5['MaxVS']/6, fmt='-', color= 'blue', ecolor='blue', elinewidth=1, capsize=1, label='Maximum [NNN = 5]')
    plt.errorbar(df_10['TOL'], df_10['MaxVM'], yerr=df_10['MaxVS'] / 6, fmt='--', color='blue', ecolor='blue', elinewidth=1,
                 capsize=1, label='Maximum [NNN = 10]')
    plt.errorbar(df_15['TOL'], df_15['MaxVM'], yerr=df_15['MaxVS'] / 6, fmt='-.', color='blue', ecolor='blue', elinewidth=1,
                 capsize=1, label='Maximum [NNN = 15]')
    plt.errorbar(df_20['TOL'], df_20['MaxVM'], yerr=df_20['MaxVS'] / 6, fmt=':', color='blue', ecolor='blue', elinewidth=1,
                capsize=1, label='Maximum [NNN = 20]')


    # Final values
    plt.errorbar(df_5['TOL'], df_5['MinVM'], yerr=df_5['MinVS']/6, fmt='-', color='red', ecolor='red', elinewidth=1, capsize=1, label='Minimum [NNN = 5]')
    plt.errorbar(df_10['TOL'], df_10['MinVM'], yerr=df_10['MinVS'] / 6, fmt='--', color='red', ecolor='red', elinewidth=1,
                 capsize=1, label='Minimum [NNN = 10]')
    plt.errorbar(df_15['TOL'], df_15['MinVM'], yerr=df_15['MinVS'] / 6, fmt='-.', color='red', ecolor='red', elinewidth=1,
                 capsize=1, label='Minimum [NNN = 15]')
    plt.errorbar(df_20['TOL'], df_20['MinVM'], yerr=df_20['MinVS'] / 6, fmt=':', color='red', ecolor='red', elinewidth=1,
                 capsize=1, label='Minimum [NNN = 20]')

    plt.xlabel('Tolerance')
    plt.ylabel('Global viscosity')
    #plt.legend(loc='center right', prop=fontP)
    plt.savefig('nnn_Mvm_mvm.eps', format='eps', dpi=300)
    plt.show()


def plot_func(df: pd.DataFrame, m: int):
    df_5 = df[df['NNN'] == 5]
    df_10 = df[df['NNN'] == 10]
    df_15 = df[df['NNN'] == 15]
    df_20 = df[df['NNN'] == 20]

    times = np.arange(0.0, 450)
    dfpar_5 = df_5[['TOL', 'KM','NM']]
    dfpar_10 = df_10[['TOL', 'KM', 'NM']]
    dfpar_15 = df_15[['TOL', 'KM', 'NM']]
    dfpar_20 = df_20[['TOL', 'KM', 'NM']]
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']

    cols = ['black', 'blue', 'green', 'grey', 'gold', 'red']

    for index, row in dfpar_5.iterrows():
        curve = ac.jmak(times, row['KM'], row['NM'])
        plt.plot(times, curve, label=f"Tol: {tolerances[index%6]} (k={row['KM']:.2E}, n={row['NM']:.2f})", color=cols[index%6])

    plt.xlabel('Discrete time')
    plt.ylabel('Global viscosity')
    plt.legend(loc='upper right')
    plt.savefig('nnn_functions_5.eps', format='eps', dpi=300)
    plt.show()

    for index, row in dfpar_10.iterrows():
        curve = ac.jmak(times, row['KM'], row['NM'])
        plt.plot(times, curve, label=f"Tol: {tolerances[index%6]} (k={row['KM']:.2E}, n={row['NM']:.2f})", color=cols[index%6])

    plt.xlabel('Discrete time')
    plt.ylabel('Global viscosity')
    plt.legend(loc='upper right')
    plt.savefig('nnn_functions_10.eps', format='eps', dpi=300)
    plt.show()

    print(dfpar_15)

    for index, row in dfpar_15.iterrows():
        curve = ac.jmak(times, row['KM'], row['NM'])
        plt.plot(times, curve, label=f"Tol: {tolerances[index%6]} (k={row['KM']:.2E}, n={row['NM']:.2f})", color=cols[index%6])

    plt.xlabel('Discrete time')
    plt.ylabel('Global viscosity')
    plt.legend(loc='upper right')
    plt.savefig('nnn_functions_15.eps', format='eps', dpi=300)
    plt.show()

    for index, row in dfpar_20.iterrows():
        curve = ac.jmak(times, row['KM'], row['NM'])
        plt.plot(times, curve, label=f"Tol: {tolerances[index%6]} (k={row['KM']:.2E}, n={row['NM']:.2f})", color=cols[index%6])

    plt.xlabel('Discrete time')
    plt.ylabel('Global viscosity')
    plt.legend(loc='upper right')
    plt.savefig('nnn_functions_20.eps', format='eps', dpi=300)
    plt.show()


def plot_reversions(df: pd.DataFrame):
    # Plot one line per case
    df_5 = df[df['NNN'] == 5]
    df_10 = df[df['NNN'] == 10]
    df_15 = df[df['NNN'] == 15]
    df_20 = df[df['NNN'] == 20]
    plt.errorbar(df_5['TOL'], df_5['RM'], yerr=df_5['RS'] / 6, fmt='-',
                 color='blue', ecolor='blue', elinewidth=1, capsize=1, label='NNN = 5')
    plt.errorbar(df_10['TOL'], df_10['RM'], yerr=df_10['RS'] / 6, fmt='--',
                 color='blue', ecolor='blue', elinewidth=1, capsize=1, label='NNN = 10')
    plt.errorbar(df_15['TOL'], df_15['RM'], yerr=df_15['RS'] / 6, fmt='-.',
                 color='blue', ecolor='blue', elinewidth=1, capsize=1, label='NNN = 15')
    plt.errorbar(df_20['TOL'], df_20['RM'], yerr=df_20['RS'] / 6, fmt=':',
                 color='blue', ecolor='blue', elinewidth=1, capsize=1, label='NNN = 20')
    plt.xlabel('Tolerance')
    plt.ylabel('Reversions')
    plt.legend(loc='upper right')
    plt.savefig('nnn_revs.eps', format='eps', dpi=300)
    plt.show()


def compute_anova_rev_no_inter(topdir: str, m: int):
    # Assemble a large experiment table with all data
    neighbors = ["5", "10", "15", "20"]
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    dfs = []

    for n in neighbors:
        for tol in tolerances:
            casedir = topdir + '/' + 'nn' + '_' + tol + '_' + n
            casetable = ac.compute_stored_runs(casedir, m, None)
            casetable['TOL'] = [ float(tol) ] * 5
            casetable['NNN'] = [ float(n) ] * 5
            dfs.append(casetable)

    df = pd.concat(dfs).reset_index(drop=True)

    # Perform a regression with the data
    results = ols('REV ~ C(TOL) + C(NNN)', data=df).fit()
    print(results.summary())
    print('\n\n\n')
    aov_table = sm.stats.anova_lm(results, typ=2)
    print(aov_table)
    print('\n\n\n')
    mct = MultiComparison(df['REV'], df['TOL'])
    mct_results = mct.tukeyhsd()
    print(mct_results)

    mcn = MultiComparison(df['REV'], df['NNN'])
    mcn_results = mcn.tukeyhsd()
    print(mcn_results)


def compute_anova_rev_inter(topdir: str, m: int):
    # Assemble a large experiment table with all data
    neighbors = ["5", "10", "15", "20"]
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    dfs = []

    for n in neighbors:
        for tol in tolerances:
            casedir = topdir + '/' + 'nn' + '_' + tol + '_' + n
            casetable = ac.compute_stored_runs(casedir, m, None)
            casetable['TOL'] = [ float(tol) ] * 5
            casetable['NNN'] = [ float(n) ] * 5
            dfs.append(casetable)

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_csv('nnn_to_manova.csv')

    # Perform a regression with the data
    results = ols('REV ~ C(TOL) + C(NNN) + C(TOL):C(NNN)', data=df).fit()
    print(results.summary())
    print('\n\n\n')
    aov_table = sm.stats.anova_lm(results, typ=2)
    print(aov_table)
    print('\n\n\n')
    mct = MultiComparison(df['REV'], df['TOL'])
    mct_results = mct.tukeyhsd()
    print(mct_results)

    mcn = MultiComparison(df['REV'], df['NNN'])
    mcn_results = mcn.tukeyhsd()
    print(mcn_results)


def compute_anova_rev_restrict_n(topdir: str, m: int):
    # Assemble a large experiment table with all data
    neighbors = ["5", "10", "15", "20"]
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    dfs = []

    for n in neighbors:
        for tol in tolerances:
            casedir = topdir + '/' + 'nn' + '_' + tol + '_' + n
            casetable = ac.compute_stored_runs(casedir, m, None)
            casetable['TOL'] = [ float(tol) ] * 5
            casetable['NNN'] = [ float(n) ] * 5
            dfs.append(casetable)

    dfa = pd.concat(dfs).reset_index(drop=True)
    df = dfa[dfa['NNN'] != 5]

    # Perform a regression with the data
    results = ols('REV ~ C(TOL) + C(NNN) + C(TOL):C(NNN)', data=df).fit()
    print(results.summary())
    print('\n\n\n')
    aov_table = sm.stats.anova_lm(results, typ=2)
    print(aov_table)
    print('\n\n\n')
    mct = MultiComparison(df['REV'], df['TOL'])
    mct_results = mct.tukeyhsd()
    print(mct_results)

    mcn = MultiComparison(df['REV'], df['NNN'])
    mcn_results = mcn.tukeyhsd()
    print(mcn_results)

def compute_anova_rev_restrict_t(topdir: str, m: int):
    # Assemble a large experiment table with all data
    neighbors = ["5", "10", "15", "20"]
    tolerances = ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0']
    dfs = []

    for n in neighbors:
        for tol in tolerances:
            casedir = topdir + '/' + 'nn' + '_' + tol + '_' + n
            casetable = ac.compute_stored_runs(casedir, m, None)
            casetable['TOL'] = [ float(tol) ] * 5
            casetable['NNN'] = [ float(n) ] * 5
            dfs.append(casetable)

    dfa = pd.concat(dfs).reset_index(drop=True)
    df = dfa[dfa['TOL'] != 1.0]

    # Perform a regression with the data
    results = ols('REV ~ C(TOL) + C(NNN) + C(TOL):C(NNN)', data=df).fit()
    print(results.summary())
    print('\n\n\n')
    aov_table = sm.stats.anova_lm(results, typ=2)
    print(aov_table)
    print('\n\n\n')
    mct = MultiComparison(df['REV'], df['TOL'])
    mct_results = mct.tukeyhsd()
    print(mct_results)

    mcn = MultiComparison(df['REV'], df['NNN'])
    mcn_results = mcn.tukeyhsd()
    print(mcn_results)


def main(directory: str, m: int, output: str):
    # Compute anovas
    compute_anova_rev_inter(directory, m)
    compute_anova_rev_no_inter(directory, m)
    compute_anova_rev_restrict_n(directory, m)
    compute_anova_rev_restrict_t(directory, m)

    # Obtain the descriptive dataset
    df = run_all(directory, m, output)

    # Plot all graphs
    plot_ivm_fvm(df)
    plot_Mvm_mvm(df)
    plot_func(df, m)
    plot_reversions(df)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])

