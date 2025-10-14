import scipy.optimize as opt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

def ComputeSimplex(c, Aeq, beq, b=None):

    res = opt.linprog(c, A_eq = Aeq, b_eq = beq, bounds = b, method='highs')

    if res.success:
        return res
    else:
        raise ValueError("Linear programming problem could not be solved.")
    

def PartA():
    # A. Run your code (or tool) on the example LP problem instance from the lecture on Simplex (n=4, m=2)
    c = np.array([7, 4, 6, 1])
    A = np.array([[1, 2, -1, -1], [-1, -5, 2, 3]])
    b = np.array([1, 1])

    solution = ComputeSimplex(c, A, b)
    print("Optimal solution for Part A:", solution.x)

def random_feasible_lp(n, m, U=100.0, rng=None):
    rng = np.random.default_rng(rng)
    x0 = rng.uniform(0, U, size=n)

    A_eq = rng.uniform(-10, 10, size=(m, n))
    b_eq = A_eq @ x0  # ensures feasibility
    c = rng.uniform(-10, 10, size=n)

    bounds = [(0.0, U)] * n
    return c, A_eq, b_eq, bounds

def avg_time(func, *args, n_trials=20):
    times = []
    for _ in range(n_trials):
        start = time.time()
        func(*args)
        end = time.time()
        times.append(end - start)
    return np.mean(times)

def PartB(n_list=[2, 10, 20, 30, 40, 50], m_list=[2, 6, 10, 14], lp_trials=1):
    '''
    B. Next try to increase the number of variables (n) and increase the number of constraints (m),
    thus:

    - Fixing m=2: Rerun the code after increasing n from 4 to 10 to 50. (in increments of 10) and n
    - For each of the n values above: Rerun the code after increasing m from 2 to 6 to 10 to 14. (in
    increments of 4, just create additional constraints)

    '''

    ret = pd.DataFrame(columns=['n', 'm', 'Time elapsed (s)', 'Number of pivots', 'Cost'])

    for n in n_list:
        for m in m_list:

            pivots = []
            times = []
            cost = 0
            for _ in range(lp_trials):
                while True:
                    # Generate random LP problem
                    c, A, b, bounds = random_feasible_lp(n, m)

                    try:
                        solution = ComputeSimplex(c, A, b, bounds)
                        elapsed = avg_time(ComputeSimplex, c, A, b, bounds)

                        pivots.append(solution.nit)
                        times.append(elapsed)
                        cost = solution.fun

                        break
                    except ValueError as e:
                        print(f"Failed to solve LP for n={n}, m={m}. Retrying...")
 
            ret.loc[len(ret)] = [n, m, np.average(times), np.average(pivots), cost]
            
    return ret

def graph_params_vs_n_or_m(df, const_val, const_var='m', filepath="outputs\\plots"):
    assert(const_var == 'n' or const_var == 'm')
    changing_var = 'n' if const_var == 'm' else 'm'

    filtered_df = df[df[const_var] == const_val]

    x_list = filtered_df[changing_var]
    pivot_list = filtered_df['Number of pivots']
    times_list = [np.round((t * 1000), 4) for t in filtered_df['Time elapsed (s)']]

    fig, ax1 = plt.subplots()
    title = "Pivot Count and Runtime with {}={} vs {}".format(const_var, const_val, changing_var)
    plt.title(title)
    ax1.set_xlabel(changing_var)
    
    pivot_color = "red"
    ax1.set_ylabel('Pivot Count', color=pivot_color)
    ax1.tick_params(axis='y', labelcolor=pivot_color)
    ax1.plot(x_list, pivot_list, color=pivot_color, marker='o')

    ax2 = ax1.twinx()

    time_color = "blue"
    ax2.set_ylabel('Time elapsed (ms)', color=time_color)
    ax2.tick_params(axis='y', labelcolor=time_color)
    ax2.plot(x_list, times_list, color=time_color, marker='o')

    fig.tight_layout()

    filename = "{}={}_vs_{}.png".format(const_var, int(const_val), changing_var)
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    plt.savefig("{}\\{}".format(filepath, filename))

def create_results_csv(df, title='tabulation', filepath="outputs\\tables"):
    stats = pd.DataFrame(df)

    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    stats.to_csv('{}\\{}.csv'.format(filepath, title), index=False)

def graph_results(df):
    n_list = list(set(df['n']))
    n_list.sort()

    m_list = list(set(df['m']))
    m_list.sort()

    for n in n_list:
        graph_params_vs_n_or_m(df, const_val=n, const_var='n', filepath="outputs\\plots\\vs_m")

    for m in m_list:
        graph_params_vs_n_or_m(df, const_val=m, const_var='m', filepath="outputs\\plots\\vs_n")


if __name__ == "__main__":
    PartA()

    # use trials = 1 just to create a csv
    single_results = PartB(lp_trials=1)
    print(single_results)
    create_results_csv(single_results)

    # use trials = 200 so that we can get an average pivot count among many randomly generated LP problems
    avg_results = PartB(lp_trials=200)
    graph_results(avg_results)
