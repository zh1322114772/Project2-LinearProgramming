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

def PartB():
    '''
    B. Next try to increase the number of variables (n) and increase the number of constraints (m),
    thus:

    - Fixing m=2: Rerun the code after increasing n from 4 to 10 to 50. (in increments of 10) and n
    - For each of the n values above: Rerun the code after increasing m from 2 to 6 to 10 to 14. (in
    increments of 4, just create additional constraints)

    '''

    ret = pd.DataFrame(columns=['n', 'm', 'Time elapsed (s)', 'Number of pivots'])

    for n in [2, 10, 20, 30, 40, 50]:
        for m in [2, 6, 10, 14]:

            while True:

                # Generate random LP problem
                c, A, b, bounds = random_feasible_lp(n, m)

                try:
                    start = time.time()
                    solution = ComputeSimplex(c, A, b, bounds)
                    end = time.time()
                    elapsed = end - start

                    ret.loc[len(ret)] = [n, m, elapsed, solution.nit]

                    break
                except ValueError as e:
                    print(f"Failed to solve LP for n={n}, m={m}. Retrying...")
            
    return ret

def graph_params_vs_n(df, m, filepath="plots\\params_vs_n"):
    filtered_df = df[df['m'] == m]

    n_list = filtered_df['n']
    times_list = filtered_df['Time elapsed (s)']
    pivot_list = filtered_df['Number of pivots']

    plt.figure()
    title = "Pivot Count and Runtime with m={} vs n".format(m)
    plt.title(title)
    plt.xlabel("n")
    plt.ylabel("Pivot Count")

    plt.plot(n_list, pivot_list, color="red")

    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    plt.savefig("{}\\{}.png".format(filepath, title))


def graph_results(df):
    m_list = list(set(df['m']))
    m_list.sort()
    
    for m in m_list:
        graph_params_vs_n(df, m)


if __name__ == "__main__":
    PartA()

    results = PartB()
    print(results)

    graph_results(results)


