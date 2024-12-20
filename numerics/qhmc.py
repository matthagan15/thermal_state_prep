from cProfile import label
import pickle
from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['text.usetex'] = True

from scipy import linalg

import math
import time as time_this

from joblib import Parallel, delayed

BETA_MAX = 1000

def prob_dist_trace_norm(p, q):
    return np.sum(np.abs(p - q))

def min_interactions_sho_markov_chain(beta, alpha, time, epsilon, dim):
    """Computes the minimum number of steps needed for the associated markov chain for the
    given parameters to reach the stationary distribution."""
    print("MARKOV SEARCH")
    target_state = np.exp(- beta * np.linspace(0.0, (dim - 1) * 1.0, dim)) / np.sum(np.exp(- beta * np.linspace(0.0, (dim - 1) * 1.0, dim)))
    target_state_nonnormal = [np.exp(- beta * ix) for ix in range(dim)]
    partfun = np.sum(target_state_nonnormal)
    target_state = np.array(target_state_nonnormal) / partfun
    target_state = np.reshape(target_state, (dim, 1))
    initial_state = (1. / dim) * np.ones((dim, 1))
    markov_operator = compute_sho_markov_chain(alpha, beta, time, dim)

    
    def f(n, threadpool):
        out = np.linalg.matrix_power(markov_operator, n) @ initial_state
        out = np.reshape(out, (dim, 1))
        # print(out)
        ret = prob_dist_trace_norm(target_state, out)
        # print("[markov] n = ", n, ", dist: ", ret)
        return ret
    # print("output")
    # f(2, None)
    # return None
    x = binary_search(f, epsilon, None)
    if type(x) == type(None):
        print("upper bound reached, returning none.")
        return None
    
    (interactions, dist ) = x
    return interactions

def compute_sho_markov_chain(alpha, beta, time, dim):
    t_mat = np.zeros((dim, dim))
    q0 = 1.0 / (1.0 + np.exp(- beta))
    q1 = 1.0 - q0
    for row_ix in range(dim):
        for col_ix in range(dim):
            if row_ix == col_ix:
                if row_ix == 0:
                    t_mat[row_ix, col_ix] = - q1
                elif row_ix == dim - 1:
                    t_mat[row_ix, col_ix] = - q0
                else:
                    t_mat[row_ix, col_ix] = - 1.
            elif row_ix == col_ix + 1:
                t_mat[row_ix, col_ix] = q1
            elif row_ix == col_ix - 1:
                t_mat[row_ix, col_ix] =  q0
            else:
                continue
    ret = np.identity(dim) + (((alpha * time) ** 2) / (2 * dim + 1.)) * t_mat
    return ret

def sample_gammas(mean, upper, num_samples):
    """Samples from a gaussian with provided mean and spectral norm of the system hamiltonian."""
    rng = np.random.default_rng()
    samples = rng.normal(mean.real, 1.5 * np.abs(upper - mean), size=(num_samples, 1))
    samples = np.abs(samples)
    return np.minimum(samples, upper)

def load_h_chain():
    with open('/Users/matt/scratch/hamiltonians/h_chain_3.pickle', 'rb') as openfile:
        h_list = list(pickle.load(openfile))
        dims = h_list.pop()
        h = np.zeros(dims, dtype=np.complex128)
        num_terms = len(h_list)
        for h_term in h_list:
            h_term = np.array(h_term)
            # h += h_term / np.max([1.0, np.linalg.norm(h_term, 2)])
            h += h_term 
        print("loaded hamiltonian with shape: ", h.shape, " and ", num_terms, " terms.")
        return h

def gue(dimensions):
    raw = np.random.normal(scale = np.sqrt(2)/2, size=(dimensions,dimensions * 2)).view(np.complex128)
    return (raw + raw.T.conj())/ np.sqrt(2)

def haar_sample(dimensions):
    raw = np.random.normal(scale = np.sqrt(2)/2, size=(dimensions,dimensions * 2)).view(np.complex128)
    q,r = np.linalg.qr(raw)
    l = np.zeros_like(q)
    for ix in range(l.shape[0]):
        l[ix, ix] = r[ix, ix] / np.abs(r[ix, ix])
    return q @ l

def my_interaction(dimensions):
    """Draws a sample of a matrix with Haar random eigenvectors and iid gausian eigenvalues"""
    u = haar_sample(dimensions)
    rng = np.random.default_rng()
    eigvals = rng.normal(0.0, 1.0, size=(dimensions, ))
    l = np.diag(eigvals)
    return u @ l @ u.T.conj()

def partrace(mat, d1, d2):
    tensor = mat.reshape([d1,d2,d1,d2])
    return np.trace(tensor, axis1=1, axis2=3)

def thermal_state(hamiltonian, beta):
    mat = linalg.expm(- beta * hamiltonian)
    partition = np.trace(mat)
    ret = 0
    try:
        ret = mat / partition
    except Exception as e:
        print("BIG ERROR:")
        print(e)
        ret = mat * 0
    return ret

def trace_distance(x, y):
    m = x - y
    v = np.linalg.eigvals(m)
    return np.abs(v).sum()

def harmonic_oscillator_hamiltonian(dimensions, gap = 1.0):
    """
    Inputs: dimensions - number of basis states to use
    (optional) gap - default 1.0 
    """
    diags = [0.5 + n * gap for n in range(dimensions)]
    return np.diag(diags)

def pathological_hamiltonian(dimensions):
    diags = []
    for ix in range(dimensions):
        diags.append(10. * np.log(1. + 1.* ix))
    return np.diag(diags)

def sqrt_hamiltonian(dimensions):
    diags = [np.sqrt(ix) for ix in range(dimensions)]
    return np.diag(diags)

BINARY_SEARCH_UPPER_LIMIT = 2 ** 16

def binary_search(f , epsilon, threadpool):
    """Returns:
- `None` if the binary search upper limit is exceeded
- `Some((L, s))` where `L` is the minimum number of interactions
needed to reach distance `epsilon` and `s` is the std deviation.
 -
 """
    upper = 1
    cur_dist = f(upper, threadpool)
    while cur_dist > epsilon:
        upper *= 2
        if upper >= BINARY_SEARCH_UPPER_LIMIT:
            print("[BINARY_SEARCH] upper limit reached.")
            return None
        # print("upper: ", upper)
        cur_dist = f(upper, threadpool)
    lower = upper / 2
    # print(f"[BINARY_SEARCH] searching in range [{lower}, {upper}].")
    while (upper - lower) > 1:
        mid = int(np.floor((lower + upper) / 2.))
        cur_dist = f(mid, threadpool)
        if cur_dist >= epsilon:
            lower = mid
        else:
            upper = mid
    # print(
        # f"[BINARY_SEARCH] min_number_interactions = {upper}, distances: {cur_dist}")
    return (upper, cur_dist)

def minimum_interactions(alpha, time, beta_e, epsilon, dim, num_samples=100):
    """
    Computes the minimum number of interactions needed to prepare a single qubit state in an 
    """
    # print("SIMULATED SEARCH")
    def f(n, threadpool):
        phi = QHMC(ham_sys=harmonic_oscillator_hamiltonian(dim), env_betas=[beta_e] * n, sys_start_beta=0.0, sim_times=[time] * n, alphas=[alpha] * n, num_monte_carlo=num_samples)
        return phi.simulate_interactions(threadpool)
    with Parallel(n_jobs=8) as threadpool:
        x = binary_search(f, epsilon, threadpool)
        print("x: ", x)
    if type(x) == type(None):
        print("upper bound reached, returning none.")
        return None
    
    (interactions, dist ) = x
    return interactions

def fixed_number_interactions(h_sys, alpha, time, beta_e, num_interactions, num_samples = 100, gamma_strategy = 'fixed'):
    """Attempts to track the distance to the target thermal state as a function of the interactions used. 
    # Returns
    A pair of numpy arrays, the first being the average distance to the target and the second the std of the dist.
    """
    phi = QHMC(ham_sys=h_sys, env_betas=[beta_e] * num_interactions, sys_start_beta=0.0, sim_times=[time] * num_interactions, alphas=[alpha] * num_interactions, num_monte_carlo=num_samples)
    threadpool = Parallel(n_jobs=8)
    return  phi.simulate_with_random_env(threadpool, gamma_strategy=gamma_strategy)
 

def fixed_num_interactions_markov(dim, alpha, time, beta, num_interactions):
    markov_op = compute_sho_markov_chain(alpha, beta, time, dim)
    target_state = np.exp(- beta * np.linspace(0.0, (dim - 1) * 1.0, dim)) / np.sum(np.exp(- beta * np.linspace(0.0, (dim - 1) * 1.0, dim)))
    target_state_nonnormal = [np.exp(- beta * ix) for ix in range(dim)]
    partfun = np.sum(target_state_nonnormal)
    target_state = np.array(target_state_nonnormal) / partfun
    target_state = np.reshape(target_state, (dim, 1))
    initial_state = (1. / dim) * np.ones((dim, 1))
    ret = []
    for n in range(num_interactions):
        out = np.linalg.matrix_power(markov_op, n) @ initial_state
        out = np.reshape(out, (dim, 1))
        dist = prob_dist_trace_norm(target_state, out)
        ret.append(dist)
    return ret


class QHMC:
    """
    This class simulates weak interactions with a bath using GUE couplings. The main method is
"channel" which simulates the time evolution channel using an average of several samples of
interactions with GUE. The specification is you provide a list of environment betas of which
to use thermal states for as an expendable resource. The system is iteratively coupled with
these environment resources and simulated for the corresponding time and coupling strength
alpha.
    """
    def __init__(
                self,
                env_betas = [1.0],
                sys_start_beta = 0.0,
                sim_times = [1],
                alphas = [0.01],
                ham_sys=np.diag([0.0, 1.0]),
                ham_env_base=np.diag([0.0, 1.0]),
                num_monte_carlo=500,
                verbose=False
                ):
        if len(env_betas) != len(sim_times) or len(sim_times) != len(alphas):
            print("[QHMC] length of input arrays do not match")
        else:
            self.betas = env_betas
            self.times = sim_times
            self.alphas = alphas
            self.sys_start_beta = sys_start_beta
            self.ham_sys = ham_sys
            self.ham_env_base = ham_env_base
            self.num_monte_carlo = num_monte_carlo
            self.verbose = verbose
            self.system_state = thermal_state(ham_sys, sys_start_beta)
            self.total_hamiltonian = np.kron(ham_sys, np.identity(ham_env_base.shape[0])) + np.kron(np.identity(ham_sys.shape[0]), ham_env_base)

    def print_params(self):
        print("betas:", self.betas)
        print("times:", self.times)
        print("alphas:", self.alphas)
        print("sys_start_beta:", self.sys_start_beta)
        print("ham_sys:", self.ham_sys)
        print("ham_env:", self.ham_env_base)
        print("num_monte_carlo:", self.num_monte_carlo)
        print("verbose:", self.verbose)

    def simulate_interactions(self, threadpool):
        """Returns the trace distance to the target thermal state after all interactions are 
        simulated."""
        output = np.zeros((self.system_state.shape[0], self.system_state.shape[1] * 2)).view(np.complex128) 
        target_state = thermal_state(self.ham_sys, self.betas[-1])
        def sampler():
            sample_state = thermal_state(self.ham_sys, self.sys_start_beta)
            for ix in range(len(self.betas)):
                rho_env = thermal_state(self.ham_env_base, self.betas[ix])
                rho_tot = np.kron(sample_state, rho_env)
                g = my_interaction(self.ham_sys.shape[0] * self.ham_env_base.shape[0])
                ham_tot = self.total_hamiltonian + self.alphas[ix] * g
                u = linalg.expm(1j * ham_tot * self.times[ix])
                raw = u @ rho_tot @ u.conj().T
                sample_state = partrace(raw, self.ham_sys.shape[0], self.ham_env_base.shape[0])
            return sample_state
        samples = threadpool(delayed(sampler)() for i in range(self.num_monte_carlo))
        for sample in samples:
            output += sample
        self.system_state = output / self.num_monte_carlo
        dist = trace_distance(self.system_state, thermal_state(self.ham_sys, self.betas[-1]))
        return dist

    def simulate_with_random_env(self, threadpool, gamma_strategy = "random"):
        avg = np.trace(self.ham_sys) / self.ham_sys.shape[0]
        h_norm = 2*np.linalg.norm(self.ham_sys, ord = 2)
        output = np.zeros((self.system_state.shape[0], self.system_state.shape[1] * 2)).view(np.complex128) 
        target_state = thermal_state(self.ham_sys, self.betas[-1])

        def sampler2():
            sample_state = thermal_state(self.ham_sys, self.sys_start_beta)
            if gamma_strategy == 'random':
                gammas = sample_gammas(avg, h_norm, len(self.betas))
            elif gamma_strategy == 'fixed':
                print("fixed gamma strat.")
                gammas = [1.0] * len(self.betas)
            else:
                print("Unsupported gamma strategy. Use 'random' or 'fixed'.")
                raise Exception

            dists = []
            for ix in range(len(self.betas)):
                if ix % 1000 == 0:
                    percent_done = ix * 100.0/len(self.betas)
                    print(percent_done, "% done")
                    # print(ix)
                rho_env = thermal_state(gammas[ix] * self.ham_env_base, self.betas[ix])
                rho_tot = np.kron(sample_state, rho_env)
                g = my_interaction(self.ham_sys.shape[0] * self.ham_env_base.shape[0])
                tmp = np.kron(self.ham_sys, np.identity(self.ham_env_base.shape[0])) + np.kron(np.identity(self.ham_sys.shape[0]), gammas[ix] * self.ham_env_base)
                ham_tot = tmp + self.alphas[ix] * g
                u = linalg.expm(1j * ham_tot * self.times[ix])
                raw = u @ rho_tot @ u.conj().T
                sample_state = partrace(raw, self.ham_sys.shape[0], self.ham_env_base.shape[0])
                dists.append(trace_distance(sample_state, target_state))
            return (sample_state, dists)

        def sampler():
            sample_state = thermal_state(self.ham_sys, self.sys_start_beta)
            gammas = sample_gammas(avg, h_norm, len(self.betas))
            dists = []
            for ix in range(len(self.betas)):
                if ix % 1000 == 0:
                    print(f"{ix / len(self.betas)}% done")
                output = 0.0 * sample_state
                for gamma in gammas:
                    rho_env = thermal_state(gamma * self.ham_env_base, self.betas[ix])
                    rho_tot = np.kron(sample_state, rho_env)
                    g = my_interaction(self.ham_sys.shape[0] * self.ham_env_base.shape[0])
                    ham_tot = self.total_hamiltonian + self.alphas[ix] * g
                    u = linalg.expm(1j * ham_tot * self.times[ix])
                    raw = u @ rho_tot @ u.conj().T
                    output += partrace(raw, self.ham_sys.shape[0], self.ham_env_base.shape[0])
                sample_state = output / (len(gammas) * 1.0)
                dists.append(trace_distance(sample_state, target_state))
            return (sample_state, dists)
        parallel_rets = threadpool(delayed(sampler2)() for i in range(self.num_monte_carlo))
        avg_dists = np.zeros((1, len(self.betas)))
        dist_matrix = np.zeros((len(parallel_rets), len(self.betas)))
        ground_state_prob = 0.0
        for ix in range(len(parallel_rets)):
            (sample_state, dists) = parallel_rets[ix]
            ground_state_prob += np.abs(sample_state[0,0])
            sampled_dists = np.array(dists).reshape((1, len(self.betas)))
            dist_matrix[ix, : ] = sampled_dists
        print('avg ground state prob: ', ground_state_prob / len(parallel_rets))
        return (np.mean(dist_matrix, axis=0), np.std(dist_matrix, axis=0))
        

def test_hamiltonian_gap():
    # h1 = harmonic_oscillator_hamiltonian(10)
    h1 = sqrt_hamiltonian(10)
    h2 = harmonic_oscillator_hamiltonian(2)
    # This took 650 seconds for 50 iters and resulted in final error around 74. Probably need to do ~300 iters.
    iters = 200
    betas = [1.] * iters
    times = [100.] * iters
    alphas = [0.01] * iters
    qhmc = QHMC(ham_sys = h1, ham_env_base=h2, env_betas=betas, sim_times=times, alphas=alphas, verbose=True)
    qhmc.compute_betas_and_errors()

def test_dimension():
    sys_dimensions = [5, 10, 15, 20, 25, 30]
    env_dimension = 10
    env_hamiltonian = harmonic_oscillator_hamiltonian(env_dimension)
    for sys_dim in sys_dimensions:
        qhmc = QHMC(ham_sys = harmonic_oscillator_hamiltonian(sys_dim), env_betas=[1.], sim_times=[100.], alphas=[0.01], ham_env_base=env_hamiltonian, num_monte_carlo=500, sys_start_beta = 0.5)
        print("dimension: ", sys_dim)
        print("output error: ", qhmc.compute_error_with_target_beta())

def test_beta():
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    env_dimension = 10
    sys_dim = 40
    env_hamiltonian = harmonic_oscillator_hamiltonian(env_dimension)
    for beta in betas:
        qhmc = QHMC(ham_sys = harmonic_oscillator_hamiltonian(sys_dim), env_betas=[1.], sim_times=[100.], alphas = [0.01], ham_env_base = env_hamiltonian, num_monte_carlo=500, sys_start_beta=beta)
        print("beta: ", beta)
        print("output error: ", qhmc.compute_error_with_target_beta())

def plot_sho_error_v_interaction():
    
    time = 100.
    n_int = 300
    beta = 3.0
    alpha = 0.005
    dim = 4
    x = [ix for ix in range(n_int)]
    y, yerr = fixed_number_interactions(harmonic_oscillator_hamiltonian(dim), alpha, time, beta, n_int, num_samples=100, gamma_strategy='fixed')
    markov_pred = fixed_num_interactions_markov(dim, alpha, time, beta, n_int)
    plt.errorbar(x, y, yerr, label="Simulated")
    plt.plot(x, markov_pred, label="Markov Pred.")
    plt.legend(loc='upper right')
    plt.ylabel(r"$|| \rho(\beta) - \Phi^L (\rho(0)) ||$")
    plt.xlabel(r"$L$")
    plt.title(r'$|| \rho(\beta) - \Phi^L (\rho(0)) || $ as a function of L W/ dim=4 SHO, $\alpha = 0.005, t = 100.$')
    plt.show() 
    return

def plot_sho_tot_time_vs_time():
    alphas = np.linspace(0.01, 0.001, 5)
    times = np.logspace(np.log10(10), np.log10(1000.), 20)
    epsilon = 0.05
    dim = 4
    betas = np.logspace(np.log10(1e-1), np.log10(dim), 10, base=10)
    beta = 4.0
    y = []
    # markov_pred = []
    results = {}

    for alpha in alphas:
        for time in times:
            print("alpha, time: ", alpha, time)
            ret = minimum_interactions(alpha, time, beta, epsilon, dim, num_samples=80)
            if ret is None:
                results_full = False
                continue
            data = results.get(alpha, [])
            data.append((time, ret * time))
            results[alpha] = data
    
    for k,v in results.items():
        print("alpha: ", k)
        print("results: ", v)
        x, y = zip(*v)
        plt.plot(x, y, label='a = {:.4}'.format(k))
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel(r"$t$")
    # plt.title("Total simulation time for dim = 4 Harmonic Oscillator to cool to beta = 4 vs time per interaction")
    plt.ylabel("Total Sim Time $L t$")
    plt.legend(loc="upper right")
    plt.show()

def plot_sho_interaction_v_beta():
    # alphas = np.logspace(np.log10(0.0001), np.log10(0.001), 4)
    alphas = [0.01]
    times = np.logspace(np.log10(10), np.log10(1000.), 10)
    epsilon = 0.05
    dim = 4
    betas = np.logspace(np.log10(1e-1), np.log10(dim), 10, base=10)
    y = []
    # markov_pred = []
    results = {}

    for alpha in alphas:
        for time in times:
            print("alpha, time: ", alpha, time)
            y = []
            results_full = True
            for beta in betas:
                ret = minimum_interactions(alpha, time, beta, epsilon, dim, num_samples=8)
                if ret is None:
                    results_full = False
                    break
                y.append(ret * time)
            if results_full:
                results[(alpha, time)] = y
    for k,v in results.items():
        print("alpha, t: ", k)
        print("results: ", v)
        plt.plot(betas, v, label='a = {:.4}, t={:1.4}'.format(k[0], k[1]))
    plt.yscale('log')
    plt.xlabel(r"$\beta$")
    plt.title("Total time of simulation vs. beta for varying single interaction times.")
    plt.ylabel("Total Sim Time")
    plt.legend(loc="lower right")
    plt.show()
    # for beta_e in np.logspace(np.log10(1e-1), np.log10(4.0), 10,base=10.):
    #     # beta_e = 0.0 + 0.3 * ix
    #     print("computing beta_e = ", beta_e)
    #     res = minimum_interactions(alpha, time, beta_e, epsilon, dim, num_samples=2)
    #     if res == None:
    #         continue
    #     x.append(beta_e)
        # y.append(res)
        # markov_pred.append(min_interactions_sho_markov_chain(beta_e, alpha, time, epsilon, dim))
    # plt.plot(x, y, label="Simulated")
    # plt.plot(x, markov_pred, label="Markov Pred.")

    # plt.title(r"Minimum interactions needed for $|| \rho(\beta) - \Phi^L (\rho(0)) || < 0.05 $, $\alpha = 0.005, t = 100.0$")
    # plt.legend(loc="lower right")
    # plt.show()
    return

if __name__ == "__main__":
    start = time_this.time()
    plot_sho_tot_time_vs_time()