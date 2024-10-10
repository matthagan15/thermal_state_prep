from cProfile import label
from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

import math
import time as time_this

BETA_MAX = 1000

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
    # let (q, r) = gauss_array.qr_square().unwrap();
    #     let mut lambda = Array2::<c64>::zeros((dim, dim).f());
    #     for ix in 0..dim {
    #         lambda[[ix, ix]] = r[[ix, ix]] / r[[ix, ix]].norm();
    #     }
    #     q.dot(&lambda)

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

BINARY_SEARCH_UPPER_LIMIT = 2 ** 18

def binary_search(f , epsilon):
    """Returns:
- `None` if the binary search upper limit is exceeded
- `Some((L, s))` where `L` is the minimum number of interactions
needed to reach distance `epsilon` and `s` is the std deviation.
 -
 """
    upper = 1
    cur_dist = f(upper)
    while cur_dist > epsilon:
        upper *= 2
        if upper >= BINARY_SEARCH_UPPER_LIMIT:
            print("[BINARY_SEARCH] upper limit reached.")
            return None
        cur_dist = f(upper)
    lower = upper / 2
    print(f"[BINARY_SEARCH] searching in range [{lower}, {upper}].")
    while (upper - lower) > 1:
        mid = int(np.floor((lower + upper) / 2.))
        cur_dist = f(mid)
        if cur_dist >= epsilon:
            lower = mid
        else:
            upper = mid
    print(
        f"[BINARY_SEARCH] min_number_interactions = {upper}, distances: {cur_dist}")
    return (upper, cur_dist)

def minimum_interactions(alpha, time, beta_e, epsilon, num_samples=100):
    """
    Computes the minimum number of interactions needed to prepare a single qubit state in an 
    """
    def f(n):
        phi = QHMC(env_betas=[beta_e] * n, sys_start_beta=0.0, sim_times=[time] * n, alphas=[alpha] * n, num_monte_carlo=100)
        return phi.simulate_interactions()
    x = binary_search(f, epsilon)
    print("x: ", x)
    if type(x) == type(None):
        print("upper bound reached, returning none.")
        return None
    
    (interactions, dist ) = x
    return interactions

 
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

    # TODO: Fix this to just return avg_out and add other helpers, need to eliminate find_best_beta
    def channel(self, rho_sys, env_beta, alpha, time):
        if rho_sys.shape != self.ham_sys.shape:
            if self.verbose:
                print("[QHMC.channel] rho.shape, self.ham_sys.shape", rho_sys.shape, ",", self.ham_sys.shape)
            return None
        dim1, dim2 = self.ham_sys.shape[0], self.ham_env_base.shape[0] 
        rho_env = thermal_state(self.ham_env_base, env_beta)
        rho_tot = np.kron(rho_sys, rho_env)
        avg_out = np.zeros((rho_sys.shape[0], rho_sys.shape[1] * 2)).view(np.complex128)
        h = np.kron(self.ham_sys, np.identity(dim2 )) + np.kron(np.identity(dim1), self.ham_env_base)
        for sample in range(self.num_monte_carlo):
            g = gue(dim1 * dim2)
            ham_tot = h + alpha * g
            u = linalg.expm(1j * ham_tot * time)
            raw = u @ rho_tot @ u.conj().T
            out = partrace(raw, dim1, dim2)
            avg_out += out.view(np.complex128) / self.num_monte_carlo
        # b, e = find_best_beta(avg_out, self.ham_sys)
        # print("b,e:", b, ", ", e)
        # return (avg_out, b, e)
        return avg_out

    def simulate_interactions(self):
        """Returns the trace distance to the target thermal state after all interactions are 
        simulated."""
        output = np.zeros((self.system_state.shape[0], self.system_state.shape[1] * 2)).view(np.complex128) 
        for sample in range(self.num_monte_carlo):
            sample_state = thermal_state(self.ham_sys, self.sys_start_beta)
            for ix in range(len(self.betas)):
                rho_env = thermal_state(self.ham_env_base, self.betas[ix])
                rho_tot = np.kron(sample_state, rho_env)
                g = my_interaction(self.ham_sys.shape[0] * self.ham_env_base.shape[0])
                ham_tot = self.total_hamiltonian + self.alphas[ix] * g
                u = linalg.expm(1j * ham_tot * self.times[ix])
                raw = u @ rho_tot @ u.conj().T
                sample_state = partrace(raw, self.ham_sys.shape[0], self.ham_env_base.shape[0])
            output += sample_state
        self.system_state = output / self.num_monte_carlo
        dist = trace_distance(self.system_state, thermal_state(self.ham_sys, self.betas[-1]))
        return dist

    def compute_error_with_target_beta(self):
        """
        Simulate the interactions and give the frobenius norm distance between
        the output system state and the ideal system state. The ideal system
        state is defined to be the thermal state at the temperature of the
        last provided beta for the environment.
        """
        self.simulate_interactions()
        ideal = thermal_state(self.ham_sys, self.betas[-1])
        difference = ideal - self.system_state
        return trace_distance(ideal, self.system_state)
    
    def compute_betas_and_errors(self):
        start_time = time_this.time()
        if self.verbose:
            
            print("#"*75)
            print("[compute_betas_and_errors] start time:", start_time)
        rho = thermal_state(self.ham_sys, self.sys_start_beta)
        ideal_output = thermal_state(self.ham_sys, self.betas[-1])
        ret_betas, ret_errors = [], []
        rho_diags = []
        sys_fro_errors = []
        for ix in range(len(self.betas)):
            print("percent done:", float(ix) / len(self.betas))
            rho = self.channel(rho, self.betas[ix], self.alphas[ix], self.times[ix])
            # ret_betas.append(out_beta)
            # ret_errors.append(error)
            rho_diags.append(np.abs(np.diagonal(rho)))
            sys_fro_errors.append(np.linalg.norm(rho - ideal_output))
        # print("fro error w/ ideal:", np.linalg.norm(rho - thermal_state(self.ham_sys, self.betas[0]), ord='fro'))
        print("final frobenius error compared to ideal:", sys_fro_errors[-1])
        print("done. total time:", time_this.time() - start_time)
        if self.verbose:
            for ix in range(len(rho_diags)):
                print("ix:", ix)
                # print("ret_beta:", ret_betas[ix])
                print("ret_error:", sys_fro_errors[ix])
                print("params: alpha=", self.alphas[ix], ", env_beta=", self.betas[ix], ", sim_time=", self.times[ix])
                print("Ground state prob:", rho_diags[ix][0])
                l = "#refreshes=" + str(ix) + ", err: {:4.2f}".format(sys_fro_errors[ix])
                # We want to plot five points equally spaced.
                plot_ix_length = math.ceil(len(rho_diags) / 5.)
                if ix % 40 == 39:
                    plt.plot(rho_diags[ix], label=l)
            avg_env_state = thermal_state(self.ham_env_base, np.mean(self.betas))
            plt.plot(np.abs(np.diagonal(ideal_output)), '-.', label="ideal output")
            # plt.plot(np.diagonal(self.ham_sys), '+', label="hamiltonian")
            plt.title("Thermalization of sqrt(x) w/ single qubit")
            plt.xlabel("Eigenvector Number")
            plt.ylabel("State Overlap")
            # plt.plot(range(rho_diags[ix].shape[0]), true, "--", label="error:" + str(ret_errors[ix]))

            plt.legend()
            plt.show()

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

if __name__ == "__main__":
    # test()
    start = time_this.time()
    # test_dimension()
    alpha = 0.001
    time = 100.
    epsilon = 0.05
    x = []
    y = []
    for ix in range(20):
        beta_e = 0.0 + 0.3 * ix
        print("computing beta_e = ", beta_e)
        res = minimum_interactions(alpha, time, beta_e, epsilon)
        if res == None:
            continue
        x.append(beta_e)
        y.append(res)
    end = time_this.time()
    print("took this many seconds: ", end - start)
    print("x: ", x)
    print("y: ", y)
    plt.plot(x, y)
    plt.show()
    # test_beta()
    # qhmc = QHMC(ham_sys = harmonic_oscillator_hamiltonian(20), ham_env_base = harmonic_oscillator_hamiltonian(5), sys_start_beta = 0.5, env_betas = [1.], sim_times = [100.], alphas = [0.001], num_monte_carlo = 500)
    # qhmc.test_distance_with_interactions()
    
