from audioop import avg
from cProfile import label
from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from skopt import gbrt_minimize
from skopt.space import Real
import math
import time as time_this

BETA_MAX = 1000

def gue(dimensions):
    raw = np.random.normal(scale = np.sqrt(2)/2, size=(dimensions,dimensions * 2)).view(np.complex128)
    return (raw + raw.T.conj())/ np.sqrt(2)

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

# returns (beta, fro error)
def find_best_beta(rho, ham, verbose=False):
    dims = [Real(0.0, BETA_MAX)]
    def obj_fun(dim):
        beta = dim[0]
        # print("beta:", beta)
        # print("matrix:")
        # print(linalg.expm(-beta * ham))
        guess = thermal_state(ham, beta)
        return np.linalg.norm(rho - guess)
    ret = gbrt_minimize(obj_fun, dims, x0=[1.0], verbose=verbose)
    return (ret.x, ret.fun)

# This class simulates weak interactions with a bath using GUE couplings. The main method is
# "channel" which simulates the time evolution channel using an average of several samples of
# interactions with GUE. The specification is you provide a list of environment betas of which
# to use thermal states for as an expendable resource. The system is iteratively coupled with
# these environment resources and simulated for the corresponding time and coupling strength
# alpha. 
class QHMC:
    def __init__(
                self,
                env_betas = [1.0],
                sys_start_beta = 0.0,
                sim_times = [1],
                alphas = [0.01],
                ham_sys=np.zeros((2,2)),
                ham_env_base=np.diag([0.5, 1.5]),
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
        
        # return a dictionary mapping to a list of sys output betas and erros, do we only return average beta and avg error?
        # or return a "list of lists" to let another function process? 
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

if __name__ == "__main__":
    # test()
    start = time_this.time()
    test_hamiltonian_gap()
    end = time_this.time()
    print("took this many seconds: ", end - start)
