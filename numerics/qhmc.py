from audioop import avg
from cProfile import label
from tabnanny import verbose
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from skopt import gbrt_minimize
from skopt.space import Real
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
    diags = [0.5 + n for n in range(dimensions)]
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

class QHMC:
    def __init__(
                self,
                env_betas = [1.0],
                sys_start_beta = 0.0,
                sim_times = [1],
                alphas = [0.01],
                used_env_per_beta = 1,
                ham_sys=np.zeros((2,2)),
                ham_env_base=np.diag([0.5, 1.5]),
                num_monte_carlo=10,
                verbose=False
                ):
        if len(env_betas) != len(sim_times) or len(sim_times) != len(alphas):
            print("[QHMC] length of input arrays do not match")
        else:
            self.betas = env_betas
            self.times = sim_times
            self.alphas = alphas
            self.env_per_iter = used_env_per_beta
            self.sys_start_beta = sys_start_beta
            self.ham_sys = ham_sys
            self.ham_env_base = ham_env_base
            self.num_monte_carlo = num_monte_carlo
            self.verbose = verbose

    def print_params(self):
        print("betas:", self.betas)
        print("times:", self.times)
        print("alphas:", self.alphas)
        print("env_per_iter:", self.env_per_iter)
        print("sys_start_beta:", self.sys_start_beta)
        print("ham_sys:", self.ham_sys)
        print("ham_env:", self.ham_env_base)
        print("num_monte_carlo:", self.num_monte_carlo)
        print("verbose:", self.verbose)

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
        b, e = find_best_beta(avg_out, self.ham_sys)
        # print("b,e:", b, ", ", e)
        return (avg_out, b, e)
        
        # return a dictionary mapping to a list of sys output betas and erros, do we only return average beta and avg error?
        # or return a "list of lists" to let another function process? 
    def compute_betas_and_errors(self):
        if self.verbose:
            start_time = time_this.time()
            print("#"*75)
            print("[compute_betas_and_errors] start time:", start_time)
        rho = thermal_state(self.ham_sys, self.sys_start_beta)
        ret_betas, ret_errors = [], []
        rho_diags = []
        for ix in range(len(self.betas)):
            if self.verbose:
                print("percent done:", float(ix) / len(self.betas))
            rho, out_beta, error = self.channel(rho, self.betas[ix], self.alphas[ix], self.times[ix])
            ret_betas.append(out_beta)
            ret_errors.append(error)
            rho_diags.append(np.abs(np.diagonal(rho)))
        print("done.")
        if self.verbose:
            for ix in range(len(rho_diags)):
                print("ix:", ix)
                print("ret_beta:", ret_betas[ix])
                print("ret_error:", ret_errors[ix])
                print("params: alpha=", self.alphas[ix], ", env_beta=", self.betas[ix], ", sim_time=", self.times[ix])
                print("Ground state prob:", rho_diags[ix][0])
                true = np.diagonal(thermal_state(self.ham_sys, self.betas[ix]))
                l = "ix=" + str(ix) + ", e: {:4.2f}".format(ret_errors[ix])
                if ix % 9 == 0 or ix == len(rho_diags) - 1:
                    plt.plot(rho_diags[ix], label=l)
            avg_env_state = thermal_state(self.ham_env_base, np.mean(self.betas))
            plt.plot(np.abs(np.diagonal(avg_env_state)), '-.', label="avg_env_state")
                # plt.plot(range(rho_diags[ix].shape[0]), true, "--", label="error:" + str(ret_errors[ix]))

            plt.legend()
            plt.show()

def test():
    h1 = harmonic_oscillator_hamiltonian(10)
    h2 = harmonic_oscillator_hamiltonian(10)
    betas = [2] * 45
    times = [50] * 45
    alphas = [0.005] * 45
    qhmc = QHMC(ham_sys = h1, ham_env_base=h2, env_betas=betas, sim_times=times, alphas=alphas, verbose=True)
    # qhmc.print_params()
    qhmc.compute_betas_and_errors()

test()