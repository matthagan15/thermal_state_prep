import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from skopt import gbrt_minimize
from skopt.space import Real

def thermal_state(hamiltonian, beta):
    mat = linalg.expm(- beta * hamiltonian)
    partition = np.trace(mat)
    return mat / partition

def harmonic_oscillator_hamiltonian(dimensions, gap = 1.0):
    diags = [0.5 + n for n in range(dimensions)]
    return np.diag(diags)

def gue(dimensions):
    raw = np.random.normal(scale = np.sqrt(2)/2, size=(dimensions,dimensions * 2)).view(np.complex128)
    return (raw + raw.T.conj())/ np.sqrt(2)

def partrace(mat, d1, d2):
    tensor = mat.reshape([d1,d2,d1,d2])
    return np.trace(tensor, axis1=1, axis2=3)

def channel(rho, beta_env, time, alpha, dim1=10, dim2=10):
    oscillator1 = harmonic_oscillator_hamiltonian(dim1)
    oscillator2 = harmonic_oscillator_hamiltonian(dim2)
    rho_env =thermal_state(oscillator2, beta_env)
    rho_tot = np.kron(rho, rho_env)
    interactions = gue(dim1 * dim2)
    ham = np.kron(oscillator1, np.identity(dim2)) + np.kron(np.identity(dim1), oscillator2) + alpha * interactions
    u = linalg.expm(1j * ham * time)
    raw = u @ rho_tot @ u.conj().T
    return partrace(raw, dim1, dim2)

def repeat_channel(beta0, betaf, dim1=10, dim2=10):
    h_sys = harmonic_oscillator_hamiltonian(dim1)
    rho = thermal_state(h_sys, beta0)
    for n in range(100):
        beta = beta0 + (n*0.01) * betaf
        rho = channel(rho, beta, 100, 0.01, dim1=dim1, dim2=dim2)
    get_beta_min(rho, h_sys)


def get_beta_min(rho, ham):
    dims = [Real(0.0, 1e3)]
    def obj_fun(dim):
        beta = dim[0]
        guess = thermal_state(ham, beta)
        return np.linalg.norm(rho - guess)
    ret = gbrt_minimize(obj_fun, dims, x0=[1.0])
    print("found beta: ", ret.x)
    print("frobenius diff norm:", ret.fun)

def test():
    d1 = 4
    d2 = 8
    b1 = 0.
    b2 = 1.
    t = 100.
    alpha = 0.1
    h1 = harmonic_oscillator_hamiltonian(d1)
    therm1 = thermal_state(h1, b1)
    out = channel(therm1, b2, t, alpha, dim1=d1, dim2=d2)
    print("energy 1:", np.trace(h1 @ therm1))
    print("oscillator 1")
    print(h1)
    print("output")
    print(out)
    print("energy out:", np.trace(out @ h1))
    print("#" * 50)
    print("output diagonals:", np.abs(np.diagonal(out)))
    print("input thermal state:", np.abs(np.diagonal(therm1)))
    print("delta:", np.abs(np.diagonal(out)) - np.abs(np.diagonal(therm1)))
    get_beta_min(out, h1)
    repeat_channel(0.0, 1.0)
test()