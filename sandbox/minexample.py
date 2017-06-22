# coding: utf-8


import scipy.integrate
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt
import magnics

# Material parameters
Ms = 8.6e5  # saturation magnetisation (A/m)
alpha = 0.1  # Gilbert damping
gamma = 2.211e5  # gyromagnetic ratio
A = 1e-11 # exchange constant (J/m)
#A = 0
D = 0*1.58e-3 # DMI constant (FeGe: D = 1.58e-3 J/m²)
K = 0*5e3 # anisotropy constant (Co: K = 4.5e5 J/m³)

# External magnetic field.
B = 0.1  # (T)
mu0 = 4 * np.pi * 1e-7  # vacuum permeability

# Zeeman field
H = Ms / 2 * fe.Constant((0,0,1))
# easy axis
ea = fe.Constant((0,1,1))

# mesh parameters
d = 100e-9
thickness = 100e-9
nx = ny = 20
nz = 10

# create mesh
p1 = fe.Point(0, 0, 0)
p2 = fe.Point(d, d, thickness)
mesh = fe.BoxMesh(p1, p2, nx, ny, nz)


m, Heff, u, v, V = magnics.vectorspace(mesh)

def effective_field(m, volume=None):
    w_Zeeman = - mu0 * Ms * fe.dot(m, H)
    w_exchange = A  * fe.inner(fe.grad(m), fe.grad(m))
    w_DMI = D * fe.inner(m, fe.curl(m))
    w_ani = - K * fe.inner(m, ea)**2
    w = w_Zeeman + w_exchange + w_DMI + w_ani
    return -1/(mu0*Ms) * fe.derivative(w*fe.dx, m)

# Effective field
Heff_form = effective_field(m)

# Preassemble projection Matrix
Amat = fe.assemble(fe.dot(u, v)*fe.dx)

LU = fe.LUSolver()
LU.set_operator(Amat)


def compute_dmdt(m):
    """Convenience function that does all in one go"""

    # Assemble RHS
    b = fe.assemble(Heff_form)

    # Project onto Heff
    LU.solve(Heff.vector(), b)

    LLG = -gamma/(1+alpha*alpha)*fe.cross(m, Heff) - alpha*gamma/(1+alpha*alpha)*fe.cross(m, fe.cross(m, Heff))

    result = fe.assemble(fe.dot(LLG, v)*fe.dP)
    return result.array()

# function for integration of system of ODEs

def rhs_micromagnetic(m_vector_array, t, counter=[0]):
    assert isinstance(m_vector_array, np.ndarray)
    m.vector()[:] = m_vector_array[:]

    dmdt = compute_dmdt(m)
    return dmdt

m_init = fe.Constant((1, 0, 0))
m = fe.interpolate(m_init, V)
ts = np.linspace(0, 5e-10, 100)
# empty call of time integrator, just to get FEniCS to cache all forms etc
rhs_micromagnetic(m.vector().array(), 0)

ms = scipy.integrate.odeint(rhs_micromagnetic, y0=m.vector().array(), t=ts, rtol=1e-10, atol=1e-10)


def macrospin_analytic_solution(alpha, gamma, H, t_array):
    """
    Computes the analytic solution of magnetisation x component
    as a function of time for the macrospin in applied external
    magnetic field H.

    Source: PhD Thesis Matteo Franchin,
    http://eprints.soton.ac.uk/161207/1.hasCoversheetVersion/thesis.pdf,
    Appendix B, page 127

    """
    t0 = 1 / (gamma * alpha * H) * np.log(np.sin(np.pi / 2) / \
                                          (1 + np.cos(np.pi / 2)))
    mx_analytic = []
    for t in t_array:
        phi = gamma * H * t                                     # (B16)
        costheta = np.tanh(gamma * alpha * H * (t - t0))        # (B17)
        sintheta = 1 / np.cosh(gamma * alpha * H * (t - t0))    # (B18)
        mx_analytic.append(sintheta * np.cos(phi))

    return np.array(mx_analytic)

mx_analytic = macrospin_analytic_solution(alpha, gamma, Ms/2, ts)

tmp2 = ms[:,0:1]  # might be m_x, m_y, m_z of first vector
# pylab.plot(ts, tmp2, 'o-')
# pylab.plot(ts, mx_analytic, 'x-')
# pylab.legend(['simulation', 'analytical'])
# pylab.xlabel('$t\,$[s]')
# pylab.ylabel('$<M_x>$')

difference = tmp2[:,0] - mx_analytic
print("max deviation: {}".format(max(abs(difference))))
# pylab.plot(ts, difference, 'o-')
# pylab.legend(['simulation - analytical'])
# pylab.xlabel('$t\,$[s]')
# pylab.ylabel('$<\Delta M_x>$')


# add quick test
eps = 0.01
ref = 0.0655908475021
assert ref - eps < max(abs(difference)) < ref + eps
