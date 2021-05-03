"""
This is the code for Dendritic Growth benchmark.
The domain is a square with no-flux bcs.
Follow the link for the description of the benchmark:
https://pages.nist.gov/pfhub/benchmarks/benchmark3.ipynb/
"""

from ach_model import AllenCahnHeatModel

import firedrake as fd
from firedrake.petsc import PETSc
import firedrake_ts
import ufl
import time

PETSc.Sys.Print("Dendritic Growth benchmark.")

# Flag for saving the results for paraview
save_solution = True

# Create mesh and build function space
Lx = 960.0
Ly = 960.0
mesh = fd.RectangleMesh(100, 100, Lx, Ly, quadrilateral=True)


def create_initial_conditions(x: ufl.geometry.SpatialCoordinate, undercooling: float):
    center = (0.0, 0.0)
    radius = 8.0
    epsilon = 1.0

    values = [None, None]

    # phase function
    distance_function = (
        ufl.sqrt((x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2) - radius
    )
    values[0] = -ufl.operators.tanh(
        distance_function / (ufl.sqrt(2.0) * epsilon)
    )  # in range -1 : 1

    # temperature
    values[1] = undercooling

    return ufl.as_vector(values)


# Define function spaces P1 for Allen-Cahn, and P1 for Heat
P1_finite_element = fd.FunctionSpace(mesh, "P", 1).ufl_element()
W = fd.FunctionSpace(mesh, fd.MixedElement([P1_finite_element, P1_finite_element]))

# Create dict with model parameters
w_0 = 1.0  # Interface thickness
m = 4.0  # Rotational symmetry order
epsilon_4 = 0.05  # Anisotropy strength
theta_0 = 0.0  # Offset angle
tau_0 = 1.0  # Relaxation time
D = fd.Constant(10.0)  # Diffusion coefficient
undercooling = -0.3  # Undercooling


def a(phi):
    # phi = 1e5*phi
    n = ufl.grad(phi) / ufl.sqrt(ufl.dot(ufl.grad(phi), ufl.grad(phi)) + 1e-5)

    theta = ufl.atan_2(n[1], n[0])

    n = 1e5 * n + ufl.as_vector([1e-5, 1e-5])
    xx = n[1] / n[0]
    # theta = ufl.asin(xx/ ufl.sqrt(1 + xx**2))
    theta = ufl.atan(xx)

    return 1.0 + epsilon_4 * ufl.cos(m * (theta - theta_0))


def gradient_energy_coefficient_func(phi):
    W = w_0 * a(phi)
    kappa = W ** 2
    return kappa


def kinetic_coefficient_func(phi):
    tau = tau_0 * a(phi) ** 2
    return tau


def free_energy_density(phi, U):
    lmbda = D * tau_0 / (0.6267 * w_0 ** 2)
    f_chem = (
        -0.5 * phi ** 2
        + 0.25 * phi ** 4
        + lmbda * U * phi * (1.0 - (2.0 / 3.0) * phi ** 2 + 0.2 * phi ** 4)
    )
    return f_chem


model_parameters = {
    "KineticCoefficient": kinetic_coefficient_func,
    "GradientEnergyCoefficient": gradient_energy_coefficient_func,
    "LocalFreeEnergyDensityFunction": free_energy_density,
    "ThermalDiffusion": lambda T: D,
}

model = AllenCahnHeatModel(W, model_parameters)

F = model.residual_form
w = model.solution_function
wdot = model.solutiondot_function

bcs = []

PETSc.Sys.Print(f"Problem is set up!")

if save_solution:
    output_file = fd.File("res3/output.pvd")


def monitor(ts, steps, time, X):
    phi, T = w.split()

    # PETSc.Sys.Print(f'Iteration #{steps}. Current time: {float(time)}, and solution norm is {fd.norm(w)}')

    if save_solution:
        output_file.write(phi, T, time=time)


problem = firedrake_ts.DAEProblem(F, w, wdot, (0.0, 1500), bcs=bcs)
solver = firedrake_ts.DAESolver(problem, monitor_callback=monitor)

solver.ts.setMaxSNESFailures(
    -1
)  # allow an unlimited number of failures (step will be rejected and retried)

snes = solver.ts.getSNES()  # Nonlinear solver
snes.setTolerances(
    max_it=10
)  # Stop nonlinear solve after 10 iterations (TS will retry with shorter step)

# Ensure everything is reset
x = fd.SpatialCoordinate(mesh)
w_init = create_initial_conditions(x, undercooling)
for i in range(len(w_init)):
    w.sub(i).interpolate(w_init[i])

PETSc.Sys.Print("Ready to start timestepping!")
PETSc.Sys.Print(f"Norm of initial conditions vector {fd.norm(w)}")

# And now actual solving procedure
t1 = time.time()

solver.solve()

t2 = time.time()

spent_time = t2 - t1
ts = solver.ts
PETSc.Sys.Print(f"Time spent is {spent_time}")
PETSc.Sys.Print(
    "steps %d (%d rejected, %d SNES fails), nonlinear its %d, linear its %d"
    % (
        ts.getStepNumber(),
        ts.getStepRejections(),
        ts.getSNESFailures(),
        ts.getSNESIterations(),
        ts.getKSPIterations(),
    )
)
