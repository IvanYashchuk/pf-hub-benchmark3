"""
This is the code for Dendritic Growth benchmark.
The domain is a square with no-flux bcs.
Follow the link for the description of the benchmark:
https://pages.nist.gov/pfhub/benchmarks/benchmark3.ipynb/
"""

from collections import namedtuple
import firedrake as fd
import ufl


class ProblemModel:
    """Base class for problem formulation.
    Attributes:
        residual (Form): Residual formulation for the problem.
        jacobian (Form): Jacobian of the problem.
        solution_function (Function): Current time step solution.
        solutiondot_function (Function): Current time derivative solution.
        initial_function (Function): Initial.
    """

    def __init__(self):
        self._residual = None
        self._jacobian = None
        self._solution_function = None
        self._solutiondot_function = None
        self._initial_function = None

    @property
    def residual_form(self):
        """Return the residual.
        Returns:
            Form: UFL form representing the specified residual formulation.
        Raises:
            NotImplementedError: If the residual is not set.
        """
        return self._residual or NotImplementedError

    @property
    def jacobian_form(self):
        """Return the Jacobian.
        Returns:
            Form: UFL form representing the Jacobian.
        Raises:
            NotImplementedError: If the Jacobian is not set.
        """
        return self._jacobian or NotImplementedError

    @property
    def solution_function(self):
        """Return the current time step solution.
        Returns:
            Function: Current time step solution.
        Raises:
            NotImplementedError: If the solution_function is not set.
        """
        return self._solution_function or NotImplementedError

    @property
    def solutiondot_function(self):
        """Return the current time derivative.
        Returns:
            Function: Current time derivative.
        Raises:
            NotImplementedError: If the solution_function is not set.
        """
        return self._solutiondot_function or NotImplementedError


class AllenCahnHeatModel(ProblemModel):
    def __init__(self, function_space, model_parameters):
        super(AllenCahnHeatModel, self).__init__()

        self._solution_function = fd.Function(function_space)
        self._solutiondot_function = fd.Function(function_space)
        test_function = fd.TestFunction(function_space)

        # Split mixed FE function into separate functions and put them into namedtuple
        FiniteElementFunction = namedtuple("FiniteElementFunction", ["phi", "T"])

        split_solution_function = FiniteElementFunction(
            *ufl.split(self._solution_function)
        )
        split_solutiondot_function = FiniteElementFunction(
            *ufl.split(self._solutiondot_function)
        )
        split_test_function = FiniteElementFunction(*ufl.split(test_function))

        finite_element_functions = (
            split_solution_function,
            split_solutiondot_function,
            split_test_function,
        )

        F_AC = get_allen_cahn_weak_form(*finite_element_functions, model_parameters)
        F_heat = get_heat_weak_form(*finite_element_functions, model_parameters)

        F = F_AC + F_heat

        self._residual = F


def get_allen_cahn_weak_form(
    solution_functions, solutiondot_functions, test_functions, model_parameters
):
    """Return a weak form for Allen-Cahn equation.
    Allen-Cahn (AC) is solved to obtain the phase field.
    Args:
        solution_functions (tuple(Function)): Current time step solution.
        solutiondot_functions (tuple(Function)): Current time derivative.
        test_functions (tuple(TestFunction)): An appropriate test function.
        model_parameters (Dictionary): Model parameters.
    Returns:
        Form: UFL form representing the Allen-Cahn in residual formulation.
    """

    try:
        # Free energy density function may depend on temperature
        T = solution_functions.T
        dTdt = solutiondot_functions.T
    except AttributeError:
        raise AttributeError(
            "Current implementation Allen-Cahn model expects \
            Temperature function to couple with \
            (LocalFreeEnergyDensityFunction is assumed to \
            depend both on phase field and temperature functions)"
        )
        T = None
        dTdt = None

    phi = solution_functions.phi
    dphi_dt = solutiondot_functions.phi
    phi_ = test_functions.phi

    f_chem = model_parameters["LocalFreeEnergyDensityFunction"]
    tau = model_parameters["KineticCoefficient"]
    kappa = model_parameters["GradientEnergyCoefficient"]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    F_total = 0.5 * kappa(phi) * inner(grad(phi), grad(phi)) + f_chem(phi, T)

    # Functional derivative of F_total w.r.t phi in the test function direction phi_
    dFdphi = ufl.derivative(F_total, phi, phi_)
    # firedrake fails without the following call
    dFdphi = ufl.algorithms.expand_derivatives(dFdphi)

    F_AC = tau(phi) * inner(dphi_dt, phi_) * dx + dFdphi * dx

    return F_AC


def get_heat_weak_form(
    solution_functions, solutiondot_functions, test_functions, model_parameters
):
    """Return a weak form for convective Heat equation.
    The heat equation is solved to get temperature field. The equation is in the convective form.
    Solid-Liquid phase transition is modeled using nonlinear latent heat source.
    Args:
        solution_functions (tuple(Function)): Current time step solution.
        solutiondot_functions (tuple(Function)): Current time derivative.
        test_functions (tuple(TestFunction)): An appropriate test function.
        model_parameters (Dictionary): Model parameters.
    Returns:
        Form: UFL form representing the heat equation in residual formulation.
    """

    try:
        # Heat equation may depend on phase field
        phi = solution_functions.phi
        dphi_dt = solutiondot_functions.phi
    except AttributeError:
        phi = None

    T = solution_functions.T
    dT_dt = solutiondot_functions.T
    T_ = test_functions.T

    kappa = model_parameters["ThermalDiffusion"]

    inner, grad, dx = ufl.inner, ufl.grad, ufl.dx
    F_heat = inner(dT_dt, T_) * dx + inner(kappa(T) * grad(T), grad(T_)) * dx

    if phi is not None:
        F_heat += -0.5 * dphi_dt * T_ * dx

    return F_heat
