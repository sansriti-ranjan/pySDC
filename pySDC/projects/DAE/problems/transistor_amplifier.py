import warnings
import numpy as np
from scipy.interpolate import interp1d

from pySDC.projects.DAE.misc.ProblemDAE import ptype_dae
from pySDC.implementations.datatype_classes.mesh import mesh


# Helper function
def _transistor(u_in):
    return 1e-6 * (np.exp(u_in / 0.026) - 1)

class one_transistor_amplifier(ptype_dae): 
    """
    The one transistor amplifier example from pg. 404 Solving ODE II by Hairer and Wanner
    The problem is an index-1 DAE
    """
    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        super(one_transistor_amplifier, self).__init__(problem_params, dtype_u, dtype_f)
        # load reference solution 
        data = np.load(r'pySDC/projects/DAE/misc/data/one_trans_amp.npy')
        x = data[:, 0]
        # The last column contains the input signal
        y = data[:, 1:-1]
        self.u_ref = interp1d(x, y, kind='cubic', axis=0, fill_value='extrapolate')
        self.t_end = x[-1]

    def eval_f(self, u, du, t):
        """
        Routine to evaluate the implicit representation of the problem i.e. F(u', u, t)
        Args:
            u (dtype_u): the current values. This parameter has been "hijacked" to contain [u', u] in this case to enable evaluation of the implicit representation
            t (float): current time
        Returns:
            Current value of F(), 8 components
        """
        du = u[5:10]
        u = u[0:5]
        u_b = 6.
        u_e = 0.4 * np.sin(200 * np.pi * t)
        alpha = 0.99
        r_0 = 1000
        r_k = 9000
        c_1, c_2, c_3 = 1e-6, 2e-6, 3e-6
        f = self.dtype_f(self.init)
        f[:] = ((u_e - u[0]) / r_0 + c_1 * (du[1] - du[0]),
                     (u_b - u[1]) / r_k - u[1] / r_k + c_1 * (du[0] - du[1]) - (1 - alpha) * _transistor(u[1] - u[2]),
                     _transistor(u[1] - u[2]) - u[2] / r_k - c_2 * du[2],
                     (u_b - u[3]) / r_k + c_3 * (du[4] - du[3]) - alpha * _transistor(u[1] - u[2]),
                     -u[4] / r_k + c_3 * (du[3] - du[4]))
        return f
        
    def u_exact(self, t):
        """
        Approximation of the exact solution generated by spline interpolation of an extremely accurate numerical reference solution. 
        Args: 
            t (float): current time
        Returns: 
            Mesh containing fixed initial value, 5 components
        """
        me = self.dtype_u(self.init)
        
        if t < self.t_end:
            me[:] = self.u_ref(t)
        else:
            warnings.warn("Requested time exceeds domain of the reference solution. Returning zero.")
            me[:] = (0, 0, 0, 0, 0)
        return me

class two_transistor_amplifier(ptype_dae): 
    """
    The two transistor amplifier example from page 108 "The numerical solution of differential-algebraic systems by Runge-Kutta methods" Hairer et al. 
    The problem is an index-1 DAE
    """
    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        super(two_transistor_amplifier, self).__init__(problem_params, dtype_u, dtype_f)
        # load reference solution 
        data = np.load(r'pySDC/projects/DAE/misc/data/two_trans_amp.npy')
        x = data[:, 0]
        # The last column contains the input signal
        y = data[:, 1:-1]
        self.u_ref = interp1d(x, y, kind='cubic', axis=0, fill_value='extrapolate')
        self.t_end = x[-1]

    def eval_f(self, u, du, t):
        """
        Routine to evaluate the implicit representation of the problem i.e. F(u', u, t)
        Args:
            u (dtype_u): the current values. This parameter has been "hijacked" to contain [u', u] in this case to enable evaluation of the implicit representation
            t (float): current time
        Returns:
            Current value of F(), 8 components
        """
        du = u[8:16]
        u = u[0:8]
        u_b = 6.
        u_e = 0.1 * np.sin(200 * np.pi * t)
        alpha = 0.99
        r_0 = 1000.
        r_k = 9000.
        c_1, c_2, c_3, c_4, c_5 = 1e-6, 2e-6, 3e-6, 4e-6, 5e-6
        f = self.dtype_f(self.init)
        f[:] =((u_e - u[0]) / r_0 - c_1 * (du[0] - du[1]),
                        (u_b - u[1]) / r_k - u[1] / r_k + c_1 * (du[0] - du[1]) + (alpha - 1) * _transistor(u[1] - u[2]),
                        _transistor(u[1] - u[2]) - u[2] / r_k - c_2 * du[2],
                        (u_b - u[3]) / r_k - c_3 * (du[3] - du[4]) - alpha * _transistor(u[1] - u[2]),
                        (u_b - u[4]) / r_k - u[4] / r_k + c_3 * (du[3] - du[4]) + (alpha - 1) * _transistor(u[4] - u[5]),
                        _transistor(u[4] - u[5]) - u[5] / r_k - c_4 * du[5],
                        (u_b - u[6]) / r_k - c_5 * (du[6] - du[7]) - alpha * _transistor(u[4] - u[5]),
                        -u[7] / r_k + c_5 * (du[6] - du[7]))
        return f
        
    def u_exact(self, t):
        """
        Dummy exact solution that should only be used to get initial conditions for the problem
        This makes initialisation compatible with problems that have a known analytical solution 
        Could be used to output a reference solution if generated/available
        Args: 
            t (float): current time 
        Returns: 
            Mesh containing fixed initial value, 5 components
        """
        me = self.dtype_u(self.init)
        
        if t < self.t_end:
            me[:] = self.u_ref(t)
        else:
            warnings.warn("Requested time exceeds domain of the reference solution. Returning zero.")
            me[:] = (0, 0, 0, 0, 0, 0, 0, 0)
        return me
