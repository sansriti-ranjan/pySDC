import cupy as cp
import numpy as np

from pySDC.core.Errors import ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.cupy_mesh import cupy_mesh, imex_cupy_mesh


class allencahn2d_imex(ptype):  # pragma: no cover
    """
    Example implementing Allen-Cahn equation in 2D using FFTs for solving linear parts, IMEX time-stepping

    Attributes:
        xvalues: grid points in space
        dx: cupy_mesh width
        lap: spectral operator for Laplacian
        rfft_object: planned real FFT for forward transformation
        irfft_object: planned IFFT for backward transformation
    """

    dtype_u = cupy_mesh
    dtype_f = imex_cupy_mesh

    def __init__(self, nvars, nu, eps, radius, L=1.0, init_type='circle'):
        """Initialization routine"""

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if len(nvars) != 2:
            raise ProblemError('this is a 2d example, got %s' % nvars)
        if nvars[0] != nvars[1]:
            raise ProblemError('need a square domain, got %s' % nvars)
        if nvars[0] % 2 != 0:
            raise ProblemError('the setup requires nvars = 2^p per dimension')

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super().__init__(init=(nvars, None, cp.dtype('float64')))
        self._makeAttributeAndRegister(
            'nvars', 'nu', 'eps', 'radius', 'L', 'init_type', localVars=locals(), readOnly=True
        )

        self.dx = self.L / self.nvars[0]  # could be useful for hooks, too.
        self.xvalues = cp.array([i * self.dx - self.L / 2.0 for i in range(self.nvars[0])])

        kx = cp.zeros(self.init[0][0])
        ky = cp.zeros(self.init[0][1] // 2 + 1)

        kx[: int(self.init[0][0] / 2) + 1] = 2 * np.pi / self.L * cp.arange(0, int(self.init[0][0] / 2) + 1)
        kx[int(self.init[0][0] / 2) + 1 :] = (
            2 * np.pi / self.L * cp.arange(int(self.init[0][0] / 2) + 1 - self.init[0][0], 0)
        )
        ky[:] = 2 * np.pi / self.L * cp.arange(0, self.init[0][1] // 2 + 1)

        xv, yv = cp.meshgrid(kx, ky, indexing='ij')
        self.lap = -(xv**2) - yv**2

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)
        v = u.flatten()
        tmp = self.lap * cp.fft.rfft2(u)
        f.impl[:] = cp.fft.irfft2(tmp)
        if self.eps > 0:
            f.expl[:] = (1.0 / self.eps**2 * v * (1.0 - v**self.nu)).reshape(self.nvars)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)
        tmp = cp.fft.rfft2(rhs) / (1.0 - factor * self.lap)
        me[:] = cp.fft.irfft2(tmp)

        return me

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        assert t == 0, 'ERROR: u_exact only valid for t=0'
        me = self.dtype_u(self.init, val=0.0)
        if self.init_type == 'circle':
            xv, yv = cp.meshgrid(self.xvalues, self.xvalues, indexing='ij')
            me[:, :] = cp.tanh((self.radius - cp.sqrt(xv**2 + yv**2)) / (cp.sqrt(2) * self.eps))
        elif self.init_type == 'checkerboard':
            xv, yv = cp.meshgrid(self.xvalues, self.xvalues)
            me[:, :] = cp.sin(2.0 * np.pi * xv) * cp.sin(2.0 * np.pi * yv)
        elif self.init_type == 'random':
            me[:, :] = cp.random.uniform(-1, 1, self.init)
        else:
            raise NotImplementedError('type of initial value not implemented, got %s' % self.init_type)

        return me


class allencahn2d_imex_stab(allencahn2d_imex):
    """
    Example implementing Allen-Cahn equation in 2D using FFTs for solving linear parts, IMEX time-stepping with
    stabilized splitting

    Attributes:
        xvalues: grid points in space
        dx: mesh width
        lap: spectral operator for Laplacian
        rfft_object: planned real FFT for forward transformation
        irfft_object: planned IFFT for backward transformation
    """

    def __init__(self, nvars, nu, eps, radius, L=1.0, init_type='circle'):
        super().__init__(nvars, nu, eps, radius, L, init_type)
        self.lap -= 2.0 / self.eps**2

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """

        f = self.dtype_f(self.init)
        v = u.flatten()
        tmp = self.lap * cp.fft.rfft2(u)
        f.impl[:] = cp.fft.irfft2(tmp)
        if self.eps > 0:
            f.expl[:] = (1.0 / self.eps**2 * v * (1.0 - v**self.nu) + 2.0 / self.eps**2 * v).reshape(self.nvars)
        return f

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple FFT solver for the diffusion part

        Args:
            rhs (dtype_f): right-hand side for the linear system
            factor (float) : abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver (not used here so far)
            t (float): current time (e.g. for time-dependent BCs)

        Returns:
            dtype_u: solution as mesh
        """

        me = self.dtype_u(self.init)

        tmp = cp.fft.rfft2(rhs) / (1.0 - factor * self.lap)
        me[:] = cp.fft.irfft2(tmp)

        return me
