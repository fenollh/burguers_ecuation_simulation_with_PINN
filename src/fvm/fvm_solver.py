import numpy as np
import matplotlib.pyplot as plt


class BurgersFVM:
    """Solucionador de la ecuación de Burgers inviscida 1D mediante Volúmenes Finitos
    y flujo numérico de Godunov.
    """

    def __init__(
        self, x_start=0.0, x_end=2.0 * np.pi, nx=400, t_final=2.0, cfl=0.9
    ):
        self.x_start = x_start
        self.x_end = x_end
        self.nx = nx
        self.t_final = t_final
        self.cfl = cfl

        # Malla espacial
        self.dx = (x_end - x_start) / nx
        self.x = np.linspace(x_start, x_end, nx, endpoint=False)

    @staticmethod
    def burgers_flux(u):
        """Flujo f(u) = 1/2 * u^2"""
        return 0.5 * u**2

    def godunov_flux(self, uL, uR):
        """Flujo numérico de Godunov"""
        flux = np.zeros_like(uL)

        # Caso Rarfacción: uL <= uR
        rarefaction = uL <= uR
        if np.any(rarefaction):
            uL_r = uL[rarefaction]
            uR_r = uR[rarefaction]
            flux_r = np.where(
                uL_r >= 0,
                self.burgers_flux(uL_r),
                np.where(uR_r <= 0, self.burgers_flux(uR_r), 0.0),
            )
            flux[rarefaction] = flux_r

        # Caso Choque: uL > uR
        shock = ~rarefaction
        if np.any(shock):
            uL_s = uL[shock]
            uR_s = uR[shock]
            s = 0.5 * (uL_s + uR_s)  # Velocidad del choque
            flux_s = np.where(
                s >= 0, self.burgers_flux(uL_s), self.burgers_flux(uR_s)
            )
            flux[shock] = flux_s

        return flux

    def apply_free_boundary(self, u):
        """Condición de contorno libre usando celdas fantasma."""
        u_ext = np.empty(len(u) + 2, dtype=u.dtype)
        u_ext[1:-1] = u
        u_ext[0] = u[0]
        u_ext[-1] = u[-1]
        return u_ext

    def solve(self, u0_func):
        """Ejecuta la simulación.

        - u0_func: Función que evalúa la condición inicial.
        """
        u = u0_func(self.x)

        # Malla temporal uniforme para el Dataset (Nt x Nx)
        U_history = [u.copy()]
        t_eval = [0.0]

        t = 0.0

        while t < self.t_final:
            max_speed = np.max(np.abs(u))
            dt = (
                self.cfl * self.dx / max_speed
                if max_speed >= 1e-12
                else self.cfl * self.dx
            )
            dt += min(0, self.t_final - t - dt )#controlamos no pasarnos de t_final
            
            # Actualización FVM
            u_ext = self.apply_free_boundary(u)
            flux = self.godunov_flux(u_ext[:-1], u_ext[1:])
            u = u - (dt / self.dx) * (flux[1:] - flux[:-1])
            t += dt

            U_history.append(u.copy())
            t_eval.append(t.copy())

        return self.x, t_eval, U_history