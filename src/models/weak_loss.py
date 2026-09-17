import torch
import torch.nn as nn

#TODA ESTE ARCHIVO ESTA MAL

class WeakBurgersLoss(nn.Module):

    def __init__(
        self,
        x_bounds=(0.0, 4.0 * torch.pi),
        t_bounds=(0.0, 3.0),
        num_test_x=4,
        num_test_t=4,
        lambda_ph=0.01,
    ): 
        super().__init__()
        self.lambda_ph = lambda_ph
        self.mse = nn.MSELoss()

        self.register_buffer("x_min", torch.tensor(x_bounds[0]))
        self.register_buffer("t_min", torch.tensor(t_bounds[0]))

        self.Lx = x_bounds[1] - x_bounds[0]
        self.Lt = t_bounds[1] - t_bounds[0]

        # Frecuencias test.
        # m y n representan el numero de funciones test en espacio y tiempo. Cada funcion test
        # sera una combinacion de senos y cosenos con diferentes frecuencias.
        m = torch.arange(1, num_test_x + 1, dtype=torch.float32)
        n = torch.arange(1, num_test_t + 1, dtype=torch.float32)
        grid_m, grid_n = torch.meshgrid(m, n, indexing="ij")

        self.register_buffer("m", grid_m.reshape(1, -1))
        self.register_buffer("n", grid_n.reshape(1, -1))

    def forward(self, y_pred, y_true, X):
        # 1. Pérdida de los datos. Calculada con MSE
        mse_loss = self.mse(y_pred, y_true)

        # 2. Extraer coordenadas y normalizar a [0, 1]
        t = X[:, 0:1]
        x = X[:, 1:2]

                #OJOOOOO: Aqui no faltaria un -1 para que normalice 
                # a [-1, 1] como en la capa de normalizacion?

        x_tilde = (x - self.x_min) / self.Lx
        t_tilde = (t - self.t_min) / self.Lt

        # 3. Funciones test y sus derivadas analíticas
        sin_m_x = torch.sin(self.m * torch.pi * x_tilde)
        cos_m_x = torch.cos(self.m * torch.pi * x_tilde)

        sin_n_t = torch.sin(self.n * torch.pi * t_tilde)
        cos_n_t = torch.cos(self.n * torch.pi * t_tilde)

        v_t = (self.n * torch.pi / self.Lt) * sin_m_x * cos_n_t
        v_x = (self.m * torch.pi / self.Lx) * cos_m_x * sin_n_t

        # 4. Formulación débil: + u * v_t + (0.5 * u^2) * v_x
        flux = 0.5 * (y_pred**2)
        integrand = +y_pred * v_t + flux * v_x

        # 5. Promedio sobre el batch (estabilidad numérica)
        weak_residuals = torch.mean(integrand, dim=0)
        weak_pde_loss = torch.mean(weak_residuals**2)

        # Evitar NaN devolviendo solo valores válidos
        if torch.isnan(weak_pde_loss) or torch.isinf(weak_pde_loss):
            return mse_loss

        return (1 - self.lambda_ph) * mse_loss + self.lambda_ph * weak_pde_loss
