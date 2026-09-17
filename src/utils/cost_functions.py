import torch 
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, pde_weight):
        super(CustomLoss, self).__init__()
        self.pde_weight = pde_weight #Peso de MSE

    def forward(self, predictions, targets, X):
        mse_loss = torch.mean((predictions - targets) ** 2)
        X_grads = torch.autograd.grad(predictions, X, 
                                      grad_outputs=torch.ones_like(predictions), 
                                      create_graph=True)[0]
        u_t = X_grads[:, 0:1]
        u_x = X_grads[:, 1:2]

        pde_loss = torch.mean((u_t + predictions*u_x)**2)
        total_loss = (1 - self.pde_weight) * mse_loss + self.pde_weight * pde_loss
        return total_loss
