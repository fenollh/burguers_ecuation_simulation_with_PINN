import sys
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))
root_path = Path(__file__).resolve().parents[2]

from src.models.classical_DNN import ClassicalDNN
from src.handle_data.dataset import get_dataloader
from src.utils.plot_data import animate_solution

def predict(x_start, x_end, nx, t_final):
    model = ClassicalDNN()
    state_dict = torch.load(root_path / "models" / "checkpoints" / "classical_DNN.pth")
    model.load_state_dict(state_dict)
    model.eval()

    X_mesh = np.linspace(x_start, x_end, nx)
    t_mesh = np.linspace(0, t_final, nx)
    X, T = np.meshgrid(X_mesh, t_mesh, indexing='ij')

    x_tensor = torch.from_numpy(X.flatten()).float().unsqueeze(1)
    t_tensor = torch.from_numpy(T.flatten()).float().unsqueeze(1)

    with torch.no_grad():
        predictions = model(torch.cat((t_tensor, x_tensor), dim=1))
    predictions = predictions.numpy().reshape(X.shape)
    return X_mesh, t_mesh, predictions


if __name__ == "__main__":
    x_start = 0.0
    x_end = 4.0 * np.pi
    nx = 400
    t_final = 4.0

    X, T, U = predict(x_start, x_end, nx, t_final)
    print("Predicciones realizadas exitosamente.")
    animate_solution(X, T, U, save_path = root_path / "animations" / "classical_DNN2.gif")