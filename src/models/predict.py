#-----------------  IMPORTACIÓN DE LAS HERRAMIENTAS NECESARIAS
import sys
import numpy as np
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))
root_path = Path(__file__).resolve().parents[2]

import config
from src.models.classical_DNN import ClassicalDNN
from src.handle_data.dataset import get_dataloader
from src.utils.plot_data import animate_solution


#-----------------  INICIALIZACIÓN DE LAS VARIABLES DE LA SIMULACIÓN

x_start = config.X_START
x_end = config.X_END
nx = config.NX_PRED
t_final = config.T_FINAL_SIMULATION
t_final_data = config.T_FINAL_DATA
condicion_inicial = config.CONDICION_INICIAL
simulation_id = config.SIMULATION_ID
model_type = config.MODEL_TYPE

model_filename = f"{model_type}_{simulation_id}.pth"
state_dict = torch.load(root_path / "models" / "checkpoints" / model_filename)
save_path = root_path / "animations" / f"{model_type}_{simulation_id}.gif"



#-----------------  PROGRAMA PRINCIPAL

def predict(x_start, x_end, nx, t_final):
    #Cargamos el modelo ya entrenado y lo ponemos en modo evaluación
    model = ClassicalDNN((x_start, x_end), (0.0, t_final_data))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Modelo {model_type} cargado exitosamente desde {model_filename}")

    #Creamos una malla de puntos (x,t) sobre la cual haremos la predicción
    X_mesh = np.linspace(x_start, x_end, nx)
    t_mesh = np.linspace(0, t_final, nx)
    X, T = np.meshgrid(X_mesh, t_mesh, indexing='ij')
    x_tensor = torch.from_numpy(X.flatten()).float().unsqueeze(1)
    t_tensor = torch.from_numpy(T.flatten()).float().unsqueeze(1)
    print(f"Malla de predicción creada correctamente")

    #Predecimos el valor de u en cada punto de la malla usando el modelo entrenado
    with torch.no_grad():
        predictions = model(torch.cat((t_tensor, x_tensor), dim=1))
    predictions = predictions.numpy().reshape(X.shape)
    print(f"Predicción completada exitosamente")
    return X_mesh, t_mesh, predictions


if __name__ == "__main__":
    X, T, U = predict(x_start, x_end, nx, t_final)
    animate_solution(X, T, U, save_path = save_path)