#-----------------  IMPORTACIÓN DE LAS HERRAMIENTAS NECESARIAS
import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

import config
from src.models.classical_DNN import ClassicalDNN
from src.handle_data.dataset import get_dataloader
from src.models.training_loop import training_loop
from src.utils.cost_functions import CustomLoss


#-----------------  INICIALIZACIÓN DE LAS VARIABLES DE ENTRENAMIENTO

x_start = config.X_START
x_end = config.X_END
t_final = config.T_FINAL_DATA
condicion_inicial = config.CONDICION_INICIAL
simulation_id = config.SIMULATION_ID
model_type = config.MODEL_TYPE

proyect_root = Path(__file__).resolve().parents[2]
data_path = proyect_root / "data" / "raw" / f"burguers_inviscid_raw_{simulation_id}.npz"
model_path = f"{model_type}_{simulation_id}.pth"
learning_rate = 0.01
pde_weight = 0.5



if __name__ == "__main__":

    #Instanciamos el modelo, dataloaders, loss function y optimizador
    model = ClassicalDNN((x_start, x_end), (0.0, t_final))
    dataloaders = get_dataloader(data_path, batch_size=16)
    print("Modelo y dataloaders creados exitosamente.")
    print(f"Se usarán {len(dataloaders[0].dataset)} muestras de entrenamiento")
    match model_type:
        case 'classical_DNN':
            loss_fn = torch.nn.MSELoss()
        case 'PINN':
            loss_fn = CustomLoss(pde_weight)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    #Entrenamos el modelo
    print("Iniciando entrenamiento del modelo...")
    training_loop(dataloaders[0], model, loss_fn, optimizer, model_path)
