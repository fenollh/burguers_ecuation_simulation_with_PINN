import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import config

model_type = config.MODEL_TYPE

def training_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    save_path
):
    model.train()

    num_batches = len(dataloader)
    porcentajes_listados = []
    for batch, (X,y) in enumerate(dataloader):
        X.requires_grad_(True) #habilitamos el calculo de gradientes para X
        y_pred = model(X)
        match model_type:
            case 'classical_DNN':
                 loss = loss_fn(y_pred, y)
            case 'PINN':
                 loss = loss_fn(y_pred, y, X)
       

        loss.backward() #calculamos los gradientes de la funciones de perdida
        optimizer.step() #actualizamos los pesos de la red
        optimizer.zero_grad() #reiniciamos los gradientes para la siguiente iteracion

        #Para ir informando del progreso
        porcentaje = int(100 * (batch + 1) / num_batches)
        if porcentaje % 10 == 0 and porcentaje not in porcentajes_listados:
            print(f"Batch: {batch + 1} | Progreso: {porcentaje:.0f}%")
            porcentajes_listados.append(porcentaje)

    root_path = Path(__file__).resolve().parents[2]
    output_path = root_path / "models" / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / save_path

    torch.save(model.state_dict(), file_path)
    print(f"Parametros guardados exitosamente en {file_path}")
