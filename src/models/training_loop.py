import torch
from pathlib import Path

def training_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    save_path
):
    model.train()

    for batch, (X,y) in enumerate(dataloader):
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        loss.backward() #calculamos los gradientes de la funciones de perdida
        optimizer.step() #actualizamos los pesos de la red
        optimizer.zero_grad() #reiniciamos los gradientes para la siguiente iteracion

    root_path = Path(__file__).resolve().parents[2]
    output_path = root_path / "models" / "checkpoints"
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / save_path

    torch.save(model.state_dict(), file_path)
    print(f"Parametros guardados exitosamente en {file_path}")
