import sys
from pathlib import Path
import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.classical_DNN import ClassicalDNN
from src.handle_data.dataset import get_dataloader
from src.models.training_loop import training_loop

proyect_root = Path(__file__).resolve().parents[2]
data_path = proyect_root / "data" / "raw" / "burguers_inviscid_raw.npz"

def main():
    model = ClassicalDNN()
    print("Modelo creado exitosamente.")
    dataloaders = get_dataloader(data_path, batch_size=16)
    print("Dataloaders creados exitosamente.")
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    save_path = "classical_DNN.pth"
    print("Iniciando entrenamiento del modelo...")
    training_loop(dataloaders[0], model, loss_fn, optimizer, save_path)

if __name__ == "__main__":
    main()
