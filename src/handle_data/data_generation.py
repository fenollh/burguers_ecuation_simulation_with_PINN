import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fvm.fvm_solver import BurgersFVM

class GenerateData:
    def __init__(
        self, x_start=0.0, x_end=2.0 * np.pi, nx=400, t_final=2.0, cfl=0.9, ic='sine'
    ):
        self.x_start = x_start
        self.x_end = x_end
        self.nx = nx
        self.t_final = t_final
        self.cfl = cfl
        self.ic = ic

    def generate(self, animate=False):

        #-------------- Corremos la simulacion -------------------------

        solver = BurgersFVM(self.x_start, self.x_end, self.nx, self.t_final, self.cfl)
        """
        Corremos la simulación con condiciones iniciales concretas
        Nos devuelve:
           - X: Mallado sobre el que se ha corrido la simulación
           - t: Lista de tiempos en los cuales se ha calculado una solucion (Notar que la seleccion del incremento en tiempo es dinamica)
           - U: Matriz, historial de valores de u en cada (t,x)
        """
        match self.ic:
            case 'sine':
                X, t_eval, U = solver.solve(lambda x: np.sin(x))
            case 'step':
                X, t_eval, U = solver.solve(lambda x: 1 if x>=0 else 0)   

        #---------------- Guardamos los datos -----------------------------

        t_data = np.array(t_eval, dtype=np.float32)
        x_data = np.array(X, dtype=np.float32)
        u_data = np.array(U, dtype=np.float32)

        project_root = Path(__file__).resolve().parents[2]
        output_dir = project_root / "data" / "raw"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / "burguers_inviscid_raw.npz"

        if animate:
            from utils.plot_data import animate_solution
            animate_solution(x_data, t_data, u_data, save_path= project_root / "animations" / "fvm2.gif")

        np.savez_compressed(file_path, x=x_data, t=t_data, u=u_data)
