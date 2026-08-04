import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))

from data_generation import GenerateData

x_start=0.0
x_end=4.0 * np.pi
nx=400
t_final=4.0
cfl=0.9
ic='sine'

if __name__ == "__main__": 
    print("Iniciando programa")
    generate_data = GenerateData(x_start, x_end, nx, t_final, cfl, ic)
    generate_data.generate(True)
    print("Finalizando programa")