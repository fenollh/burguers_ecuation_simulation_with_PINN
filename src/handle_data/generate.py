#IMPORTAMOS LAS HERRAMIENTAS NECESARIAS

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.handle_data.data_generation import GenerateData
import config

#INICIALIZACIÓN DE LAS VARIABLES DE LA SIMULACIÓN

x_start=config.X_START
x_end=config.X_END
nx=config.NX
t_final=config.T_FINAL_DATA
cfl=config.CFL
condicion_inicial=config.CONDICION_INICIAL
simulation_id=config.SIMULATION_ID

#EJECUCIÓN DE LA SIMULACIÓN Y GENERACION DE LOS DATOS Y ANIMACIONES

if __name__ == "__main__": 
    print("Iniciando programa")
    generate_data = GenerateData(x_start, x_end, nx, t_final, cfl, condicion_inicial, simulation_id)
    generate_data.generate(True)
    print("Finalizando programa")