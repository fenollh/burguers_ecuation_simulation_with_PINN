import numpy as np

CONDICION_INICIAL = 'dsin'  # 'sine', 'step', 'linear', 'tanh', 'dsin
X_START = 0
X_END = 2*np.pi

T_FINAL_DATA = 0.2
T_FINAL_SIMULATION = 3.0

MODEL_TYPE = 'PINN'  # 'classical_DNN', 'PINN'
NX = 3000
NX_PRED = 500
CFL = 0.9

SIMULATION_ID = "dsin_1"
