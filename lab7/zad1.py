import numpy as np
import math
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history


def endurance(x):
    y, z, u, v, w = x[1], x[2], x[3], x[4], x[5]
    return np.exp(-2 * (y - np.sin(x[0])) ** 2) + np.sin(z * u) + np.cos(v * w)

def objective_function(swarm):
    return -np.array([endurance(particle) for particle in swarm])

options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

# Zakresy dla wszystkich sześciu zmiennych (x, y, z, u, v, w) w przedziale [0, 1)
x_max = np.ones(6)
x_min = np.zeros(6)
my_bounds = (x_min, x_max)

optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)
cost, pos = optimizer.optimize(objective_function, iters=100)

print(f"Najlepszy koszt: {cost}")
print(f": {pos}")

plot_cost_history(optimizer.cost_history)
plt.show()