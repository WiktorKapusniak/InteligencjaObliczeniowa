import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")


# COORDS = (
#     (20, 52),
#     (43, 50),
#     (20, 84),
#     (70, 65),
#     (29, 90),
#     (87, 83),
#     (73, 23),
# )
COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (14,89),
    (59,81),
    (1,1),
    (98,87)
)

def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


plot_nodes()

# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,
#                     pheromone_evaporation_rate=0.40, pheromone_constant=1000.0,
#                     iterations=300)

colony = AntColony(COORDS, ant_count=500, alpha=0.65, beta=1.75,
                    pheromone_evaporation_rate=0.60, pheromone_constant=500.0,
                    iterations=700)
optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )


plt.show()



# 1 Parowanie feromonów (pheromone_evaporation_rate):
#    - 0.5-0.6 – lepsza eksploracja kosztem stabilności.

# 2 Współczynniki alpha i beta:
#    - alpha: 0.7-0.8 – większy wpływ feromonów.
#    - beta: 1.5-2.0 – większa rola odległości przy dużej liczbie punktów.

# 3 Stała feromonowa (pheromone_constant):
#    - 500.0 – mniej feromonów, lepsza eksploracja.

# 4 Iteracje (iterations):
#    700 – większa dokładność, ale dłuższy czas działania.