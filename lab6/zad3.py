import numpy as np
import pygad
import time

labirynt = np.array([
    [0,0,0,0,1,0,0,0,0,0],
    [1,1,1,0,1,0,1,1,1,0],
    [0,0,1,0,0,0,0,0,1,0],
    [0,0,1,1,1,1,1,0,1,0],
    [0,0,0,0,0,0,1,0,1,0],
    [0,1,1,1,1,0,1,0,1,0],
    [0,1,0,0,0,0,1,0,1,0],
    [0,1,0,1,1,1,1,0,1,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,1,1,1,1,1,1,1,1,0]
])

gene_space = [0, 1, 2, 3]
dlugosc_chromosomu = 30

def fitness_func(ga_instance, solution, solution_idx):
    x, y = 0, 0
    for move in solution:
        if move == 0 and x > 0 and labirynt[x-1][y] == 0:
            x -= 1
        elif move == 1 and x < 9 and labirynt[x+1][y] == 0:
            x += 1
        elif move == 2 and y > 0 and labirynt[x][y-1] == 0:
            y -= 1
        elif move == 3 and y < 9 and labirynt[x][y+1] == 0:
            y += 1

        if (x, y) == (9, 9):
            return 100


    dist = abs(9 - x) + abs(9 - y)
    return 100 - dist

# Statystyki
czasy = []
udane = 0

for i in range(10):
    start = time.time()

    ga_instance = pygad.GA(
        num_generations=1000,
        sol_per_pop=200,
        num_parents_mating=20,
        fitness_func=fitness_func,
        gene_space=gene_space,
        num_genes=dlugosc_chromosomu,
        parent_selection_type="tournament",
        mutation_type="random",
        mutation_percent_genes=15,
        stop_criteria=["reach_100"],
        suppress_warnings=True
    )

    ga_instance.run()
    end = time.time()

    best_fitness = ga_instance.best_solution()[1]
    print(f"Próba {i+1}: Fitness = {best_fitness}, Czas = {round(end - start, 3)}s")

    if best_fitness >= 100:
        udane += 1
        czasy.append(end - start)

# Podsumowanie
print("\n=== Statystyki ===")
print(f"Udało się znaleźć ścieżkę w {udane}/10 prób.")
print(f"Skuteczność: {udane*10}%")

if udane > 0:
    print(f"Średni czas sukcesu: {round(sum(czasy)/len(czasy), 3)}s")
else:
    print("Nie udało się znaleźć rozwiązania.")
print(f"Skuteczność: {round((udane / 10) * 100, 2)}%")

if udane > 0:
    sredni_czas = sum(czasy) / len(czasy)
    print(f"Średni czas udanej próby: {round(sredni_czas, 3)}s")
else:
    print("Nie udało się znaleźć żadnej ścieżki.")