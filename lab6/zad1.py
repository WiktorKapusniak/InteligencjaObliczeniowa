
import pygad
import numpy
import time

items = [
    {"nazwa": "zegar", "wartosc": 100, "waga": 7},
    {"nazwa": "obraz-pejzaż", "wartosc": 300, "waga": 7},
    {"nazwa": "obraz-portret", "wartosc": 200, "waga": 6},
    {"nazwa": "radio", "wartosc": 40, "waga": 2},
    {"nazwa": "laptop", "wartosc": 500, "waga": 5},
    {"nazwa": "lampka nocna", "wartosc": 70, "waga": 6},
    {"nazwa": "srebrne sztućce", "wartosc": 100, "waga": 1},
    {"nazwa": "porcelana", "wartosc": 250, "waga": 3},
    {"nazwa": "figura z brązu", "wartosc": 300, "waga": 10},
    {"nazwa": "skórzana torebka", "wartosc": 280, "waga": 3},
    {"nazwa": "odkurzacz", "wartosc": 300, "waga": 15}
]

# definiujemy parametry chromosomu
# geny to liczby: 0 lub 1
gene_space = [0, 1]

# definiujemy funkcję fitness
def fitness_func(ga_instance, solution, solution_idx):
    wartosc = 0
    waga = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            wartosc += items[i]["wartosc"]
            waga += items[i]["waga"]
    if waga > 25:
        return 0
    return wartosc
# inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
ga_instance = pygad.GA(gene_space=[0, 1],
        num_generations=100,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=len(items),
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10)

# uruchomienie algorytmu
ga_instance.run()

# podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))


# wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()

import time
import pygad
import numpy

successes = 0
czasy = []

for i in range(10):
    start = time.time()

    ga_instance = pygad.GA(
        gene_space=[0, 1],
        num_generations=100,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=20,
        num_genes=len(items),
        parent_selection_type="sss",
        keep_parents=2,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=10,
        stop_criteria=["reach_1630"]
    )

    ga_instance.run()

    solution, solution_fitness, _ = ga_instance.best_solution()

    end = time.time()
    elapsed = end - start

    if solution_fitness == 1630:
        successes += 1
        czasy.append(elapsed)

print(f"\nSkuteczność: {successes}/10 ({(successes / 10) * 100:.0f}%)")
if len(czasy) > 0:
    sredni_czas = sum(czasy) / len(czasy)
    print(f"Średni czas dla udanych prób: {sredni_czas:.4f} sekund")
else:
    print("Nie udało się znaleźć żadnego rozwiązania z wartością 1630.")
