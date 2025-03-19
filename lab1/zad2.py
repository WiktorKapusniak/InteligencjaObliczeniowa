import math
import matplotlib.pyplot as plt
from random import randint

def oblicz_odleglosc(v, h, kat):
    """ Oblicza odległość, na jaką doleci pocisk """
    rad = math.radians(kat)
    sqrt_wartosc = v**2 * math.sin(rad)**2 + 2 * 9.81 * h
    if sqrt_wartosc < 0:
        return None  # Błędna wartość, zwracamy None
    d = (v * math.sin(rad) + math.sqrt(sqrt_wartosc)) * (v * math.cos(rad) / 9.81)
    return d

def rysuj_trajektorie(v, h, kat):
    """ Rysuje trajektorię lotu pocisku """
    rad = math.radians(kat)
    vx = v * math.cos(rad)
    vy = v * math.sin(rad)
    
    # Symulacja lotu
    t_max = (vy + math.sqrt(vy**2 + 2 * 9.81 * h)) / 9.81  # Całkowity czas lotu
    t_values = [i * 0.05 for i in range(int(t_max / 0.05) + 1)]  # Wartości czasu
    
    x_values = [vx * t for t in t_values]
    y_values = [h + vy * t - 0.5 * 9.81 * t**2 for t in t_values]
    
    # Rysowanie wykresu
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label=f"Trajektoria (kąt={kat}°)")
    plt.axhline(0, color='black', linewidth=0.5)  # Linia ziemi
    plt.axvline(0, color='black', linewidth=0.5)  # Początek
    plt.xlabel("Odległość (m)")
    plt.ylabel("Wysokość (m)")
    plt.title("Trajektoria lotu pocisku z trebusza")
    plt.legend()
    plt.savefig("trajektoria.png")
    print("Wykres został zapisany jako trajektoria.png")

def gra_w_trebusz():
    cel = randint(50, 340)
    margines = [cel - 5, cel + 5]
    v = 50
    h = 100
    proby = 0

    print(f"Cel znajduje się w odległości: {cel} m")
    
    while True:
        try:
            kat = int(input("Podaj kąt wyrzutu (w stopniach): "))
            odleglosc = oblicz_odleglosc(v, h, kat)
            
            if odleglosc is None:
                print("Niepoprawne dane, spróbuj ponownie.")
                continue

            print(f"Pocisk przeleciał: {round(odleglosc, 2)} m")

            if margines[0] <= odleglosc <= margines[1]:
                print("Cel trafiony!")
                rysuj_trajektorie(v, h, kat)
                break
            else:
                print("Pudło!")
                proby += 1
        except ValueError:
            print("Podaj poprawną liczbę!")
    
    print(f"Trafiłeś za {proby + 1} razem")

gra_w_trebusz()
