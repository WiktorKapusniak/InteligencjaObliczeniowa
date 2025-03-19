from datetime import datetime

def oblicz_biorytmy(dni_zycia):
    """Oblicza biorytmy na podstawie wzorów sinusoidalnych."""
    fizyczny = round(sin((2 * pi * dni_zycia) / 23), 2)
    emocjonalny = round(sin((2 * pi * dni_zycia) / 28), 2)
    intelektualny = round(sin((2 * pi * dni_zycia) / 33), 2)
    return fizyczny, emocjonalny, intelektualny

def ocena_wyniku(fizyczny, emocjonalny, intelektualny):
    """Ocenia wyniki i wyświetla odpowiedni komunikat."""
    if fizyczny > 0.5 or emocjonalny > 0.5 or intelektualny > 0.5:
        print("Gratulacje! Masz dziś dobry dzień! 😊")
    elif fizyczny < -0.5 or emocjonalny < -0.5 or intelektualny < -0.5:
        print("Nie przejmuj się, dziś może być trudniej, ale jutro będzie lepiej! 💪")
    else:
        print("To neutralny dzień, wszystko może się zdarzyć! 😃")

def main():
    """Główna funkcja programu."""
    # Pobranie danych od użytkownika
    imie = input("Podaj swoje imię: ")
    rok_urodzenia = int(input("Podaj rok urodzenia (YYYY): "))
    miesiac_urodzenia = int(input("Podaj miesiąc urodzenia (MM): "))
    dzien_urodzenia = int(input("Podaj dzień urodzenia (DD): "))

    # Obliczenie liczby dni od urodzenia
    data_urodzenia = datetime(rok_urodzenia, miesiac_urodzenia, dzien_urodzenia)
    dzisiaj = datetime.today()
    dni_zycia = (dzisiaj - data_urodzenia).days

    # Obliczenie biorytmów
    fizyczny, emocjonalny, intelektualny = oblicz_biorytmy(dni_zycia)

    # Wyświetlenie wyników
    print(f"\nCześć, {imie}! Dzisiaj masz {dni_zycia} dzień życia.")
    print(f"Twoje biorytmy na dziś:")
    print(f"📊 Fizyczny: {fizyczny}")
    print(f"💙 Emocjonalny: {emocjonalny}")
    print(f"🧠 Intelektualny: {intelektualny}")

    # Ocena wyniku i odpowiedni komunikat
    ocena_wyniku(fizyczny, emocjonalny, intelektualny)

if __name__ == "__main__":
    from math import sin, pi
    main()
