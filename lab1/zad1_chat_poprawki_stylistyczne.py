import datetime
import math

def oblicz_biorytmy(ilosc_dni):
    """Oblicza wartości fal biorytmicznych."""
    return {
        "fizyczna": math.sin(((2 * math.pi) / 23) * ilosc_dni),
        "emocjonalna": math.sin(((2 * math.pi) / 28) * ilosc_dni),
        "intelektualna": math.sin(((2 * math.pi) / 33) * ilosc_dni)
    }

def okresl_faze(fala, okres, ilosc_dni):
    """Określa fazę biorytmu i ewentualną poprawę następnego dnia."""
    if fala > 0.5:
        return "jesteś w fazie wysokiej energii"
    elif fala < -0.5:
        nastepny_dzien = math.sin(((2 * math.pi) / okres) * (ilosc_dni + 1))
        return "jesteś w fazie niskiej energii, ale jutro będzie lepiej" if nastepny_dzien > fala else "jesteś w fazie niskiej energii"
    return "jesteś w fazie neutralnej energii"

def main():
    """Główna funkcja programu."""
    imie = input("Podaj imię: ")
    nazwisko = input("Podaj nazwisko: ")
    rok_urodzenia = int(input("Podaj rok urodzenia: "))
    miesiac_urodzenia = int(input("Podaj miesiąc urodzenia: "))
    dzien_urodzenia = int(input("Podaj dzień urodzenia: "))

    data_urodzenia = datetime.date(rok_urodzenia, miesiac_urodzenia, dzien_urodzenia)
    dzisiaj = datetime.date.today()
    ilosc_dni = (dzisiaj - data_urodzenia).days

    biorytmy = oblicz_biorytmy(ilosc_dni)
    opis_fizyczny = okresl_faze(biorytmy["fizyczna"], 23, ilosc_dni)
    opis_emocjonalny = okresl_faze(biorytmy["emocjonalna"], 28, ilosc_dni)
    opis_intelektualny = okresl_faze(biorytmy["intelektualna"], 33, ilosc_dni)

    return (f"Hello {imie} {nazwisko}, żyjesz już {ilosc_dni} dni. Twoje biorytmy to:\n"
            f"Fizyczny: {biorytmy['fizyczna']:.3f} - {opis_fizyczny}\n"
            f"Emocjonalny: {biorytmy['emocjonalna']:.3f} - {opis_emocjonalny}\n"
            f"Intelektualny: {biorytmy['intelektualna']:.3f} - {opis_intelektualny}")

print(main())