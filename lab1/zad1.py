import datetime
import math
def main():
    imie = input("Podaj imie: ")
    nazwisko = input("Podaj nazwisko: ")
    rok_urodzenia = int(input("Podaj rok urodzenia: "))
    miesiac_urodzenia = int(input("Podaj miesiac urodzenia: "))
    dzien_urodzenia = int(input("Podaj dzien urodzenia: "))
    data_urodzenia = datetime.date(rok_urodzenia, miesiac_urodzenia, dzien_urodzenia)
    dzisiaj = datetime.date.today()
    ilosc_dni = (dzisiaj - data_urodzenia).days
    
    fizyczna_fala =  math.sin(((2*math.pi)/23)*ilosc_dni)
    emocjonalna_fala = math.sin(((2*math.pi)/28)*ilosc_dni)
    intelektualna_fala = math.sin(((2*math.pi)/33)*ilosc_dni)
    opis_fizyczna_fala = "jestes w fazie neutralnej energii fizycznej"
    opis_emocjonalna_fala = "jestes w fazie neutralnej energii emocjonalnej"
    opis_intelektualna_fala = "jestes w fazie neutralnej energii intelektualnej"
        
    if fizyczna_fala > 0.5:
        opis_fizyczna_fala = "jestes w fazie wysokiej energii fizycznej"
    elif fizyczna_fala < -0.5:
        opis_fizyczna_fala = "jestes w fazie niskiej energii fizycznej"
        nastepny_dzien = ilosc_dni+1
        if math.sin(((2*math.pi)/23)*nastepny_dzien) > fizyczna_fala:
            opis_fizyczna_fala += " ale jutro bedzie lepiej"    
    
    if emocjonalna_fala > 0.5:
        opis_emocjonalna_fala = "jestes w fazie wysokiej energii emocjonalnej"
    elif emocjonalna_fala < -0.5:
        opis_emocjonalna_fala = "jestes w fazie niskiej energii emocjonalnej"
        nastepny_dzien = ilosc_dni+1
        if math.sin(((2*math.pi)/28)*nastepny_dzien) > emocjonalna_fala:
            opis_emocjonalna_fala += " ale jutro bedzie lepiej"
        
    if intelektualna_fala > 0.5:
        opis_intelektualna_fala = "jestes w fazie wysokiej energii intelektualnej"
    elif intelektualna_fala < -0.5:
        opis_intelektualna_fala = "jestes w fazie niskiej energii intelektualnej"
        nastepny_dzien = ilosc_dni+1
        if math.sin(((2*math.pi)/33)*nastepny_dzien) > intelektualna_fala:
            opis_intelektualna_fala += " ale jutro bedzie lepiej"
        
    return f"Hello {imie} {nazwisko}, zyjesz juz {ilosc_dni} dni. Twoje biorytmy to: \nFizyczny: {fizyczna_fala} - {opis_fizyczna_fala}\nEmocjonalny: {emocjonalna_fala} - {opis_emocjonalna_fala}\nIntelektualny: {intelektualna_fala} - {opis_intelektualna_fala}"


    
    
print(main())




#na pisanie tego programu poswiecilem około 20 minut