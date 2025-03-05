import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import re
numar_joc = 4
if not os.path.exists("./imagini_txt"):
    os.makedirs("./imagini_txt")
#----------------------------------------------------------------------------------------------------------------------
def scoate_imagine_tabla(image_path):
    image = cv2.imread(image_path)

    blue, green, red = cv2.split(image)

    #aplicam filtrul Gaussian pentru reducerea zgomotului
    blurred_image = cv2.GaussianBlur(blue, (5, 5), 0)

    #aplicam thresholding
    _, thresh = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    suprafata_maxima = 0
    x_max = 0
    y_max = 0
    w_max = 0
    h_max = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        suprafata = w*h
        if suprafata > suprafata_maxima:
            suprafata_maxima = suprafata
            x_max = x
            y_max = y
            w_max = w
            h_max = h

    cropped_image = image[y_max:y_max+h_max, x_max:x_max+w_max]

    #extrag conturul tablei de joc

    x = 260
    y = 260
    w = 1460
    h = 1470
    cropped_board = cropped_image[y:y+h, x:x+w]

    cropped_board = cv2.resize(cropped_board, (1400, 1400))

    return cropped_board


templates_tabla = scoate_imagine_tabla("./imagini_auxiliare/04.jpg")
cv2.imwrite("./templates/imagine.jpg", templates_tabla)


def creeaza_templates():
    # Creez template-urile pentru cifre
    tabla = scoate_imagine_tabla("./imagini_auxiliare/04.jpg")

    lista_templates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,27,28,30,32,35,36,40,42,45,48,49,50,54,56,60,63,64,70,72,80,81,90]
    i = 0

    index_linie = 0
    index_coloana = 0

    while index_linie < 13:
        if index_linie == 12:
            while index_coloana < 7:
                x = index_coloana*100
                y = index_linie*100
                roi = tabla[y:y+100, x:x+100]
                roi = cv2.resize(roi, (100, 100))
                height, width = roi.shape[:2]

                center_x, center_y = width // 2, height // 2

                new_width , new_height = int(width / 1.1), int(height / 1.1)

                x1 = max(center_x - new_width // 2, 0)
                x2 = min(center_x + new_width // 2, width)
                y1 = max(center_y - new_height // 2, 0)
                y2 = min(center_y + new_height // 2, height)

                roi = roi[y1:y2, x1:x2]

                roi = cv2.resize(roi, (100, 100))
                cv2.imwrite(f"./templates/{lista_templates[i]}.jpg", roi)
                i += 1
                index_coloana += 2
        else:
            while index_coloana < 14:
                x = index_coloana*100
                y = index_linie*100
                roi = tabla[y:y+100, x:x+100]
                roi = cv2.resize(roi, (100, 100))
                height, width = roi.shape[:2]

                center_x, center_y = width // 2, height // 2

                new_width , new_height = int(width / 1.1), int(height / 1.1)

                x1 = max(center_x - new_width // 2, 0)
                x2 = min(center_x + new_width // 2, width)
                y1 = max(center_y - new_height // 2, 0)
                y2 = min(center_y + new_height // 2, height)

                roi = roi[y1:y2, x1:x2]

                roi = cv2.resize(roi, (100, 100))
                cv2.imwrite(f"./templates/{lista_templates[i]}.jpg", roi)
                i += 1
                index_coloana += 2
        index_linie += 2
        index_coloana = 0
    pass

# creeaza_templates()

matrice_piese = np.zeros((14, 14))
def resize_image(image, size=(100, 100)):
    return cv2.resize(image, size)

def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def compare_histograms(image, template_path):
    image = resize_image(image)
    image = normalize_image(image).flatten()
    image_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    image_hist = cv2.normalize(image_hist, image_hist).flatten()

    template_path = resize_image(template_path)
    template_path = normalize_image(template_path).flatten()
    template_hist = cv2.calcHist([template_path], [0], None, [256], [0, 256])
    template_hist = cv2.normalize(template_hist, template_hist).flatten()

    similarity = cv2.compareHist(image_hist, template_hist, cv2.HISTCMP_BHATTACHARYYA)

    similarity = 1 - similarity

    return similarity

lista_templates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,24,25,27,28,30,32,35,36,40,42,45,48,49,50,54,56,60,63,64,70,72,80,81,90]

def gaseste_pozitie_ultima_piesa(ultima_poza,poza_curenta):
    image = scoate_imagine_tabla(ultima_poza)
    image2 = scoate_imagine_tabla(poza_curenta)

    #gaseste pozitia piesei din poza_curenta in poza_ultima
    #pentru asta parcurg fiecare patratel si compar culoarea medie a acestuia
    #daca culoarea medie a patratelului din poza_curenta este diferita de culoarea medie a patratelului din poza_ultima
    #atunci am gasit pozitia piesei dare iau in considerare doar patratelul care are cea mai mare diferenta de culoare

    max_diff = 0

    for i in range(14):
        for j in range(14):
            x = i*100
            y = j*100
            culoare_medie = np.mean(image[y:y+100, x:x+100], axis=(0, 1))
            culoare_medie_test = np.mean(image2[y:y+100, x:x+100], axis=(0, 1))
            #diff este diferenta in modul a culorilor medii
            diff = np.sum(np.abs(culoare_medie - culoare_medie_test))

            if diff > max_diff:
                max_diff = diff
                pozitie = (i, j)

    matrice_piese[pozitie[1], pozitie[0]] = 1

    #trebuie sa intoarcem ce piesa am gasit adica cifra din intervalul 0 - 90
    #decupeaza patratul de 100x100 din poza_curenta de la pozitia gasita

    x = pozitie[0]*100
    y = pozitie[1]*100

    roi = image2[y:y+100, x:x+100]
    # roi = cv2.resize(roi, (100, 100))

  
    # cifra = pytesseract.image_to_string(roi, config="--psm 6")
    # #extragem doar cifre
    # cifra = ''.join(filter(str.isdigit, cifra))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #aplica GaussianBlur
    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    # _,tresh = cv2.threshold(roi,50,255,cv2.THRESH_BINARY)
    _,tresh = cv2.threshold(roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    roi_mega = np.full(shape=(300, 300), dtype=np.uint8, fill_value=255)
    roi_mega[roi.shape[0]: 200, roi.shape[1]: 200] = tresh
    #afiseaza roi_mega
    # cv2.imshow("roi_mega", roi_mega)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # roi = cv2.resize(roi, (100, 100))
    # roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

    # #afiseaza roi
    # cv2.imshow("roi", roi_mega)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    maxi=-np.inf
    poz=-1
    dict_cifre_similaritati = {}
    dict_numere_similaritati = {}
    for i in lista_templates:
        template = cv2.imread(f"./templates/{i}.jpg")
        #template = cv2.resize(template, (100, 100))
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.GaussianBlur(template, (5, 5), 0)
        _,template = cv2.threshold(template,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        result = cv2.matchTemplate(roi_mega, template, cv2.TM_CCOEFF_NORMED)
        corr=np.max(result)
        if i < 10:
            dict_cifre_similaritati[i] = corr
        else:
            dict_numere_similaritati[i] = corr
        #print(corr)
        #afiseaza template
        # cv2.imshow("template", template)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if corr > maxi:
            maxi = corr
            poz = i

    #daca poz se afla in lista_similaritati_numere atunci intorc pozitie[0], pozitie[1], poz
    #ordoneaza dictionarul descrescator dupa valori
    dict_cifre_similaritati = dict(sorted(dict_cifre_similaritati.items(), key=lambda item: item[1], reverse=True))
    dict_numere_similaritati = dict(sorted(dict_numere_similaritati.items(), key=lambda item: item[1], reverse=True))

    if poz in dict_numere_similaritati:
        return (pozitie[1], pozitie[0], poz)
    else:
        #daca exista 2 cifre cu similaritate mare atunci verific daca nu cumva numarul format din ele 
        #are si el similiaritate mare atunci poz este defapt numarul altfel este cifra cu similaritatea cea mai mare
        values_cifre = list(dict_cifre_similaritati.values())
        first_value = values_cifre[0]
        second_value = values_cifre[1]

        keys_cifre = list(dict_cifre_similaritati.keys())
        first_key = keys_cifre[0]
        second_key = keys_cifre[1]

        values_numere = list(dict_numere_similaritati.values())
        first_value_numere = values_numere[0]

        keys_numere = list(dict_numere_similaritati.keys())
        first_key_numere = keys_numere[0]

        if first_value - first_value_numere < 0.15:
            return (pozitie[1], pozitie[0], first_key_numere)
        
        return (pozitie[1], pozitie[0], first_key)



def gaseste_pozitii_piese(folder_poze, numar_joc):
    #poza_tabla_goala = scoate_imagine_tabla("./imagini_auxiliare/01.jpg")
    ultima_poza = "./imagini_auxiliare/01.jpg"
    #parcurge toate pozele din folderul de poze
    lista_poze = [
        f for f in os.listdir(folder_poze) 
        if os.path.isfile(os.path.join(folder_poze, f)) 
        and f.lower().endswith(".jpg") 
        and f.startswith(f"{numar_joc}_")  # Numele începe cu numărul jocului
    ]
    
    # print(lista_poze)

    # Sortare fișiere care respectă formatul
    sorted_files = sorted(
        [f for f in lista_poze if re.search(r'_(\d+)', f)],
        key=lambda x: int(re.search(r'_(\d+)', x).group(1))
    )
    lista_poze = sorted_files
    # Rezultatul
    #print(sorted_files)

    for poza in lista_poze:
        poza_curenta = f"./{folder_poze}/{poza}"
        pozitie_gasita = gaseste_pozitie_ultima_piesa(ultima_poza, poza_curenta)
        ultima_poza = f"./{folder_poze}/{poza}"
        #extrage din nume poza totul pana la punct
        nume_piesa = poza.split(".")[0]
        #creaza un fisier cu nume piesa .txt
        with open(f"./imagini_txt/{nume_piesa}.txt", "w") as f:
            #in fisier scriu pozitie_gasita[0] iar pozitie_gasita[1] o sa fie sub forma de litera
            f.write(f"{pozitie_gasita[0] + 1}{chr(pozitie_gasita[1] + 65)} {pozitie_gasita[2]}")
        #inchide 
        f.close()

    #print(matrice_piese)



tabla_gol = [
    [30, -1, -1, -1, -1, -1, 30, 30, -1, -1, -1, -1, -1, 30],
    [-1, 20, -1, -1, 14, -1, -1, -1, -1, 14, -1, -1, 20, -1],
    [-1, -1, 20, -1, -1, 12, -1, -1, 12, -1, -1, 20, -1, -1],
    [-1, -1, -1, 20, -1, -1, 11, 13, -1, -1, 20, -1, -1, -1],
    [-1, 14, -1, -1, 20, -1, 13, 11, -1, 20, -1, -1, 14, -1],
    [-1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, 12, -1, -1],
    [30, -1, -1, 13, 11, -1, -1, -1, -1, 13, 11, -1, -1, 30],
    [30, -1, -1, 11, 13, -1, -1, -1, -1, 11, 13, -1, -1, 30],
    [-1, -1, 12, -1, -1, -1, -1, -1, -1, -1, -1, 12, -1, -1],
    [-1, 14, -1, -1, 20, -1, 11, 13, -1, 20, -1, -1, 14, -1],
    [-1, -1, -1, 20, -1, -1, 13, 11, -1, -1, 20, -1, -1, -1],
    [-1, -1, 20, -1, -1, 12, -1, -1, 12, -1, -1, 20, -1, -1],
    [-1, 20, -1, -1, 14, -1, -1, -1, -1, 14, -1, -1, 20, -1],
    [30, -1, -1, -1, -1, -1, 30, 30, -1, -1, -1, -1, -1, 30],
]

def citeste_miscari(fisier_miscari):
    """Citește fișierul de mișcări și returnează o listă cu intervalele de mutări."""
    miscari = []
    ultima_miscare = 0
    with open(fisier_miscari, 'r') as f:
        randuri = f.readlines()
    
    for i, linie in enumerate(randuri):
        jucator, start = linie.split()
        start = int(start)
        if i + 1 < len(randuri):  # Dacă există un rând următor
            _, next_start = randuri[i + 1].split()
            next_start = int(next_start)
            nr_mutari = next_start - start
        else:  # Ultimul rând
            nr_mutari = 50 - start + 1  # Totalul de mutări finale
            ultima_miscare = 1
        miscari.append((jucator, start, nr_mutari,ultima_miscare))
    return miscari

def citeste_piesa(fisier_piesa):
    """Citește un fișier cu detaliile unei piese și returnează poziția și valoarea."""
    with open(fisier_piesa, 'r') as f:
        linie = f.readline().strip()
        linie_joc, valoare = linie.split()
        linie_joc = linie_joc.upper()
        linia = int(linie_joc[:-1])-1  # Extragem linia ca număr
        coloana = ord(linie_joc[-1]) - ord('A')  # Convertim coloana în număr
        valoare = int(valoare)
        return linia, coloana, valoare
    

def verifica_bonus(tabla, linie, coloana, valoare):
    """Verifică dacă valoarea piesei poate fi obținută prin operații matematice cu piesele vecine."""
    # Obține valorile pieselor vecine (sus, jos, stânga, dreapta)
    vecini = []
    if linie > 1:  # Piesa de sus
        vecini.append(tabla[linie-1][coloana])
        vecini.append(tabla[linie-2][coloana])
    if linie < 12:  # Piesa de jos
        vecini.append(tabla[linie+1][coloana])
        vecini.append(tabla[linie+2][coloana])
    if coloana > 1:  # Piesa din stânga
        vecini.append(tabla[linie][coloana-1])
        vecini.append(tabla[linie][coloana-2])
    if coloana < 12:  # Piesa din dreapta
        vecini.append(tabla[linie][coloana+1])
        vecini.append(tabla[linie][coloana+2])
    
    # Verifică dacă valoarea piesei poate fi obținută printr-o operație între vecini
    operatii_adev = 0
    
    index = 0
    while index < len(vecini):
        if tabla_gol[linie][coloana] == 11:
            if vecini[index] !=-1 and vecini[index+1] != -1 and vecini[index] + vecini[index+1] == valoare:
                operatii_adev+=1
        elif tabla_gol[linie][coloana] == 13:  
            if vecini[index] !=-1 and vecini[index+1] != -1 and vecini[index] * vecini[index+1] == valoare:
                operatii_adev+=1
        elif tabla_gol[linie][coloana] == 12:
            if vecini[index] !=-1 and vecini[index+1] != -1 and abs(vecini[index] - vecini[index+1]) == valoare:
                operatii_adev+=1
        elif tabla_gol[linie][coloana] == 14:
            if vecini[index] >= vecini[index+1]:
                if vecini[index] !=-1 and vecini[index+1] != -1 and vecini[index+1] !=0 and vecini[index] / vecini[index+1]   == valoare:
                    operatii_adev+=1
            elif vecini[index] !=-1 and vecini[index+1] != -1 and vecini[index] !=0 and vecini[index+1] / vecini[index]  == valoare:
                    operatii_adev+=1
        else:
            if vecini[index] !=-1 and vecini[index+1] != -1 and vecini[index] + vecini[index+1] == valoare:
                operatii_adev+=1
            elif vecini[index] !=-1 and vecini[index+1] != -1 and vecini[index] * vecini[index+1] == valoare:
                operatii_adev+=1
            elif vecini[index] !=-1 and vecini[index+1] != -1 and abs(vecini[index] - vecini[index+1]) == valoare:
                operatii_adev+=1
            elif vecini[index] >= vecini[index+1]:
                if vecini[index] !=-1 and vecini[index+1] != -1 and vecini[index+1] !=0 and vecini[index] / vecini[index+1]   == valoare:
                    operatii_adev+=1
                elif vecini[index] !=-1 and vecini[index+1] != -1 and vecini[index] !=0 and vecini[index+1] / vecini[index]  == valoare:
                    operatii_adev+=1 
        index+=2
    return operatii_adev

# Apelarea funcției `verifica_bonus` în cadrul logicii care gestionează tabla


def verifica_constrangeri(tabla, tabla_gol, linia, coloana, valoare):
    
    # Convertim la index de matrice (0-based)
    linia_idx = linia 
    coloana_idx = coloana 

    # Calcul bonus pentru poziții speciale din tabla_gol
    bonus = 1
    if tabla_gol[linia_idx][coloana_idx] == 30:
        bonus = 3
    elif tabla_gol[linia_idx][coloana_idx] == 20:
        bonus = 2

    # Adăugăm valoarea piesei pentru fiecare ecuație îndeplinită
    scor = 0
    tabla[linia_idx][coloana_idx] = valoare  # Temporar adăugăm piesa pe tablă pentru verificări
    operatii_adev = verifica_bonus(tabla, linia, coloana, valoare)
    # print(f"kjbfjhsvfhgs {operatii_adev}")
    # #print(tabla)
    # print(linia,coloana)
    # for x in tabla:
    #     print(x)

    if operatii_adev > 0:
        # Înmulțește valoarea piesei cu numărul de operații adevărate
        valoare *= operatii_adev


    # Adăugăm scorul cu bonus
    scor = scor + valoare * bonus
    #tabla[linia_idx][coloana_idx] = -1  # Revenim la starea inițială a tablei
    return scor


tabla = [[-1] * 14 for _ in range(14)]  # Tabla inițială
tabla[6][6] = 1
tabla[6][7] = 2
tabla[7][6] = 3
tabla[7][7] = 4
def proceseaza_miscari(miscari, folder_imagini, tabla_gol,numar_joc):
    """Parcurge mișcările, procesează piesele și verifică regulile."""
    #tabla = [[0] * 14 for _ in range(14)]  # Tabla inițială
    gaseste_pozitii_piese("./antrenare", numar_joc)
    scor_player1 = 0
    scor_player2 = 0

    for jucator, start, nr_miscari,ultima_miscare in miscari:
        # print(f"{jucator} face {nr_miscari} mutări, începând de la {start}.")
        
        #scor_mutare = 0
        if jucator == 'Player1':
            scor_player1 = 0  # Resetăm scorul player1
        else:
            scor_player2 = 0  # Resetăm scorul player2
        
        for mutare_index in range(start, start + nr_miscari):
            fisier = f"{folder_imagini}/{numar_joc}_{mutare_index:02d}.txt"
            linia, coloana, valoare = citeste_piesa(fisier)

            # Afișează mutarea curentă:
            # print(f"Mutarea {mutare_index}: ({linia}, {coloana}) = {valoare}")
            
            # Verifică validitatea mutării și calculează scorul
            scor_mutare = verifica_constrangeri(tabla, tabla_gol, linia, coloana, valoare)
            
            # Actualizează scorul jucătorului
            if jucator == 'Player1':
                scor_player1 += scor_mutare
                # print(f"Scor player1 după mutarea {mutare_index}: {scor_player1}")
                #scor_mutare = 0
            else:
                scor_player2 += scor_mutare
                #print(f"Scor player2 după mutarea {mutare_index}: {scor_player2}")
                #scor_mutare = 0

            # Actualizează tabla:
            tabla[linia ][coloana ] = valoare
        with open(f"./imagini_txt/{numar_joc}_scores.txt","a") as fisier:
            if ultima_miscare == 1:
                if jucator == "Player1":
                    fisier.write(f"{jucator} {start} {scor_player1}")
                elif jucator == "Player2":
                    fisier.write(f"{jucator} {start} {scor_player2}")
            else:
                if jucator == "Player1":
                    fisier.write(f"{jucator} {start} {scor_player1}\n")
                elif jucator == "Player2":
                    fisier.write(f"{jucator} {start} {scor_player2}\n")
        fisier.close()
            #scor_mutare = 0
            

    #print(f"Scor final player1: {scor_player1}")
    #print(f"Scor final player2: {scor_player2}")

# Exemplu de utilizare:

fisier_miscari = f"./antrenare/{numar_joc}_turns.txt"
with open(fisier_miscari, "r") as fisier_sursa, open(f"./imagini_txt/{numar_joc}_turns.txt", "w") as fisier_destinatie:
    continut = fisier_sursa.read()  # Citește tot conținutul fișierului sursă
    fisier_destinatie.write(continut)
fisier_sursa.close()
fisier_destinatie.close()
folder_imagini = "./imagini_txt"
miscari = citeste_miscari(fisier_miscari)
# proceseaza_miscari(miscari, folder_imagini)

proceseaza_miscari(miscari, folder_imagini, tabla_gol,numar_joc)

#gaseste_pozitii_piese("imagini_antrenare", 4)

#pozitie = gaseste_pozitie_ultima_piesa("imagini_antrenare/3_40.jpg", "imagini_antrenare/3_41.jpg")

#print(tab
# for x in tabla:
#     print(x)
