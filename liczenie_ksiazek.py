import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ----------- Wczytanie obrazu półki z książkami
img = cv2.imread("polka.jpg")

# Poprawa jasności i kontrastu przez CLAHE i wyrównanie histogramu

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_poprawiony = clahe.apply(l)
img_po = cv2.cvtColor(cv2.merge((l_poprawiony, a, b)), cv2.COLOR_LAB2BGR)

# ----------- Podgląd histogramu przed i po korekcji jasności

plt.hist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256], alpha=0.5, label="Przed")
plt.hist(cv2.cvtColor(img_po, cv2.COLOR_BGR2GRAY).ravel(), 256, [0, 256], alpha=0.5, label="Po (CLAHE)")
plt.legend()
plt.savefig("histogram.png")

#------------ Detekcja książek (w zbiorze obiektów COCO klasa "book" ma numer 73)

model = YOLO("yolov8n.pt")
wynik = model.predict(source=img_po, classes=[73], conf=0.25, save=True)[0]

#------  Liczenie i zapis obrazu z ramkami ze skryptu

liczba_ksiazek = len(wynik.boxes)
obraz_z_ramkami = wynik.plot()
cv2.putText(obraz_z_ramkami, f"Ksiazki: {liczba_ksiazek}", (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 2)
cv2.imwrite("wynik.jpg", obraz_z_ramkami)

print("Liczba wykrytych książek:", liczba_ksiazek)