# Aufgabe 2 - Oject Counting

Der Code zu der Aufgabe Object Counting wurde mit hilfe von Phython und OpenCV getestet undzum laufen gebracht.
Daher ist die funktionsfähingkeit auf diese Programme beschränkt.

Das gegebene Ziel, für die Erfüllung, dieses Codes ist es in Bildern verschieden farbiger Kaugummies die Anzahl der jeweiligen Kaugummies jeder Farbe zu zählen.

Dafür werden zunächst die Farben, in ihrem Spektrum des HSV-Farbraums und den zugehörigen Ranges, definiert.

Darauf folgen morphologische Operationen, die zur Einteilung, Erkennung, Verdickung oder Verdünnung der Objekte in der Maske sind.

Wie und in welcher Reihenfolge die Farben gezählt werden wird als nächstes in einem Array fesgelegt. 
In den darauf folgenden For-Schleifen werden die Bilder eingefügt, die Masken erstellt und in den Masken die Farben gezählt.
Dabei wird in der MAske noch drauf geachte dass es sich um Kaugummis handelt in dem noch die Rundheit überprüft wird.

Um nun auch im original Bild zu erkennen um welche Kaugummis es sich handelt, wird ein grünes Kästchen um diese gezeichnet und ein roter Kreis makiert den Mittelpunkt.

Zum Schluss werden die gezählten Kaugummis ausgegeben.