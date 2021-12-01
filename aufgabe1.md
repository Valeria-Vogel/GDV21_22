Assignment #1 - Optical Illusion
Nach einer Aufgabenstellung wurde in unserem Code, mit Python und OpenCV, eine animierte optische Täuschung erstellt, wie genau wird hier erklärt.
Um den Code in Python erst funktionsfähig zu machen, müssen zuallererst einige Werte importiert werden, wie für OpenCV „cv2“, oder „numpy as np“ und Copy.
Im nächsten Schritt werden Variablen deklariert, um mit diesen weiterzuarbeiten z.B. im Nachhinein ein wert verändern sollte.
Nach dem wir unsere Werte festgelegt haben wird nun ein Bild mit Farbverlauf ein „Gradient Image“ für die Illusion generiert. Um dies zu bewerkstelligen, wird erst ein schwarzes Bild erstellt, danach wird in einer Schleife der Farbverlauf erschaffen, der von dunkel nach hell übergeht.
Daraufhin wird ein Kästchen aus der Mitte des Farbverlaufs herauskopiert.
Als vergleich wird das kopierte Bild in den Dunklen und hellen Bereich gesetzt.
Fast zu guter Letzt wird in einer While-Schleife die Bewegung des kopierten Rechtecks festgelegt, auf wie vielen Pixeln er sich um wie viele Pixel bewegt. Noch dazu muss festgelegt werden, dass nach dem Das Rechteck durchgelaufen ist, dass der Farbverlauf auf seien Vorherigen zustand zurückgesetzt wird und kein Schmiereffekt entsteht
Zum Schluss wird alles in einem Vidoebild ausgegeben, dieses Fenster ist auch nach Belieben Verstellbar und anpassbar.
