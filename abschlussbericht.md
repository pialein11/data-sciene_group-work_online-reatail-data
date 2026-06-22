# Abschlussbericht – Online Retail Data Science Projekt

## 1. Projektübersicht

Dieses Projekt analysiert den **Online Retail Transaktionsdatensatz** (UCI / Sakar et al., 2019) und wendet verschiedene Machine-Learning-Verfahren an, um Kundenverhalten zu verstehen und vorherzusagen. Die Analyse gliedert sich in vier Schritte:

1. Datenbereinigung und Exploration (Notebook 01)
2. Unsupervised Learning: Kundensegmentierung (Notebook 02)
3. Supervised Classification: High-Value-Customer erkennen (Notebook 03)
4. Supervised Regression: Kundenumsatz vorhersagen (Notebook 04)

---

## 2. Datensatz

Der Datensatz enthält **541.910 Transaktionszeilen** mit 8 Spalten (InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country).

### Datenqualität

| Problem | Anzahl |
|---|---|
| Fehlende CustomerID | 135.081 (25 %) |
| Fehlende Description | 1.892 |
| Stornierungen (InvoiceNo beginnt mit „C") | 9.288 (1,7 %) |
| Negative Mengen / Preise | vorhanden |

### Bereinigungsstrategie

Für die Machine-Learning-Notebooks wurden alle Zeilen mit fehlender CustomerID, negativer Menge oder negativem Preis sowie Stornierungen entfernt. Es verbleiben **397.884 gültige Verkaufszeilen** von **4.338 eindeutigen Kunden** aus **38 Ländern**.

### Wichtige Kennzahlen

| Kennzahl | Wert |
|---|---|
| Gesamtumsatz | 8.911.407,90 € |
| Eindeutige Rechnungen | 18.533 |
| Durchschnittlicher Warenkorb | 480,84 € |

---

## 3. Machine-Learning-Strategie

Da der Datensatz keine vorgefertigte Zielvariable enthält, wurden drei unterschiedliche Ansätze gewählt:

- **Unsupervised**: Kundensegmentierung ohne Zielvariable (K-Means, hierarchisches Clustering)
- **Classification**: Aus dem Umsatz abgeleitete Zielvariable „HighValueCustomer" (Top 25 %)
- **Regression**: Vorhersage des numerischen Gesamtumsatzes pro Kunde

---

## 4. Ergebnisse

### 4.1 Notebook 01 – Datenbereinigung & Exploration

Die Rohdatei verwendet Semikolon-Trennung, Dezimalkomma und UTF-8-BOM-Encoding. Das Einlesen erfordert `sep=";"`, `decimal=","`, `encoding="utf-8-sig"` und `on_bad_lines="skip"` (mindestens eine korrupte Zeile im Rohdatensatz).

Wichtigste Erkenntnisse:
- **United Kingdom** dominiert mit über 80 % des Umsatzes
- Stornierungsrate: 1,7 % der Transaktionen
- Starke Ausreißer beim Umsatz: Maximalwert 280.206 € (einzelner Kunde), Median 674 €

### 4.2 Notebook 02 – Unsupervised Learning (Kundensegmentierung)

**Preprocessing**: Log-Transformation (`np.log1p`) + StandardScaler, da Retail-Umsätze stark rechtschief verteilt sind.

**Elbow-Methode & Silhouette Score:**

| k | Silhouette Score | SSE |
|---|---|---|
| 2 | **0.4084** | 14.269 |
| 3 | 0.3040 | 11.277 |
| 4 | 0.3175 | 9.405 |
| 5 | 0.2752 | 8.082 |

Das Notebook wählt automatisch **k = 2** (höchster Silhouette Score). Das ist mathematisch korrekt.

**⚠️ Wichtige Einschränkung:** k = 2 ist statistisch am trennschärfsten, weil die Daten dominiert werden durch eine Grundstruktur (Gelegenheitskäufer vs. aktive Kunden). Für ein realistischeres Kundensegmentierungsmodell wäre **k = 3 oder k = 4** interpretativ sinnvoller, auch wenn der Score etwas niedriger ist.

**Cluster-Ergebnisse bei k = 2:**

| Cluster | Ø Umsatz | Ø Bestellungen | Ø Produkte | Ø Aktive Tage | Interpretation |
|---|---|---|---|---|---|
| 0 | 4.637 € | 8,6 | 118 | 7,6 | **Stammkunden** |
| 1 | 478 € | 1,7 | 27 | 1,6 | **Gelegenheitskäufer** |

**Clustergrößen:** Cluster 0 = 1.644 Kunden (38 %), Cluster 1 = 2.694 Kunden (62 %)

**Vergleich KMeans vs. hierarchisches Clustering (Agglomerative, Ward):**
Die Zuordnung stimmt bei Cluster 1 zu 98 % überein (2.637 von 2.694). Bei Cluster 0 gibt es stärkere Abweichungen – das hierarchische Verfahren splittert diesen Bereich anders.

**PCA:** Die ersten zwei Hauptkomponenten erklären einen erheblichen Teil der Varianz, reichen aber nicht für vollständige Separation.

### 4.3 Notebook 03 – Supervised Classification

**Zielvariable:** HighValueCustomer = 1, wenn Gesamtumsatz ≥ 75. Perzentil (Schwelle: 1.661,74 €)  
**Klassenverteilung:** 3.253 normale Kunden (75 %), 1.085 High-Value-Kunden (25 %)  
**Data Leakage:** korrekt vermieden – `total_revenue` wird nicht als Feature verwendet

**Modellergebnisse:**

| Modell | Accuracy |
|---|---|
| Decision Tree (max_depth=5) | **96,5 %** |
| K-Nearest Neighbors (k=5) | 96,5 % |
| Naive Bayes | 90,6 % |

**Classification Report (Decision Tree):**

| Klasse | Precision | Recall | F1 |
|---|---|---|---|
| Normal Customer | 0.97 | 0.99 | 0.98 |
| High-Value Customer | 0.96 | 0.90 | 0.93 |

Die Accuracy von 96,5 % ist trotz ausgeglichener Klassen (25/75) sehr hoch. Decision Tree und KNN performen identisch – das deutet darauf hin, dass die Klassengrenze gut trennbar ist, was bei einer quantilbasierten Zielvariable zu erwarten ist.

### 4.4 Notebook 04 – Supervised Regression

**Zielvariable:** Gesamtumsatz pro Kunde (log-transformiert für Training)

**Modellergebnisse:**

| Modell | MAE | RMSE | R² (log) |
|---|---|---|---|
| Linear Regression | 3.701 € | 54.063 € | 0.620 |
| **Decision Tree Regression** | **718 €** | **4.085 €** | **0.897** |
| Polynomial Regression | 77.505.902 € | 2.283.357.168 € | 0.599 |

**⚠️ Kritischer Befund – Polynomial Regression:** Die polynomiale Regression (Grad 2) zeigt massiv überhöhte Fehler (MAE > 77 Mio. €). Das ist ein klassisches **Overfitting**-Problem: Durch die quadratischen Terme und die hohe Anzahl von Features entstehen bei Ausreißern extreme Vorhersagen. Dieses Ergebnis sollte im Bericht explizit diskutiert werden.

**Bestes Modell:** Decision Tree Regression mit R² = 0.897 – erklärt 90 % der Varianz im log-transformierten Umsatz.

---

## 5. Fazit

Das Projekt zeigt einen vollständigen Data-Science-Workflow von Datenbereinigung über Exploration bis zu drei ML-Ansätzen. Die wichtigsten Erkenntnisse:

- Ca. 62 % der Kunden sind Gelegenheitskäufer mit nur 1–2 Bestellungen
- High-Value-Kunden (Top 25 %) können mit 96,5 % Genauigkeit erkannt werden
- Der Kundenumsatz lässt sich mit einem Decision Tree gut vorhersagen (R² ≈ 0.90)
- Die polynomiale Regression ist für diesen Datensatz nicht geeignet (Overfitting)

---

## 6. Einsatz von LLMs

Für dieses Projekt wurden Large Language Models (Claude von Anthropic) unterstützend eingesetzt für:

- **Erklärung von Konzepten** (z. B. Log-Transformation, Silhouette Score, Data Leakage)
- **Code-Review** (Überprüfung auf korrekte sklearn-Verwendung)
- **Formulierung von Kommentaren und Dokumentation**

Alle generierten Code-Abschnitte wurden von allen Gruppenmitgliedern verstanden, überprüft und bei Bedarf angepasst. Die inhaltlichen Entscheidungen (Wahl der Features, ML-Strategie, Interpretation der Ergebnisse) wurden eigenständig getroffen.

