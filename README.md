# Rozpoznawanie marek samochodów (YOLOv8 + MobileNetV2)

Aplikacja desktopowa PyQt5 do rozpoznawania marek pojazdów. YOLOv8 wykrywa pojazd (z priorytetem największego i najbardziej wycentrowanego boxu), wycina kadr, MobileNetV2 klasyfikuje markę (wagi w `runs/*/final.pth`), Grad‑CAM wyjaśnia decyzję, a wyniki lądują w SQLite z historią, statystykami i eksportem.

## Co działa już (MVP)
- Jedno‑plikowa predykcja w GUI (z overlay Grad‑CAM i znakiem wodnym SVG marki).
- Tryb batch: wybór wielu plików, tabela wyników, eksport do CSV.
- Detekcja YOLOv8 → center‑biased crop → MobileNetV2 klasyfikacja.
- Heatmapa Grad‑CAM jako pół‑transparentny overlay z przełącznikiem widoczności.
- Zapis wyników do SQLite + kopie obrazów w `.db_images/` dla stabilnych eksportów.
- Okno historii: filtrowanie, podgląd, statystyki, eksport wybranych/all do PDF.
- Dziennik zdarzeń `history.jsonl` do analityki/feedbacku (correct=False po zgłoszeniu błędu).
- Obsługa popularnych formatów obrazów: `png, jpg, jpeg, bmp, webp`.
- Leniwe ładowanie YOLO, cache klasyfikatora i mapy etykiet.
- Twarde bramkowanie „brak pojazdu” (bez zapisu, jasny komunikat w GUI).

## Najważniejsze funkcje (stan aktualny)
**Pipeline / AI**
- YOLOv8 → wybór kadru (największy box z biasem na środek) → MobileNetV2 → Grad‑CAM overlay (jedno przejście forward/backward).
- Obsługa wielu formatów wejściowych: `png, jpg, jpeg, bmp, webp`.
- Lenliwe ładowanie YOLO (pierwsza predykcja), cache klasyfikatora + mapy etykiet.
- Twarde bramkowanie: brak pojazdu ⇒ brak klasyfikacji, brak zapisu do bazy.

**Interfejs / UX**
- Wycentrowany layout z nakładaną (a nie wypychającą) heatmapą.
- Przełącznik „Pokaż / Schowaj heatmapę” – overlay bez zmiany geometrii.
- Pół‑transparentne, skalowalne SVG logo marki jako watermark (chowa się przy heatmapie).
- Tryb batch (wielokrotne pliki) z tabelą wyników + eksport do CSV.
- Przycisk „Zgłoś błąd” – odwrócenie oznaczenia rekordu (feedback: correct=False + zapis do JSONL i DB).
- Blokada zmiany rozmiaru okna (spójna prezentacja) + automatyczne odświeżanie przy alt‑tab / przenoszeniu.

**Dane / Historia**
- SQLite (`car_history.db`) + kopie obrazów w `.db_images/` gwarantujące stabilny eksport.
- Historia: filtrowanie, podgląd, statystyki, eksport wybranych lub wszystkich do PDF.
- Dziennik JSONL (`history.jsonl`) równoległy do DB (surowe zdarzenia / feedback).

**Testy i narzędzia**
- Pytest: CRUD DB, eksport PDF, bramkowanie (no-vehicle gate).
- Skrypty pomocnicze w `tools/` (augmentacje, testy modelu itp.).

**Stabilność / Techniczne**
- Oczyszczony QSS (usunięty `transition` – brak ostrzeżeń Qt).
- Odporność na zmianę fokusu (heatmapa i logo repozycjonują się po zdarzeniach Move / Activate).
- Minimalizacja ponownych odczytów obrazu – skalowanie w locie.

## Wymagania i instalacja
- Python 3.10+ (CUDA opcjonalnie – jeśli dostępna, model sam przejdzie na GPU)

Linux/macOS (Bash):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Uruchomienie
```bash
python GUI.py
```

Modele:
- YOLOv8: plik `yolov8s.pt` w katalogu głównym repo (lub zmień ścieżkę w `GUI.py`).
- Klasyfikator: umieść `final.pth` i `label_map.json` w `runs/<dowolna_nazwa>/` (aplikacja wczyta najnowsze).

## Jak używać
1. Kliknij „Wybierz obraz(y)”:
	- Jeden plik ⇒ natychmiastowa predykcja.
	- Wiele plików ⇒ tryb batch (tabela + opcja zapisu CSV).
2. Po udanej detekcji zobaczysz: markę, pewność (%), logo (SVG) i możesz przełączyć heatmapę.
3. Rekord automatycznie zapisuje się jako poprawny (correct=True).
4. Jeśli wynik jest błędny – użyj „Zgłoś błąd” (oznaczy w DB i doda wpis do JSONL).
5. Zdjęcie bez pojazdu ⇒ komunikat i brak zapisu.
6. W „Historia” możesz filtrować i eksportować PDF (pojedynczy / wybrane / wszystkie).

## Struktura projektu
- `GUI.py` – interfejs, batch, heatmap overlay, SVG watermark, feedback.
- `predict.py` – YOLO + center‑biased crop, MobileNetV2, Grad‑CAM (1 przejście), kodowanie heatmapy (base64 PNG).
- `database.py` – SQLite CRUD, kopiowanie obrazów, eksport PDF.
- `history_viewer.py` – okno historii + akcje eksportu.
- `runs/<...>/final.pth`, `label_map.json` – checkpoint klasyfikatora + mapowanie labeli.
- `tools/` – dodatkowe narzędzia (augmentacje, split, testy modelu itp.).
- `tests/` – testy (DB / PDF / bramkowanie no‑vehicle).
- `history.jsonl` – strumień zdarzeń (feedback & automatyczne zapisy poprawnych wyników).

## Architektura (skrót)
1. YOLOv8 (Ultralytics) → detekcja; wybór boxu = największy + środek ekranu (score: area * center_weight).
2. Wycięty kadr → transformacje → MobileNetV2 (logity + softmax w pamięci).
3. Hook na ostatni blok cech → Grad‑CAM → kolorowanie → PNG → base64 (GUI overlay).
4. Walidacja: brak pojazdu ⇒ return z flagą `no_vehicle`.
5. Zapis: rekord (brand, confidence, czas, correct) + kopia obrazu.
6. Opcjonalny feedback użytkownika nadpisuje pole `correct`.

## Testy
Uruchom testy lokalnie:
```bash
pytest -q
```

Uwaga: jeśli uruchamiasz testy ze skryptów w `tools/`, dodaj plik `tools/conftest.py`, który dopisze katalog repo do `sys.path` (aby moduły jak `predict` były widoczne).

## Build EXE (jeden plik)
- Zainstaluj zależności w wirtualnym środowisku (`.venv`).
- Zbuduj exe:
	- w VS Code: uruchom task "Build AutoDentifier (onefile)".
	- lub w PowerShell:

```powershell
.\.venv\Scripts\pyinstaller.exe --clean --noconfirm .\car_app.spec
```

Wynik: `dist/AutoDentifier.exe` (samowystarczalny).

Notatki runtime:
- Wbudowane są: `config.yaml`, `yolov8s.pt`, `logos/`, `samples/`, oraz cały katalog `runs/` (jeśli istnieje w repo podczas budowy).
- Pliki użytkownika (baza, logi, kopie obrazów) trafiają do `%LOCALAPPDATA%/AutoDentifier`.
- Aplikacja używa ścieżek kompatybilnych z PyInstaller onefile (`sys._MEIPASS`).

## Rozwiązywanie problemów
| Problem | Rozwiązanie |
|--------|-------------|
| `ModuleNotFoundError` w `tools/` | Dodaj `tools/conftest.py` ustawiający `sys.path` na root repo. |
| `No classifier checkpoint found` | Dodaj `final.pth` + `label_map.json` do `runs/<nazwa>/` (najświeższy katalog zostanie użyty). |
| Brak YOLO lub błąd wag | Upewnij się, że `yolov8s.pt` jest w katalogu głównym albo pozwól na auto‑download. |
| Heatmapa nie pokazuje się | Sprawdź czy `heatmap` jest w zwróconym słowniku (aktywna warstwa Grad‑CAM + pojazd). |
| Logo nie widać | Brak pliku SVG w `logos/` lub marka spoza mapy (`brand_map`). |
| Okno „rozjeżdża się” po alt‑tab | Mechanizm eventów Move/Activate automatycznie repozycjonuje; upewnij się, że nie modyfikowano `_post_window_adjust`. |
| Ostrzeżenia Qt o stylach | Używaj tylko właściwości zawartych w aktualnym QSS (usunięto unsupported `transition`). |

## Uwagi i licencje
- Artefakty danych (DB i kopie obrazów) są wykluczone z repo przez `.gitignore`.
- Zweryfikuj licencje na dane treningowe i opisz źródła w pracy.
- Warto dodać „Model Card” (`MODEL_CARD.md`) z danymi, metrykami i ograniczeniami.

## Dalsze kroki (opcjonalnie)
- Slider intensywności heatmapy.
- Parametryzacja współczynnika „center bias” i marginu przy cropie.
- Model Card / metryki (precision/recall per brand, confusion matrix).
- Pre‑commit (ruff/black), PyInstaller + ikona .ico.
- Dodatkowe testy (wiele pojazdów, ekstremalne proporcje, niska ekspozycja).

---
Aktualizacja README odzwierciedla zmiany w GUI (overlay, batch, feedback), pipeline (center‑biased crop, lazy load) i stabilność layoutu.
