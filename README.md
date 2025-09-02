# Rozpoznawanie marek samochodów (YOLOv8 + MobileNetV2)

Aplikacja desktopowa PyQt5 do rozpoznawania marek pojazdów. YOLOv8 wykrywa pojazd i wycina kadr, MobileNetV2 klasyfikuje markę (wagi w `runs/*/final.pth`), Grad‑CAM wyjaśnia decyzję, a wyniki są zapisywane w SQLite z możliwością eksportu do PDF.

## Najważniejsze funkcje
- Pełny pipeline: YOLOv8 → wycinek → MobileNetV2 → nakładka Grad‑CAM
- Twarde bramkowanie pojazdu: zdjęcia bez pojazdu są odrzucane (brak zapisu do historii/DB)
- Trwałość danych lokalnie: SQLite (`car_history.db`) + kopie obrazów w ukrytym katalogu `.db_images/`
- Historia: podgląd, statystyki i eksport PDF (pojedynczy / zaznaczone / wszystkie)
- Testy: pytest dla bazy/PDF oraz logiki bramkowania

## Wymagania i instalacja
- Python 3.10+ (CUDA opcjonalnie)

```bash
python -m venv .venv
source .venv/bin/activate
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
1) Kliknij „Wybierz obraz” i wskaż zdjęcie.
2) Jeśli wykryto pojazd, zobaczysz przewidywaną markę, pewność i możesz włączyć heatmapę.
3) Rekord jest domyślnie zapisywany do bazy (correct=True). W „Historii” obejrzysz i wyeksportujesz PDF.
4) Jeśli nie wykryto pojazdu, pojawi się komunikat i nic nie zostanie zapisane.

## Struktura projektu
- `GUI.py` — główne okno i logika UI
- `predict.py` — detekcja, wybór wycinka, Grad‑CAM i klasyfikacja
- `database.py` — operacje SQLite i eksport PDF
- `history_viewer.py` — przegląd historii i akcje eksportu
- `runs/…/final.pth`, `label_map.json` — wagi i etykiety klasyfikatora
- `tools/` — skrypty pomocnicze
- `tests/` — testy pytest (CRUD DB + eksport PDF + bramkowanie)

## Architektura (wysoki poziom)
- YOLOv8 (Ultralytics) wykrywa obiekty; kandydaci są filtrowani po etykietach „pojazdowych” i progu ufności.
- Wybrany wycinek (największy spełniający warunki; fallback: największy box) trafia do MobileNetV2.
- Grad‑CAM z ostatniego bloku cech tworzy mapę uwagi, nakładaną na wycinek dla wyjaśnialności.
- Wyniki trafiają do SQLite, a obraz jest kopiowany do `.db_images/` dla stabilnych eksportów.

## Testy
Uruchom testy lokalnie:
```bash
pytest -q
```

Uwaga: jeśli uruchamiasz testy ze skryptów w `tools/`, dodaj plik `tools/conftest.py`, który dopisze katalog repo do `sys.path` (aby moduły jak `predict` były widoczne).

## Rozwiązywanie problemów
- „ModuleNotFoundError” w skryptach `tools/`: dodaj `tools/conftest.py` z dopisaniem katalogu repo do `sys.path`.
- „No classifier checkpoint found”: umieść `final.pth` i `label_map.json` w `runs/<nazwa>/`.
- Brak YOLO: zapewnij obecność `yolov8s.pt` lub pozwól Ultralytics pobrać wagi przy pierwszym uruchomieniu.
- Ostrzeżenia Qt o QSS: używaj tylko wspieranych właściwości; obecny styl jest oczyszczony.

## Uwagi i licencje
- Artefakty danych (DB i kopie obrazów) są wykluczone z repo przez `.gitignore`.
- Zweryfikuj licencje na dane treningowe i opisz źródła w pracy.
- Warto dodać „Model Card” (`MODEL_CARD.md`) z danymi, metrykami i ograniczeniami.

## Dalsze kroki (opcjonalnie)
- Pre‑commit (ruff/black), pakietowanie PyInstallerem, ikona aplikacji.
- Dodatkowe testy dla przypadków brzegowych (wiele pojazdów, małe kadry, słabe oświetlenie).
