# Neuro-Randki (EEG Love Tester & Brain Friendship Graph)

**Neuro-Randki** to biometryczna aplikacja w stylu "Love Tester" oraz interaktywny "Graf Przyjaźni Mózgów", zaprojektowana dla wydarzeń, lodołamaczy (ice-breakers) i imprez integracyjnych. Dwóch użytkowników (lub pojedynczy gracze w trybie solo) podłączają zestawy słuchawkowe EEG, wykonują zsynchronizowane zadania z bodźcami wizualno-słuchowymi, a podobieństwo ich fal mózgowych jest analizowane za pomocą ekstrakcji cech oraz sieci neuronowych PyTorch (`MasterEEGNet`).

![Neuro-Randki Interface](https://github.com/user-attachments/assets/4b2a3913-f7a3-4f72-bf6d-6387c4f58848)

---

## 🌟 Kluczowe Funkcje

- 💖 **Tryb Pary (Love Tester):** Dwóch graczy rywalizuje/bada poziom dopasowania fal mózgowych. Po zakończeniu sekwencji bodźców generowany jest wynik procentowy, animacja konfetti, miejsce w tabeli liderów par oraz kod QR odsyłający do grafu.
- 🧠 **Tryb Solo i "Bliźniak Mózgowy" (Brain Twin):** Możliwość samodzielnego wykonania badania bez pary. Po zapisaniu odcisku neuronowego aplikacja wyszukuje w bazie najbardziej pasującego uczestnika i wskazuje poziom zgodności.
- 🕸️ **Interaktywny Graf Przyjaźni Mózgów (D3.js):** Wizualizacja 2D w czasie rzeczywistym przedstawiająca powiązania fal mózgowych między wszystkimi uczestnikami wydarzenia.
  - Wyszukiwarka uczestników po pseudonimie.
  - Wykresy fal neuronowych w czasie rzeczywistym i wskaźniki pasm EEG (Delta, Theta, Alpha, Beta, Gamma).
  - Ranking powiązań oraz szczegółowy podgląd pasm i najbliższych "mózgowych sąsiadów".
- 🎧 **Integracja ze Sprzętem EEG BrainAccess:**
  - Skanowanie urządzeń przez Bluetooth i automatyczne wykrywanie czepków.
  - Obsługa wielu urządzeń jednocześnie (osobne czepki dla Gracza 1 i Gracza 2).
  - Automatyczna kalibracja jakości sygnału i dobór 4 najlepszych kanałów.
  - Podgląd sygnału EEG na żywo (*Live EEG Preview*).
- 🤖 **Automatyczny Tryb MOCK:** W przypadku braku fizycznego czepka lub SDK BrainAccess, aplikacja przełącza się automatycznie w tryb symulacji, pozwalając na pełne przetestowanie interfejsu.
- 🔬 **3 Metody Embeddingu Mózgowego:**
  1. `handcrafted`: Ekstrakcja 13 zaawansowanych cech fizycznych na kanał (logarytm mocy pasmowej, parametry Hjortha: mobilność i złożoność, momenty statystyczne, percentyle).
  2. `neural`: Głębokie reprezentacje wygenerowane przez sieć `MasterEEGNet` w PyTorch (wektor 128D z modelu `models/openset_master_best.pth`).
  3. `hybrid`: Połączenie wektorów cech fizycznych i głębokich reprezentacji neuronowych.
- 🛠️ **Panel Administratora (`/admin`):**
  - Zmiana metody embeddingu na żywo oraz konfiguracja kanałów.
  - Masowe przeliczanie (*recompute*) zapisanych surowych sygnałów `.npz` nową metodą.
  - Eksport kompletnych danych uczestników i mocy pasm EEG do pliku CSV (`/admin/export`).
  - Zarządzanie i usuwanie wpisów z bazy.
- 🎨 **Stylowy Interfejs Kiosku:** Wygodny układ dla ekranu eventowego z przyciskiem przełączania motywu (**Tryb Jasny / Tryb Mroczny**), czcionką *Oxanium*, autorskimi ilustracjami SVG oraz dynamicznym podglądem zadań.

---

## 🏗️ Architektura i Stos Technologiczny

- **Backend:** Python 3, Flask
- **Przetwarzanie Sygnałów i ML:** NumPy, SciPy, PyTorch (`MasterEEGNet`)
- **Baza Danych i Przechowywanie:** SQLite (`neuro_randki.db`) z automatyczną migracją + skompresowane pliki surowych sygnałów `.npz` w `recordings/`
- **Frontend:** HTML5, Vanilla CSS3 (z obsługą dynamicznego Dark Mode), JavaScript (ES6+), D3.js v7 (Graf), QRCode.js, Canvas Confetti
- **Sprzęt EEG:** BrainAccess SDK (`brainaccess` Python package) z fallbackiem do symulatora

---

## 🔄 Przepływ Użytkownika

```mermaid
flowchart TD
    A[Strona Główna / Rejestracja] --> B{Wybór Trybu}
    B -->|Tryb Pary| C[Rejestracja Gracza 1 i Gracza 2 + Wybór Czepków]
    B -->|Tryb Solo| D[Rejestracja Uczestnika + Wybór Czepka]
    
    C --> E[Kalibracja Kanałów & Podgląd Live EEG]
    D --> E
    
    E --> F[Ekran Zadania - Multimodalne Bodźce Wizualne i Dźwiękowe]
    F --> G[Zapis Surowego Sygnału EEG .npz + Wyliczenie Embeddingu]
    
    G -->|Dla Pary| H[Wynik Dopasowania % + Konfetti + Tabela Liderów]
    G -->|Dla Solo| I[Znalezienie Bliźniaka Mózgowego Brain Twin]
    
    H --> J[Interaktywny Graf Przyjaźni Mózgów /graph]
    I --> J
```

---

## 🚀 Szybkie Uruchomienie

### 1. Automatyczne uruchomienie (Zalecane)

Skrypt `run.sh` automatycznie sprawdza środowisko Python, instaluje brakujące pakiety (`flask`, `numpy`, `scipy`, `torch` CPU wheel), inicjalizuje bazę danych SQLite, zamyka ewentualne kolidujące procesy na porcie 5000 i uruchamia serwer.

```bash
chmod +x run.sh
./run.sh
```

Aplikacja będzie dostępna pod adresem:
- **Aplikacja:** [http://127.0.0.1:5000](http://127.0.0.1:5000)
- **Graf Przyjaźni:** [http://127.0.0.1:5000/graph](http://127.0.0.1:5000/graph)
- **Panel Admina:** [http://127.0.0.1:5000/admin](http://127.0.0.1:5000/admin)

---

### 2. Ręczna instalacja i uruchomienie

Jeśli wolisz uruchomić aplikację bez skryptu pomocniczego:

```bash
# 1. Zainstaluj wymagane pakiety
pip install -r requirements.txt

# 2. (Opcjonalnie) Ustaw metodę embeddingu
export EMBED_METHOD="handcrafted" # opcje: handcrafted | neural | hybrid
export NEURAL_CHANNELS="0,1,2,3"

# 3. Uruchom serwer Flask
python app.py
```

---

## 🛠️ Panel Admina i Zmienne Konfiguracyjne

### Zmienne Środowiskowe
- `EMBED_METHOD`: Ustawia domyślną algorytmiczną metodę wyliczania podobieństwa (`handcrafted`, `neural`, `hybrid`).
- `NEURAL_CHANNELS`: Kanały przekazywane do modelu PyTorch (np. `0,1,2,3`).

### Dostęp do Panelu Admina (`/admin`)
Panel administratora umożliwia:
- Podgląd zarejestrowanych uczestników i statusu ich wektorów cech.
- Eksport danych w formacie CSV (`/admin/export`) – zawiera identyfikatory, pseudonimy, tryb oraz średnie stosunki pasm fal (delta, theta, alpha, beta, gamma).
- Przełączanie metody embeddingu bez konieczności ponownego uruchamiania serwera.
- Przeliczenie wszystkich historycznych zapisów surowych fal EEG w katalogu `recordings/` za pomocą przycisku **"Przelicz wszystkie"**.

---

## 📁 Struktura Projektu

```text
neuro-randki/
├── app.py              # Główny serwer Flask, ścieżki routingu i REST API
├── eeg.py              # Moduł obsługi sprzętu BrainAccess EEG, Bluetooth, skanowania i MOCK
├── model.py            # Model MasterEEGNet (PyTorch), ekstraktory cech (handcrafted/neural/hybrid)
├── database.py         # Obsługa SQLite, nawiązywanie połączeń i migracje schematu
├── schema.sql          # Schemat tabel bazy danych (user, session, result)
├── run.sh              # Automatyczny skrypt konfigurujący środowisko i uruchamiający aplikację
├── requirements.txt    # Zależności Python
├── models/             # Wagi modeli PyTorch (openset_master_best.pth)
├── recordings/         # Zapisane surowe pliki sygnałów EEG (.npz)
├── static/
│   ├── assets/         # Pliki graficzne (SVG, tła, ikony serc)
│   ├── css/            # Style CSS (style.css z motywem jasnym i ciemnym)
│   └── js/             # Skrypty klienckie JS
└── templates/          # Szablony HTML Jinja2
    ├── base.html       # Główny szablon z nagłówkiem i stopką
    ├── register.html   # Wybór czepków i rejestracja pary
    ├── solo.html       # Rejestracja trybu solo
    ├── waiting.html    # Kalibracja czepka, live EEG preview i odliczanie
    ├── task.html       # Multimodalny ekran zadań z bodźcami
    ├── results.html    # Wyniki pary, konfetti i leaderboard
    ├── solo_results.html # Wyniki solo i prezentacja "Bliźniaka Mózgowego"
    ├── graph.html      # Interaktywny Graf Przyjaźni D3.js
    └── admin.html      # Panel zarządczy admina i eksport danych
```

---

## 📜 Licencja & Prawa Autorskie

&copy; 2026 Technologia Neuroguard / KN Neuron. Projekt stworzony dla wydarzeń i prezentacji biometrycznych.

