# Neuro-Randki (EEG Love Tester)

## Przegląd
Internetowa aplikacja biometryczna "Love Tester" zbudowana na potrzeby wydarzeń, lodołamaczy (ice-breakers) i integracji. Dwóch użytkowników łączy się za pomocą zestawów EEG, wykonuje zsynchronizowane zadania, a podobieństwo ich fal mózgowych jest oceniane za pomocą modeli "neuroguard", aby wygenerować wynik dopasowania.

## Architektura i Stos Technologiczny
- **Backend:** Python z Flask
- **Baza danych:** SQLite (lokalna, oparta na plikach, idealna dla kiosków eventowych)
- **Frontend:** Interfejs internetowy zaprojektowany dla pojedynczego, wspólnego ekranu kiosku
- **Biometria:** Obecnie symulowana (TODO: integracja z rzeczywistą aplikacją EEG w Pythonie i modelami neuroguard)

## Przepływ Użytkownika
1. **Rejestracja:** Obaj użytkownicy wprowadzają swoje dane (Pseudonim, Wiek, Płeć - opcjonalnie) do formularza na wspólnym ekranie.
2. **Synchronizacja i Zadanie:** Na ekranie wyświetlane jest zadanie (np. oglądanie krótkiego wideo, patrzenie na określone kolory/obrazki lub czytanie tekstu).
3. **Zbieranie Danych:** Podczas wykonywania zadania (maksymalnie 5 minut, zazwyczaj krócej), aplikacja rejestruje dane EEG obu użytkowników jednocześnie (obecnie symulowane).
4. **Analiza:** Zebrane dane są przekazywane do modelu podobieństwa neuroguard (obecnie symulowane).
5. **Wyniki:** Wyświetlany jest zabawny, magiczny ekran wyników w stylu "Love Tester", pokazujący procentową zgodność ich fal mózgowych.
6. **Przechowywanie:** Wszystkie dane profili użytkowników i wyniki testów są zapisywane w bazie danych SQLite do późniejszego wglądu.

## Strategia Synchronizacji Zadań
Ponieważ aplikacja jest zaprojektowana dla jednego wspólnego ekranu kiosku, synchronizacja jest prosta. Obaj użytkownicy obserwują ten sam monitor.
- Frontend będzie zarządzał cyklem życia zadania (np. 3-sekundowe odliczanie, a następnie wyświetlenie bodźca wizualnego).
- Frontend wyśle żądanie API (lub zdarzenie WebSocket) do backendu Flask, aby rozpocząć symulowane zbieranie danych dokładnie w momencie pojawienia się bodźca i kolejne żądanie, aby je zatrzymać po zakończeniu zadania.

## Fazy Rozwoju
### Faza 1: Fundamenty i Makieta UI
- Konfiguracja aplikacji Flask i schematu bazy danych SQLite (Użytkownicy, Sesje, Wyniki).
- Budowa widoków frontendu: Formularz Rejestracji, Ekran Oczekiwania/Odliczania, Ekran Zadania i Ekran Wyników.

### Faza 2: Logika Rdzenna i Symulacja
- Implementacja maszyny stanów dla przepływu aplikacji.
- Budowa generatora symulowanych danych EEG w celu symulacji przychodzących danych z zestawu słuchawkowego.
- Budowa funkcji symulowanego modelu Neuroguard w celu zwrócenia symulowanego wyniku podobieństwa na podstawie symulowanych danych.

### Faza 3: Integracja Sprzętu i Modelu (Przyszłość)
- Zastąpienie symulowanego generatora EEG rzeczywistą aplikacją Python, która łączy się z zestawami słuchawkowymi EEG.
- Zastąpienie symulowanej funkcji podobieństwa rzeczywistą inferencją Neuroguard w PyTorch/TensorFlow.
