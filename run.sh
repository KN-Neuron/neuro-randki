#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# neuro-randki — setup & run
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }

# ── 1. Python ─────────────────────────────────────────────────────────────────
info "Sprawdzam Python..."
PYTHON=$(command -v python3 || command -v python || error "Nie znaleziono Python 3")
PY_VER=$($PYTHON --version 2>&1)
info "Używam: $PYTHON ($PY_VER)"

# ── 2. pip packages ───────────────────────────────────────────────────────────
info "Sprawdzam zależności pip..."

check_pkg() {
  $PYTHON -c "import $1" 2>/dev/null && return 0 || return 1
}

MISSING=()
for pkg in flask numpy scipy; do
  check_pkg "$pkg" || MISSING+=("$pkg")
done

# PyTorch — check separately (CPU wheel)
if ! check_pkg torch; then
  MISSING+=("torch")
fi

if [ ${#MISSING[@]} -gt 0 ]; then
  warn "Brakuje: ${MISSING[*]}"
  info "Instaluję..."
  for pkg in "${MISSING[@]}"; do
    if [ "$pkg" = "torch" ]; then
      $PYTHON -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q \
        || error "Nie mogę zainstalować torch. Zainstaluj ręcznie: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    else
      $PYTHON -m pip install "$pkg" -q || error "Nie mogę zainstalować $pkg"
    fi
  done
  info "Instalacja zakończona."
else
  info "Wszystkie pakiety pip są dostępne."
fi

# ── 3. BrainAccess SDK ────────────────────────────────────────────────────────
if check_pkg brainaccess; then
  info "BrainAccess SDK: OK"
else
  warn "BrainAccess SDK nie jest zainstalowany."
  warn "Apka uruchomi się w trybie MOCK (bez czepka EEG)."
  warn "Żeby zainstalować SDK: postępuj wg dokumentacji BrainAccess."
fi

# ── 4. Model weights ──────────────────────────────────────────────────────────
info "Sprawdzam plik modelu..."
MODEL_PATH="$SCRIPT_DIR/models/openset_master_best.pth"
if [ -f "$MODEL_PATH" ]; then
  SIZE=$(du -h "$MODEL_PATH" | cut -f1)
  info "Model znaleziony: $MODEL_PATH ($SIZE)"
else
  warn "Brak pliku modelu: $MODEL_PATH"
  warn "Apka uruchomi się z losowymi wagami (wyniki nie będą miarodajne)."
fi

# ── 5. Baza danych ────────────────────────────────────────────────────────────
info "Inicjalizuję bazę danych..."
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from app import app
with app.app_context():
    from database import init_db
    init_db()
print('  Baza OK')
" || warn "Błąd inicjalizacji bazy — kontynuuję."

# ── 6. Zabij ewentualny stary proces ─────────────────────────────────────────
OLD_PID=$(lsof -ti :5000 2>/dev/null || true)
if [ -n "$OLD_PID" ]; then
  warn "Port 5000 zajęty (PID $OLD_PID) — zatrzymuję..."
  kill "$OLD_PID" 2>/dev/null || true
  sleep 1
fi

# ── 7. Start ──────────────────────────────────────────────────────────────────
echo ""
# ── Embedding method (override via env or first arg) ─────────────────────────
export EMBED_METHOD="${EMBED_METHOD:-handcrafted}"   # handcrafted | neural | hybrid
export NEURAL_CHANNELS="${NEURAL_CHANNELS:-0,1,2,3}" # which 4 ch for neural mode

echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  🧠  neuro-randki  →  http://127.0.0.1:5000       ${NC}"
echo -e "${GREEN}  🔧  Admin panel   →  http://127.0.0.1:5000/admin?pw=neuro2025${NC}"
echo -e "${GREEN}  📊  Metoda:       ${EMBED_METHOD}                 ${NC}"
echo -e "${GREEN}  Ctrl+C żeby zatrzymać                             ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

exec $PYTHON app.py
