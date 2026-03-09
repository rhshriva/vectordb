#!/usr/bin/env bash
# dev_build.sh — build, test, and optionally run vectordb components
#
# Usage:
#   ./dev_build.sh                  # build + test everything (no FAISS)
#   ./dev_build.sh --faiss          # build + test with FAISS support
#   ./dev_build.sh --server         # build + start the HTTP server
#   ./dev_build.sh --server --faiss # build + start server with FAISS
#   ./dev_build.sh --python         # build Python wheel + smoke-test
#   ./dev_build.sh --release        # release-mode build (no tests)
#   ./dev_build.sh --help           # show this message

set -euo pipefail

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[vectordb]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[ok]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[warn]${RESET} $*"; }
die()     { echo -e "${RED}${BOLD}[error]${RESET} $*" >&2; exit 1; }

# ── argument parsing ──────────────────────────────────────────────────────────
OPT_FAISS=0
OPT_SERVER=0
OPT_PYTHON=0
OPT_RELEASE=0

for arg in "$@"; do
  case "$arg" in
    --faiss)   OPT_FAISS=1   ;;
    --server)  OPT_SERVER=1  ;;
    --python)  OPT_PYTHON=1  ;;
    --release) OPT_RELEASE=1 ;;
    --help|-h)
      sed -n '2,14p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) die "unknown option: $arg  (run with --help)" ;;
  esac
done

# ── prerequisites check ───────────────────────────────────────────────────────
info "Checking prerequisites…"

command -v cargo  &>/dev/null || die "Rust/Cargo not found. Install via: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
command -v rustc  &>/dev/null || die "rustc not found"

RUST_VERSION=$(rustc --version)
info "Using $RUST_VERSION"

if [[ $OPT_PYTHON -eq 1 ]]; then
  command -v python3  &>/dev/null || die "python3 not found"
  command -v maturin &>/dev/null  || die "maturin not found. Install via: pip install maturin"
  PYTHON_VERSION=$(python3 --version)
  info "Using $PYTHON_VERSION"
fi

if [[ $OPT_FAISS -eq 1 ]]; then
  warn "FAISS build requested. Checking for libfaiss_c…"
  if ! ldconfig -p 2>/dev/null | grep -q libfaiss_c && \
     ! ls /usr/local/lib/libfaiss_c* /usr/lib/libfaiss_c* 2>/dev/null | grep -q .; then
    warn "libfaiss_c not found in standard paths."
    warn "To build FAISS from source, add 'static' to features: edit Cargo.toml"
    warn "Continuing — cargo will error if the library truly cannot be found."
  fi
fi

# ── build configuration ───────────────────────────────────────────────────────
CARGO_FLAGS=()
FEATURES=()

if [[ $OPT_RELEASE -eq 1 ]]; then
  CARGO_FLAGS+=(--release)
  info "Build mode: release"
else
  info "Build mode: debug (use --release for an optimised binary)"
fi

if [[ $OPT_FAISS -eq 1 ]]; then
  FEATURES+=("faiss")
  info "Feature flags: faiss"
fi

FEATURE_FLAG=""
if [[ ${#FEATURES[@]} -gt 0 ]]; then
  FEATURE_FLAG="--features $(IFS=,; echo "${FEATURES[*]}")"
fi

# ── build ─────────────────────────────────────────────────────────────────────
info "Building all Rust crates…"
# shellcheck disable=SC2086
cargo build "${CARGO_FLAGS[@]}" $FEATURE_FLAG \
  -p vectordb-core \
  -p vectordb-server \
  -p vectordb-cli
success "Rust build complete"

# ── tests ─────────────────────────────────────────────────────────────────────
if [[ $OPT_SERVER -eq 0 && $OPT_PYTHON -eq 0 ]]; then
  info "Running test suite…"
  # shellcheck disable=SC2086
  cargo test $FEATURE_FLAG
  success "All tests passed"
fi

# ── Python bindings ───────────────────────────────────────────────────────────
if [[ $OPT_PYTHON -eq 1 ]]; then
  info "Building Python extension (maturin develop)…"
  # Ensure we're in a venv
  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    warn "No active Python venv detected."
    warn "Run: python3 -m venv .venv && source .venv/bin/activate"
    warn "Then re-run this script with --python"
    die "Aborting Python build: activate a venv first"
  fi
  maturin develop --release -m crates/vectordb-python/Cargo.toml
  python3 -c "import vectordb; print('vectordb module imported OK')"
  success "Python bindings built and smoke-tested"
fi

# ── server ────────────────────────────────────────────────────────────────────
if [[ $OPT_SERVER -eq 1 ]]; then
  BIN_PATH="target/debug/vectordb-server"
  [[ $OPT_RELEASE -eq 1 ]] && BIN_PATH="target/release/vectordb-server"

  info "Starting vectordb-server…"
  info "  Data directory : ${VECTORDB_DATA_DIR:-./data}"
  info "  Port           : 8080"
  info "  Auth           : ${VECTORDB_API_KEY:+enabled (key set)}${VECTORDB_API_KEY:-disabled (dev mode)}"
  [[ $OPT_FAISS -eq 1 ]] && info "  FAISS          : enabled"
  info "Press Ctrl-C to stop."
  echo ""
  RUST_LOG="${RUST_LOG:-vectordb_server=info,tower_http=debug}" \
    exec "$BIN_PATH"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
success "Done! Binaries are in:"
if [[ $OPT_RELEASE -eq 1 ]]; then
  echo "  target/release/vectordb-server"
  echo "  target/release/vdb"
else
  echo "  target/debug/vectordb-server"
  echo "  target/debug/vdb"
fi
echo ""
echo "Quick-start:"
echo "  ./dev_build.sh --server          # start the HTTP server (port 8080)"
echo "  ./dev_build.sh --python          # build Python wheel"
echo "  ./dev_build.sh --faiss --server  # server with FAISS support"
