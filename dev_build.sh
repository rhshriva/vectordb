#!/usr/bin/env bash
# dev_build.sh — build and test Quiver components
#
# Usage:
#   ./dev_build.sh                 # build + test everything (no FAISS)
#   ./dev_build.sh --faiss         # build + test with FAISS support (dynamic lib)
#   ./dev_build.sh --faiss-static  # build + test with FAISS built from source (C++ toolchain required)
#   ./dev_build.sh --python        # build Python wheel + smoke-test
#   ./dev_build.sh --release       # release-mode build (no tests)
#   ./dev_build.sh --help          # show this message

set -euo pipefail

# ── colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}${BOLD}[quiver]${RESET} $*"; }
success() { echo -e "${GREEN}${BOLD}[ok]${RESET} $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[warn]${RESET} $*"; }
die()     { echo -e "${RED}${BOLD}[error]${RESET} $*" >&2; exit 1; }

# ── argument parsing ──────────────────────────────────────────────────────────
OPT_FAISS=0        # dynamic libfaiss_c
OPT_FAISS_STATIC=0 # build FAISS from source via faiss/static feature
OPT_PYTHON=0
OPT_RELEASE=0

for arg in "$@"; do
  case "$arg" in
    --faiss)        OPT_FAISS=1        ;;
    --faiss-static) OPT_FAISS_STATIC=1 ;;
    --python)       OPT_PYTHON=1       ;;
    --release)      OPT_RELEASE=1      ;;
    --help|-h)
      sed -n '2,11p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *) die "unknown option: $arg  (run with --help)" ;;
  esac
done

# --faiss-static implies --faiss (same Cargo feature, different linker strategy)
if [[ $OPT_FAISS_STATIC -eq 1 ]]; then
  OPT_FAISS=1
fi

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

# ── FAISS library detection ────────────────────────────────────────────────────
#
# check_faiss_dynamic:
#   Returns 0 (found) and prints the library path, or returns 1 (not found).
#
check_faiss_dynamic() {
  local OS
  OS="$(uname -s)"

  # ── macOS ──────────────────────────────────────────────────────────────────
  if [[ "$OS" == "Darwin" ]]; then
    # 1. Homebrew prefix (works on both Intel and Apple Silicon)
    if command -v brew &>/dev/null; then
      local brew_prefix
      brew_prefix="$(brew --prefix faiss 2>/dev/null)" || true
      if [[ -n "$brew_prefix" && -d "$brew_prefix/lib" ]]; then
        local lib
        lib="$(ls "$brew_prefix/lib/libfaiss_c."*.dylib "$brew_prefix/lib/libfaiss_c.dylib" 2>/dev/null | head -1)"
        if [[ -n "$lib" ]]; then
          echo "$lib"
          return 0
        fi
      fi
    fi

    # 2. Common manual install locations
    for candidate in \
        /usr/local/lib/libfaiss_c.dylib \
        /opt/local/lib/libfaiss_c.dylib \
        /opt/homebrew/lib/libfaiss_c.dylib; do
      if [[ -f "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    done

    return 1  # not found on macOS

  # ── Linux ──────────────────────────────────────────────────────────────────
  else
    # 1. ldconfig cache (covers /usr/lib, /usr/local/lib and any custom paths)
    local ldconfig_line
    ldconfig_line="$(ldconfig -p 2>/dev/null | grep 'libfaiss_c\.so' | head -1)" || true
    if [[ -n "$ldconfig_line" ]]; then
      # Format: "  libfaiss_c.so.X (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libfaiss_c.so.X"
      echo "${ldconfig_line##*=> }"
      return 0
    fi

    # 2. Explicit paths (distro packages / manual builds)
    for candidate in \
        /usr/lib/libfaiss_c.so \
        /usr/local/lib/libfaiss_c.so \
        /usr/lib/x86_64-linux-gnu/libfaiss_c.so \
        /usr/lib/aarch64-linux-gnu/libfaiss_c.so; do
      if [[ -f "$candidate" ]]; then
        echo "$candidate"
        return 0
      fi
    done

    return 1  # not found on Linux
  fi
}

# ── Print platform-specific install instructions ───────────────────────────────
faiss_install_hint() {
  local OS
  OS="$(uname -s)"
  echo ""
  if [[ "$OS" == "Darwin" ]]; then
    warn "  Install FAISS via Homebrew:"
    warn "    brew install faiss"
    warn ""
    warn "  Then re-run:"
    warn "    ./dev_build.sh --faiss"
  else
    warn "  Install FAISS via your package manager, e.g.:"
    warn "    # Debian/Ubuntu:"
    warn "    sudo apt-get install libfaiss-dev"
    warn ""
    warn "    # Fedora/RHEL:"
    warn "    sudo dnf install faiss-devel"
    warn ""
    warn "  Then re-run:"
    warn "    ./dev_build.sh --faiss"
  fi
  warn ""
  warn "  Alternatively, build FAISS from source (needs a C++ toolchain + BLAS):"
  warn "    ./dev_build.sh --faiss-static"
  echo ""
}

if [[ $OPT_FAISS -eq 1 ]]; then
  if [[ $OPT_FAISS_STATIC -eq 1 ]]; then
    # Static build: no pre-installed library needed, but warn about requirements
    warn "FAISS static build requested — FAISS will be compiled from source."
    warn "This requires: a C++17 toolchain (g++ / clang++), CMake ≥ 3.17, and BLAS."
    if ! command -v cmake &>/dev/null; then
      die "cmake not found. Install it first (e.g. 'brew install cmake' or 'apt-get install cmake')."
    fi
    info "cmake found: $(cmake --version | head -1)"
  else
    # Dynamic build: libfaiss_c must be present
    info "Checking for libfaiss_c dynamic library…"
    FAISS_LIB_PATH="$(check_faiss_dynamic)" || true

    if [[ -z "$FAISS_LIB_PATH" ]]; then
      warn "libfaiss_c was NOT found on this system."
      faiss_install_hint
      die "Aborting: cannot build with --faiss without libfaiss_c. See hints above."
    fi

    success "Found libfaiss_c: $FAISS_LIB_PATH"

    # Export the library directory so the Rust build script (and linker) can
    # find it even if it is not in the default search paths (common on macOS).
    FAISS_LIB_DIR="$(dirname "$FAISS_LIB_PATH")"
    export LIBRARY_PATH="${FAISS_LIB_DIR}${LIBRARY_PATH:+:$LIBRARY_PATH}"
    if [[ "$(uname -s)" == "Darwin" ]]; then
      export DYLD_LIBRARY_PATH="${FAISS_LIB_DIR}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    else
      export LD_LIBRARY_PATH="${FAISS_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    info "Library search path updated: $FAISS_LIB_DIR"
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
  if [[ $OPT_FAISS_STATIC -eq 1 ]]; then
    FEATURES+=("faiss" "faiss/static")
    info "Feature flags: faiss, faiss/static (building FAISS from source)"
  else
    FEATURES+=("faiss")
    info "Feature flags: faiss (dynamic libfaiss_c)"
  fi
fi

FEATURE_FLAG=""
if [[ ${#FEATURES[@]} -gt 0 ]]; then
  FEATURE_FLAG="--features $(IFS=,; echo "${FEATURES[*]}")"
fi

# ── build ─────────────────────────────────────────────────────────────────────
info "Building all Rust crates…"
# shellcheck disable=SC2086
cargo build "${CARGO_FLAGS[@]}" $FEATURE_FLAG \
  -p quiver-core \
  -p quiver-server \
  -p quiver-cli
success "Rust build complete"

# ── tests ─────────────────────────────────────────────────────────────────────
if [[ $OPT_PYTHON -eq 0 ]]; then
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
  maturin develop --release -m crates/quiver-python/Cargo.toml
  python3 -c "import quiver; print('quiver module imported OK')"
  success "Python bindings built and smoke-tested"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
success "Done! Binaries are in:"
if [[ $OPT_RELEASE -eq 1 ]]; then
  echo "  target/release/quiver-server"
  echo "  target/release/vdb"
else
  echo "  target/debug/quiver-server"
  echo "  target/debug/vdb"
fi
echo ""
echo "To start the server:"
if [[ $OPT_FAISS -eq 1 ]]; then
  echo "  RUST_LOG=info ./target/debug/quiver-server   # with FAISS support"
else
  echo "  RUST_LOG=info ./target/debug/quiver-server"
fi
