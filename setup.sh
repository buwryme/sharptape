#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════════════
# Sharptape Setup Script
# Installs models, NCNN Vulkan tools, and sets up desktop integration
# ═══════════════════════════════════════════════════════════════════════════════

# ── Colors ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Paths & Setup ────────────────────────────────────────────────────────────────

models_dir="$HOME/.local/share/sharptape/models"
bin_dir="$HOME/.local/bin"
install_dir="$HOME/.local/share/sharptape/installation"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
tmp_dir=$(mktemp -d)
ncnn_build_dir="$tmp_dir/ncnn-build"

trap 'rm -rf "$tmp_dir"' EXIT

echo ""
info "Creating directory structure..."
mkdir -p "$models_dir"/{basicvsr,real-cugan,realesrgan}
mkdir -p "$bin_dir"
mkdir -p "$install_dir"
mkdir -p "$HOME"/.local/{share/{sharptape,applications,pixmaps,icons/hicolor/{scalable,48x48}/apps}}
success "Directories created"

echo ""
info "Installing Python dependencies..."
pip install -r "${script_dir}/requirements.txt"
success "Python dependencies installed"

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: Model Downloads
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
info "═══════════════════════════════════════"
info "PART 1: Downloading AI Models"
info "═══════════════════════════════════════"

# 1. basicvsr++
basicvsr_file="$models_dir/basicvsr/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
if [[ -f "$basicvsr_file" ]]; then
  warn "BasicVSR++ model already present, skipping."
else
  info "Downloading BasicVSR++ model..."
  curl -sSLA "Mozilla/5.0" -L -o "$basicvsr_file" \
    "https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"
  success "BasicVSR++ model downloaded"
fi

# 2. real-cugan models
if [[ -d "$models_dir/real-cugan/models-se" || -d "$models_dir/real-cugan/models-pro" ]]; then
  warn "Real-CUGAN models already present, skipping."
else
  info "Fetching Real-CUGAN release..."
  cugan_url=$(curl -s https://api.github.com/repos/nihui/realcugan-ncnn-vulkan/releases/latest | grep "browser_download_url" | grep "ubuntu" | cut -d '"' -f 4 | head -n 1 || true)

  if [[ -n "$cugan_url" ]]; then
    info "Downloading Real-CUGAN models..."
    curl -sSLA "Mozilla/5.0" -L -o "$tmp_dir/realcugan.zip" "$cugan_url"
    unzip -q "$tmp_dir/realcugan.zip" -d "$tmp_dir/cugan_extracted"

    find "$tmp_dir/cugan_extracted" -type d \( -name "models-nose" -o -name "models-pro" -o -name "models-se" \) -exec cp -r {} "$models_dir/real-cugan/" \;
    success "Real-CUGAN models downloaded"
  else
    warn "Could not fetch Real-CUGAN URL, skipping."
  fi
fi

# 3. realesrgan models
if compgen -G "$models_dir/realesrgan/*.bin" > /dev/null; then
  warn "Real-ESRGAN models already present, skipping."
else
  info "Downloading Real-ESRGAN models..."
  esrgan_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
  
  if curl -sSLA "Mozilla/5.0" -L -o "$tmp_dir/realesrgan.zip" "$esrgan_url" 2>/dev/null; then
    unzip -q "$tmp_dir/realesrgan.zip" -d "$tmp_dir/esrgan_extracted"
    find "$tmp_dir/esrgan_extracted" -type f \( -name "*.bin" -o -name "*.param" \) -exec cp {} "$models_dir/realesrgan/" \;
    
    # Verify files were actually copied
    if compgen -G "$models_dir/realesrgan/*.bin" > /dev/null; then
      success "Real-ESRGAN models downloaded"
    else
      error "Real-ESRGAN models failed to extract/copy. Check disk space."
    fi
  else
    error "Failed to download Real-ESRGAN models from $esrgan_url"
  fi
fi

success "All models verified"

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: NCNN Vulkan Tools (BOTH realesrgan + realcugan REQUIRED)
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
info "═══════════════════════════════════════"
info "PART 2: NCNN Vulkan Tools"
info "═══════════════════════════════════════"

# ── Method 1: Try downloading pre-built binaries (FASTEST) ────────────────────

download_prebuilt_ncnn() {
  local tool_name="$1"
  local repo_url="$2"
  local output_name="$3"

  info "Trying pre-built binary for ${tool_name}..."

  # Get latest release download URL for Ubuntu
  local download_url
  download_url=$(curl -s "https://api.github.com/repos/${repo_url}/releases/latest" | \
    grep "browser_download_url" | grep -i "ubuntu\|linux.*x86\|linux.*gz\|linux.*zip" | \
    cut -d '"' -f 4 | head -n 1 || true)

  if [[ -z "$download_url" ]]; then
    # Try any Linux binary if no specific Ubuntu one
    download_url=$(curl -s "https://api.github.com/repos/${repo_url}/releases/latest" | \
      grep "browser_download_url" | grep -i "linux" | \
      cut -d '"' -f 4 | head -n 1 || true)
  fi

  if [[ -n "$download_url" ]]; then
    info "Downloading ${tool_name} from release..."

    # Download and extract
    local archive="$tmp_dir/${output_name}.archive"
    curl -sSLA "Mozilla/5.0" -L -o "$archive" "$download_url"

    # Handle different archive formats
    mkdir -p "$tmp_dir/${output_name}_extracted"
    if [[ "$download_url" == *.tar.gz ]] || [[ "$download_url" == *.tgz ]]; then
      tar xzf "$archive" -C "$tmp_dir/${output_name}_extracted"
    elif [[ "$download_url" == *.zip ]]; then
      unzip -q "$archive" -d "$tmp_dir/${output_name}_extracted"
    else
      # Try to figure out format
      if file "$archive" | grep -q "gzip"; then
        tar xzf "$archive" -C "$tmp_dir/${output_name}_extracted"
      elif file "$archive" | grep -q "Zip"; then
        unzip -q "$archive" -d "$tmp_dir/${output_name}_extracted"
      else
        warn "Unknown archive format for ${tool_name}"
        return 1
      fi
    fi

    # Find and copy the binary
    local found_bin
    found_bin=$(find "$tmp_dir/${output_name}_extracted" -type f -name "${output_name}" 2>/dev/null | head -n 1 || true)

    if [[ -n "$found_bin" && -f "$found_bin" ]]; then
      cp "$found_bin" "$bin_dir/"
      chmod +x "$bin_dir/${output_name}"
      success "${tool_name}: installed pre-built binary"
      return 0
    else
      warn "Could not find ${output_name} in pre-built archive"
      return 1
    fi
  else
    warn "No pre-built release found for ${tool_name}"
    return 1
  fi
}

# ── Method 2: Build from source with proper submodule handling ────────────────

build_ncnn_library() {
  info "Building ncnn library from source (this may take a while)..."
  mkdir -p "$ncnn_build_dir"
  cd "$ncnn_build_dir"

  if [[ ! -d "ncnn" ]]; then
    info "Cloning ncnn repository..."
    git clone --recursive https://github.com/Tencent/ncnn.git
  fi

  # Always ensure submodules are initialized (fixes the CMake error!)
  cd ncnn
  info "Updating git submodules..."
  git submodule update --init --recursive 2>/dev/null || true

  mkdir -p build && cd build

  cmake -DCMAKE_BUILD_TYPE=Release \
        -DNCNN_VULKAN=ON \
        -DNCNN_BUILD_EXAMPLES=OFF \
        -DNCNN_BUILD_TOOLS=OFF \
        -DCMAKE_INSTALL_PREFIX="$ncnn_build_dir/ncnn-install" ..

  make -j$(nproc)
  make install/strip
  success "ncnn library built successfully"
}

build_realesrgan_from_source() {
  info "Building realesrgan-ncnn-vulkan from source..."
  cd "$tmp_dir"

  if [[ ! -d "Real-ESRGAN-ncnn-vulkan" ]]; then
    git clone --depth 1 https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan.git
  fi

  cd Real-ESRGAN-ncnn-vulkan

  # Ensure submodules are initialized
  git submodule update --init --recursive 2>/dev/null || true

  rm -rf build
  mkdir -p build && cd build

  cmake -DCMAKE_BUILD_TYPE=Release \
        -Dncnn_DIR="$ncnn_build_dir/ncnn-install/lib/cmake/ncnn" ..

  make -j$(nproc)

  if [[ -f "realesrgan-ncnn-vulkan" ]]; then
    cp realesrgan-ncnn-vulkan "$bin_dir/"
    chmod +x "$bin_dir/realesrgan-ncnn-vulkan"
    success "realesrgan-ncnn-vulkan: built and installed"
    return 0
  else
    error "Build completed but binary not found!"
    return 1
  fi
}

build_realcugan_from_source() {
  info "Building realcugan-ncnn-vulkan from source..."
  cd "$tmp_dir"

  if [[ ! -d "realcugan-ncnn-vulkan" ]]; then
    git clone --depth 1 https://github.com/nihui/realcugan-ncnn-vulkan.git
  fi

  cd realcugan-ncnn-vulkan

  # Ensure submodules are initialized
  git submodule update --init --recursive 2>/dev/null || true

  rm -rf build
  mkdir -p build && cd build

  cmake -DCMAKE_BUILD_TYPE=Release \
        -Dncnn_DIR="$ncnn_build_dir/ncnn-install/lib/cmake/ncnn" ..

  make -j$(nproc)

  if [[ -f "realcugan-ncnn-vulkan" ]]; then
    cp realcugan-ncnn-vulkan "$bin_dir/"
    chmod +x "$bin_dir/realcugan-ncnn-vulkan"
    success "realcugan-ncnn-vulkan: built and installed"
    return 0
  else
    error "Build completed but binary not found!"
    return 1
  fi
}

# ── Main NCNN Installation Logic ─────────────────────────────────────────────

install_ncnn_tool() {
  local tool_name="$1"
  local repo_url="$2"
  local bin_name="$3"
  local build_func="$4"

  # Skip if already installed
  if [[ -x "$bin_dir/$bin_name" ]]; then
    success "${tool_name}: already installed"
    return 0
  fi

  # Method 1: Try pre-built binary first
  if download_prebuilt_ncnn "$tool_name" "$repo_url" "$bin_name"; then
    return 0
  fi

  # Method 2: Build from source
  info "Pre-built binary failed, building ${tool_name} from source..."
  if $build_func; then
    return 0
  fi

  # All methods failed
  error "Failed to install ${tool_name}"
  return 1
}

# Check what needs to be installed
need_realesrgan=false
need_realcugan=false

if [[ ! -x "$bin_dir/realesrgan-ncnn-vulkan" ]]; then
  need_realesrgan=true
fi

if [[ ! -x "$bin_dir/realcugan-ncnn-vulkan" ]]; then
  need_realcugan=true
fi

if [[ "$need_realesrgan" == true ]] || [[ "$need_realcugan" == true ]] || [[ "${1:-}" == "--force-build" ]]; then
  # Build shared ncnn library (required for source builds of both tools)
  if [[ "$need_realesrgan" == true ]] || [[ "$need_realcugan" == true ]]; then
    # Try installing each tool (will try prebuilt first, then source)
    if [[ "$need_realesrgan" == true ]] || [[ "${1:-}" == "--force-build" ]]; then
      install_ncnn_tool \
        "realesrgan-ncnn-vulkan" \
        "xinntao/Real-ESRGAN-ncnn-vulkan" \
        "realesrgan-ncnn-vulkan" \
        build_realesrgan_from_source || true
    fi

    if [[ "$need_realcugan" == true ]] || [[ "${1:-}" == "--force-build" ]]; then
      install_ncnn_tool \
        "realcugan-ncnn-vulkan" \
        "nihui/realcugan-ncnn-vulkan" \
        "realcugan-ncnn-vulkan" \
        build_realcugan_from_source || true
    fi
  fi
else
  success "NCNN Vulkan tools already installed, skipping build."
  info "Use --force-build to rebuild."
fi

# Verify installations (BOTH must be present!)
echo ""
info "Verifying NCNN installations..."

all_ok=true

for bin in realesrgan-ncnn-vulkan realcugan-ncnn-vulkan; do
  if [[ -x "$bin_dir/$bin" ]]; then
    # NCNN Vulkan tools don't support --version, just verify executable works
    file_info=$(file "$bin_dir/$bin" 2>/dev/null | grep -oE 'ELF|executable|script' | head -1 || echo "ok")
    success "$bin: installed ($file_info)"
  else
    error "$bin: NOT FOUND at $bin_dir/"
    all_ok=false
  fi
done

if [[ "$all_ok" == false ]]; then
  error ""
  error "Not all NCNN tools were installed."
  error "Sharptape requires both realesrgan-ncnn-vulkan and realcugan-ncnn-vulkan."
  error "Missing tools will cause upscale to fall back to Lanczos."
  error ""
  info "Make sure build dependencies are installed, then try: ./setup.sh --force-build"
fi

# ═══════════════════════════════════════════════════════════════════
# PART 3: Desktop Integration
# ═══════════════════════════════════════════════════════════════════

echo ""
info "═══════════════════════════════════════"
info "PART 3: Desktop Integration"
info "═══════════════════════════════════════"

# 1. icon installation
icon_src="$script_dir/src/assets/icon.svg"

if [[ -f "$icon_src" ]]; then
  cp "$icon_src" "$HOME/.local/share/pixmaps/net.buwryy.Sharptape.svg"
  cp "$icon_src" "$HOME/.local/share/icons/hicolor/scalable/apps/net.buwryy.Sharptape.svg"
  cp "$icon_src" "$HOME/.local/share/icons/hicolor/48x48/apps/net.buwryy.Sharptape.svg"
  chmod 644 "$HOME/.local/share/icons/hicolor/scalable/apps/net.buwryy.Sharptape.svg"
  success "Icon installed"
fi

# 2. Install application files to installation directory
info "Installing application to $install_dir ..."
cp -r "$script_dir/src/"* "$install_dir/"
success "Application files installed"

# 2b. version file
cp "$script_dir/VERSION" "$HOME/.local/share/sharptape/VERSION"
success "Version file installed"

# 3. launcher script
info "Installing launcher to $bin_dir/sharptape ..."
cat > "$bin_dir/sharptape" << 'LAUNCHER'
#!/usr/bin/env bash
# Ensure ~/.local/bin is in PATH for subprocesses (ffmpeg, etc.)
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "$HOME/.local/share/sharptape/installation"
exec python3 app.py "$@"
LAUNCHER
chmod 755 "$bin_dir/sharptape"
success "Sharptape installed to $bin_dir/sharptape"

# 4. desktop entry
desktop_target="$HOME/.local/share/applications/net.buwryy.Sharptape.desktop"
desktop_src=""

if [[ -f "$script_dir/net.buwryy.Sharptape.desktop" ]]; then
  desktop_src="$script_dir/net.buwryy.Sharptape.desktop"
elif [[ -f "$script_dir/sharptape.desktop" ]]; then
  desktop_src="$script_dir/sharptape.desktop"
fi

if [[ -n "$desktop_src" ]]; then
  cp "$desktop_src" "$desktop_target"
  sed -i "s|\$HOME|$HOME|g; s|~|$HOME|g" "$desktop_target"
  chmod 644 "$desktop_target"

  # cache invalidation & validation
  touch "$HOME/.local/share/icons/hicolor" "$HOME/.local/share/applications"
  update-desktop-database "$HOME/.local/share/applications" &> /dev/null || true
  gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" &> /dev/null || true
  desktop-file-validate "$desktop_target" &> /dev/null || true

  success "Desktop entry installed"
else
  warn "Desktop entry file not found, creating default..."
  cat > "$desktop_target" << EOF
[Desktop Entry]
Name=Sharptape
Comment=Video Quality Enhancement Tool
Exec=$bin_dir/sharptape %F
Icon=net.buwryy.Sharptape
Terminal=false
Type=Application
Categories=AudioVideo;Video;Graphics;
MimeType=video/mp4;video/webm;video/x-matroska;video/quicktime;video/x-msvideo;
StartupNotify=true
EOF
  chmod 644 "$desktop_target"
  success "Default desktop entry created"
fi

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

echo ""
echo "Sharptape setup complete!"
echo "To run:    sharptape"
echo ""

# Ensure ~/.local/bin is in PATH for current session
if [[ ":$PATH:" != *":$bin_dir:"* ]]; then
  warn "NOTE: $bin_dir is not in your PATH!"
  info "Add this line to your ~/.bashrc or ~/.zshrc:"
  echo '  export PATH="$HOME/.local/bin:$PATH"'
fi
