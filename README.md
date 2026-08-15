# 🎬 Sharptape

<p align="center">
  <img src="src/assets/icon.svg" alt="Sharptape Icon" width="128" height="128" />
  <br />
  <b>Sharptape</b>
</p>

[![Version](https://img.shields.io/badge/version-v1.0.0-blue?style=flat-square)](VERSION)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square)](LICENSE)
[![Platform: Linux](https://img.shields.io/badge/platform-Linux-orange?style=flat-square)](README.md)

AI-powered video quality enhancement tool for Linux. Upscale, denoise, and temporally enhance your videos with OSS AI models.

Built with **GTK 4 / Libadwaita** | Runs on **NVIDIA/AMD GPUs** via Vulkan + CUDA

---

## Features

- **Real-CUGAN & Real-ESRGAN spatial upscaling** — NCNN Vulkan (blazing fast, GPU-native)
- **BasicVSR++ temporal enhancement** — PyTorch/CUDA (smooth motion, better consistency)
- **Smart pre/post processing** — Sharpening, denoising, deblocking, antialiasing
- **Auto hardware detection** — Optimal settings chosen automatically for your GPU
- **Manual control** — Override tile size, batch size, model tier for power users
- **Multi-codec output** — H.264, HEVC (VP9), ProRes with adjustable quality
- **Batch directory mode** — Process whole frame sequences in one pass
- **Precise color handling** — Color space awareness, proper range mapping

---

## Installation

### Quick Start

```bash
# Clone and setup
git clone https://github.com/buwryme/sharptape.git
cd sharptape/sharptape

# Install system deps (see below), then:
./setup.sh

# Run (after setup.sh installs the launcher)
sharptape
```

### System Requirements & Dependencies

Install these dependencies for your distribution before running `setup.sh`:

#### 1. System Packages

* **Debian / Ubuntu:**
  ```bash
  sudo apt update
  sudo apt install python3 python3-pip python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1 ffmpeg libvulkan-dev build-essential cmake git
  ```
* **Fedora:**
  ```bash
  sudo dnf install python3 python3-pip python3-gobject gtk4 libadwaita ffmpeg vulkan-loader-devel cmake gcc gcc-c++ git
  ```
* **Arch Linux:**
  ```bash
  sudo pacman -Syu python python-pip python-gobject gtk4 libadwaita ffmpeg vulkan-headers cmake base-devel git
  ```
* **openSUSE:**
  ```bash
  sudo zypper install python3-gobject gtk4 libadwaita-1-0 ffmpeg vulkan-loader-devel cmake gcc gcc-c++ git
  ```

#### 2. PyTorch (AI Temporal Pass Support)

> [!IMPORTANT]
> **GPU / CUDA Acceleration Warning:** 
> Do **not** install PyTorch (`python3-torch` or `python-pytorch`) from distribution repositories (especially on Fedora or official OSS repos), as they do not package NVIDIA CUDA support.
>
> To use the AI temporal enhancement pass (`BasicVSR++`) with GPU acceleration, let `setup.sh` install them automatically via `pip`, or run:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> ```

### Optional: AMD GPU (Radeon)

Vulkan works on AMD out of the box via Mesa/AMDGPU drivers. Official CUDA is NVIDIA-only. If you have an AMD card and wish to use the `BasicVSR++` temporal pass with GPU acceleration, install PyTorch with ROCm support:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

> [!NOTE]
> PyTorch with ROCm support is implemented but remains untested. Alternatively, configure the application to run the spatial upscalers (Real-CUGAN / Real-ESRGAN) on Vulkan with FFmpeg temporal processing.

---

## What Gets Installed

`setup.sh` downloads and builds:

- **realcugan-ncnn-vulkan** — Fast anime-style upscaler
- **realesrgan-ncnn-vulkan** — Photo-realistic upscaler
- **PyTorch** — Deep learning (via pip + requirements.txt)

Models are auto-fetched from upstream repositories.

---

## Usage

1. **Open** Sharptape
2. **Select** your input video
3. **Adjust** scale % and filter settings (or leave on auto)
4. **Choose** upscaler (Anime or Realistic) and optional temporal enhancement
5. **Pick** output codec and quality
6. **Hit** Enhance!

### Tips

- **Auto mode** = best for most people (hardware-aware optimization, may be slower than what your hardware can be pushed to do)
- **Temporal + upscaling** = best quality but slower
- **Upscaling only** = faster, still very good for video
- **Scale > 100%** with descaling = restore low-res/compressed source content

---

## 🖥️ Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux (NVIDIA/AMD) | ✅ Full | Primary target |
| Windows | 🔄 *On the way* | Planning native build |
| macOS | ❌ Unplanned | Porting Vulkan requires external drivers and significant re-architecture. Not a priority for Sharptape's current scope. |

---

## Advanced Settings

| Setting | Impact | Default | Notes |
|---------|--------|---------|-------|
| **Tile Size** | VRAM vs speed | Auto | Lower = slower, less VRAM; larger = faster, more VRAM |
| **Batch Size** | VRAM vs speed | Auto | GPU-only setting; CPU forces batch=1 |
| **Model Tier** (CUGAN) | Quality vs speed | SE | SE=fast, Pro=balanced, NOSE=best quality+slowest |
| **CRF** (H.264/HEVC) | File size | 16 | Lower = better (0-51), 16 is good default for quality |

**⚠️ Warning:** Manual overrides can crash if unsupported by your hardware. Leave on Auto unless you know your GPU's limits.

---

## 🌍 Language Support

Sharptape includes multi-language support with automatic detection from your system locale. More languages are on the way! The UI will display in your system's language automatically.

---

## Troubleshooting

### "Black output" or corrupted frames
- Lower scale % or tile size
- Close other GPU-using apps (Discord, Chrome, etc.)
- Update drivers
- Try a different GPU if you have multiple

### "Out of memory" during processing
- Auto mode failed? Manually lower batch size or tile size
- Otherwise, enable auto mode
- Reduce input video resolution
- Process smaller clips one at a time

### BasicVSR++ won't run / Hardware Compatibility
- CUDA not installed (temporal = NVIDIA only).
- **AMD/ROCm support**: PyTorch execution on AMD hardware (via ROCm) is supported but remains untested.
- **Other hardware**: Non-NVIDIA/non-AMD hardware (Intel GPUs, etc.) may not be supported by PyTorch upscaling and will fall back to CPU (slowest).
- Try upscaler-only (NCNN) mode instead, which runs natively on Vulkan.
- AMD users: use Real-CUGAN/ESRGAN (NCNN Vulkan upscalers) with FFmpeg processing instead.

### Other issues?

Check out the [issues page](https://github.com/buwryme/sharptape/issues) for known issues and potential solutions, or open a new issue if you don't find what you're looking for.

---

## Contributing

Contributions are welcome and greatly appreciated. Please refer to our [Contributing Guidelines](CONTRIBUTING.md) for details on our code standards, workflow, and submission process.

All contributors and community are expected to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md).

If our Code of Conduct is violated, please contact the project maintainers privately here on GitHub.

---

## License

Apache License 2.0 — See [LICENSE](LICENSE)

---

## 🙌 Credits

- **Real-CUGAN / Real-ESRGAN** — [xinntao](https://github.com/xinntao)
- **BasicVSR++** — [Kelvin C.K. Chan](https://github.com/ckkelvinchan/BasicVSR_PlusPlus)
- **NCNN** — [Tencent](https://github.com/Tencent/ncnn)
- **GTK 4 / Libadwaita** — GNOME foundation
