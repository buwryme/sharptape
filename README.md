<div align="center">

<img src="src/assets/icon.svg" width="128" height="128" alt="Sharptape Icon">

# sharptape

ai video upscaler & temporal enhancer for linux.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-4caf50?style=flat-square)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-linux-3b82f6?style=flat-square&logo=linux&logoColor=white)](#installation)
[![GTK](https://img.shields.io/badge/GTK-4.0%20%7C%20libadwaita-f97316?style=flat-square&logo=gnome&logoColor=white)](#features)

</div>

---

## requirements
- python
- gtk4 & libadwaita
- ffmpeg & vulkan drivers
- nvidia cuda or amd rocm (optional, for temporal pass)

## usage

```bash
git clone https://git.buwryy.net/buwryy/sharptape.git
cd sharptape/sharptape
chmod +x setup.sh; ./setup.sh
```

run `sharptape` from your terminal or app grid.

> [!IMPORTANT]
> do not install pytorch from system repos if you need gpu acceleration. let `setup.sh` handle it or use the official pip index.

## features

*   **spatial upscaling:** real-cugan (anime) & real-esrgan (photo) via ncnn-vulkan.
*   **temporal enhancement:** basicvsr++ for smooth motion consistency (pytorch/cuda).
*   **smart processing:** auto hardware detection with manual overrides for tile/batch size.
*   **codec support:** h.264, hevc, and prores output with precise color mapping.

## how it works

1.  detects gpu capabilities to set optimal tile and batch sizes.
2.  runs ncnn-vulkan upscalers for fast spatial enhancement.
3.  optionally pipes frames through pytorch for temporal smoothing.
4.  re-encodes via ffmpeg with proper color space handling.

---

for inquiries contact **@buwryy** on discord

<div align="center">

**made with ♥ by [buwryme](https://git.buwryy.net/buwryy)**

</div>
