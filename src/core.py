#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sharptape -- Core module
Constants, hardware detection, config, and utility functions.
"""

import os
import sys
import re
import gc
import json
import shutil
import signal
import subprocess
import threading
import time
import tempfile
import warnings
import contextlib
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# --- Paths & Constants ---

APP_ID = "net.buwryy.Sharptape"
DATA_DIR = Path.home() / ".local" / "share" / "sharptape"
SETTINGS_PATH = DATA_DIR / "settings.json"
VERSION_PATH = DATA_DIR / "VERSION"
LOG_DIR = DATA_DIR / "logs"

# Ensure logs directory exists on launch and create a timestamped session log file
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _SESSION_TIMESTAMP = time.strftime("%Y-%m-%d_%H.%M.%S")
    LOG_FILE = LOG_DIR / f"cmd_{_SESSION_TIMESTAMP}.log"
except Exception:
    LOG_FILE = None

def log_msg(msg: str, category: str = "Sharptape") -> None:
    """Print log message with timestamp and write to session log file."""
    t_str = time.strftime("%H:%M:%S")
    tag = f"[{category}]" if category else "[Sharptape]"
    full_line = f"[{t_str}] {tag} {msg}"
    print(full_line)
    if LOG_FILE:
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(full_line + "\n")
        except Exception:
            pass

# Read version from installed VERSION file (written by setup.sh)
try:
    __version__ = open(VERSION_PATH).read().strip()
except FileNotFoundError:
    __version__ = "unknown"

VIDEO_EXTS = (".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv",
             ".m4v", ".ts", ".mts", ".m2ts", ".vob", ".ogv")

NCNN_BINS = {
    "realcugan": "realcugan-ncnn-vulkan",
    "realesrgan": "realesrgan-ncnn-vulkan",
}

# Default install location for NCNN binaries
NCNN_BIN_DIR = Path.home() / ".local" / "bin"

MODEL_DIR = DATA_DIR / "models"

# GNOME HIG spacing constants
SPACING_XS = 6      # inline elements
SPACING_SM = 12     # standard padding
SPACING_MD = 18     # section separation
SPACING_LG = 24     # major breaks

SLIDER_CFG = {
    "resolution": {"min": 100, "max": 400, "step": 10, "default": 100, "fmt": "{}%"},
    "deblock":     {"min": 0,   "max": 100, "step": 1,   "default": 0,   "fmt": "{}%"},
    "denoise":     {"min": 0,   "max": 100, "step": 1,   "default": 0,   "fmt": "{}%"},
    "sharpen":     {"min": 0,   "max": 100, "step": 1,   "default": 0,   "fmt": "{}%"},
    "deblur":      {"min": 0,   "max": 100, "step": 1,   "default": 0,   "fmt": "{}%"},
    "antialias":   {"min":-100, "max": 100, "step": 1,   "default": 0,   "fmt": "{:+d}%"},
}

# Fixed width for value labels ensures slider alignment
VALUE_LABEL_WIDTH = 60  # pixels - enough for "+100%" or "400%"

# Progress constants for pipeline stages (0.0 to 1.0)
class PipelineProgress:
    """Progress milestones for each pipeline stage."""
    START = 0.00
    EXTRACT = 0.04
    PREFILTER = 0.08
    VSR_START = 0.18
    VSR_END = 0.45
    NCNN_START = 0.45
    NCNN_END = 0.70
    POSTFILTER = 0.75
    ASSEMBLE_START = 0.90
    ASSEMBLE_END = 1.00

# Model filenames (constants for maintainability)
BASICVSR_MODEL_FILENAME = "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth"


# --- Dynamic Hardware Detection ---

class HardwareTier(Enum):
    """Hardware capability tiers for adaptive processing."""
    NONE = 0       # No GPU / CPU only
    LOW = 1        # < 2GB VRAM (integrated GPUs, old cards)
    MEDIUM = 2     # 2-6GB VRAM (GTX 1650, RX 580, etc.)
    HIGH = 3       # 6-12GB VRAM (RTX 3060, RX 6700, etc.)
    ULTRA = 4      # > 12GB VRAM (RTX 3080+, RX 6800+, Apple Silicon)


@dataclass
class HardwareProfile:
    """Comprehensive hardware capability detection and adaptive settings.

    This class dynamically detects:
    - GPU type (NVIDIA CUDA, AMD ROCm/Vulkan, Apple Silicon MPS, Intel)
    - Available VRAM and compute capability
    - Optimal settings for each pipeline stage

    Usage:
        profile = HardwareProfile.detect()
        batch_size = profile.optimal_vsr_batch(1920, 1080)
        tile_size = profile.ncnn_tile_size
    """
    # Detection results
    gpu_name: str = "Unknown"
    gpu_vendor: str = "none"  # nvidia, amd, apple, intel, none
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    compute_capability: float = 0.0  # e.g., 7.5 for RTX 2070, 0 for CPU
    has_cuda: bool = False
    has_vulkan: bool = False
    has_mps: bool = False  # Apple Metal Performance Shaders
    # Multi-device support
    gpus: list = None  # All detected GPUs: [{"id": 0, "name": "...", "vendor": "...", "vram_total_mb": 0}, ...]
    cpu_name: str = "Unknown CPU"

    # Derived settings (auto-calculated from detection)
    tier: HardwareTier = HardwareTier.NONE
    vsr_batch_size: int = 1
    vsr_backbone_blocks: int = 10
    ncnn_tile_size: int = 128
    ncnn_jobs: str = "1:2:2"
    use_fp16: bool = False
    use_torch_compile: bool = False
    amp_enabled: bool = False

    @classmethod
    def detect(cls) -> 'HardwareProfile':
        """Detect hardware capabilities and return optimized profile."""
        profile = cls()
        profile.gpus = []  # Initialize empty GPU list
        profile._detect_cpu()
        profile._detect_gpu()
        profile._calculate_settings()
        return profile

    def _detect_gpu(self) -> None:
        """Run all GPU detection methods.

        Detection order matters: Vulkan enumerates GPUs in a system-specific order
        that typically places the iGPU (Intel/AMD APU) before discrete GPUs.
        We must match this order so our GPU list indices align with NCNN's
        Vulkan -g flag values.  Intel iGPU is detected FIRST.
        """
        self._detect_intel_gpu()        # iGPU first (Vulkan device 0 on hybrid systems)
        self._detect_nvidia_cuda()      # discrete GPUs next
        if not self.has_cuda:
            self._detect_amd_vulkan()
        if not self.has_cuda and not self.has_vulkan and not self.has_mps:
            self._detect_apple_silicon()
        self._detect_vulkan_ncnn()
        # Re-index sequentially so indices match Vulkan enumeration order
        for i, gpu in enumerate(self.gpus):
            gpu["id"] = i

    def _detect_nvidia_cuda(self) -> None:
        """Detect NVIDIA GPU(s) via nvidia-smi CLI (zero VRAM overhead).

        torch is NEVER imported here -- it would allocate ~1.5 GB CUDA context.
        All torch usage is isolated to the dedicated PyTorch processing thread.
        Supports multiple GPUs.
        """
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,compute_cap",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3
            )
            if r.returncode == 0 and r.stdout.strip():
                lines = [l.strip() for l in r.stdout.strip().splitlines() if l.strip()]
                for idx, line in enumerate(lines):
                    parts = line.split(", ")
                    if len(parts) >= 4:
                        gpu_info = {
                            "id": idx,
                            "name": parts[0].strip(),
                            "vendor": "nvidia",
                        }
                        try:
                            gpu_info["vram_total_mb"] = int(parts[1].strip())
                        except ValueError:
                            gpu_info["vram_total_mb"] = 0
                        try:
                            gpu_info["vram_free_mb"] = int(parts[2].strip())
                        except ValueError:
                            gpu_info["vram_free_mb"] = 0
                        try:
                            cc_parts = parts[3].strip().split(".")
                            if len(cc_parts) >= 2:
                                gpu_info["compute_cap"] = int(cc_parts[0]) + int(cc_parts[1]) * 0.1
                            elif len(cc_parts) == 1:
                                gpu_info["compute_cap"] = float(cc_parts[0])
                            else:
                                gpu_info["compute_cap"] = 0.0
                        except (ValueError, IndexError):
                            gpu_info["compute_cap"] = 0.0
                        self.gpus.append(gpu_info)
                # Set primary GPU to first NVIDIA entry for backward compat
                # (gpus[0] may be iGPU now, so find the first nvidia entry)
                nvidia_entries = [g for g in self.gpus if g.get("vendor") == "nvidia"]
                if nvidia_entries:
                    primary = nvidia_entries[0]
                    self.gpu_name = primary["name"]
                    self.gpu_vendor = "nvidia"
                    self.vram_total_mb = primary["vram_total_mb"]
                    self.vram_free_mb = primary["vram_free_mb"]
                    self.compute_capability = primary.get("compute_cap", 0.0)
                    self.has_cuda = True
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

    @staticmethod
    def _clean_gpu_name(name: str, vendor: str) -> str:
        """Clean up a GPU name for display.

        - Strips vendor prefix (e.g. 'Intel Corporation', 'Advanced Micro Devices, Inc.')
        - Removes codename parentheticals like '(Coffee Lake GT2)', '(TigerLake-H)'
        - Falls back to a generic but useful name if nothing meaningful remains
        """
        if not name or not name.strip():
            return f"{vendor} GPU"

        # Strip vendor prefixes that lspci sometimes includes in the device field
        vendor_prefixes = [
            "Intel Corporation", "Intel(R) Corporation",
            "Advanced Micro Devices, Inc.", "AMD",
        ]
        cleaned = name.strip()
        for prefix in vendor_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                break

        # Strip leading/trailing whitespace and dashes
        cleaned = cleaned.strip(" -")

        # Remove codename/architecture parentheticals at the end
        # e.g. "UHD Graphics 630 (Coffee Lake GT2)" -> "UHD Graphics 630"
        # but preserve meaningful suffixes like "Rev 4.0" etc.
        cleaned = re.sub(r'\s*\([^)]*(?:Lake|Gen|GT|Arc|RDNA|Navi|Navi|Polaris|Vega|Turing|Ampere|Ada|Hopper|Blackwell)[^)]*\)\s*$', '', cleaned, flags=re.IGNORECASE)
        # Also catch bare codenames like "(CNL)" or "(CML GT2)"
        cleaned = re.sub(r'\s*\([A-Z]{2,4}\s*(?:GT|UHD)?[0-9]*\)\s*$', '', cleaned)

        cleaned = cleaned.strip()
        if not cleaned:
            return f"{vendor} GPU"
        return cleaned

    @staticmethod
    def _parse_lspci_device_name(raw_line: str, vendor_prefix: str) -> str:
        """Extract a human-readable GPU device name from an lspci line.

        Uses _clean_gpu_name to strip vendor prefixes, codename
        parentheticals, and ensure a meaningful name is always returned.
        Never returns bare vendor names like 'Intel Corporation'.
        """
        line = raw_line.strip()
        candidate = None

        # Try lspci -m format: ... "Vendor" "Device Name" ...
        if '"' in line:
            parts = line.split('"')
            # parts[2] = vendor, parts[3] = device name
            if len(parts) >= 4:
                candidate = parts[3].strip()
                # If device field is just the vendor string, try the next field
                if candidate.lower() in (vendor_prefix.lower(), "intel corporation",
                                           "advanced micro devices, inc."):
                    candidate = parts[5].strip() if len(parts) >= 6 else None

        # Fallback: plain lspci format: "Slot: Class: Vendor DeviceName (rev)"
        if candidate is None and ':' in line:
            desc = line.split(':', 1)[1].strip()
            for prefix in [vendor_prefix, "Intel Corporation", "Advanced Micro Devices, Inc.", "AMD"]:
                if desc.lower().startswith(prefix.lower()):
                    candidate = desc[len(prefix):].strip()
                    break
            if candidate is None:
                candidate = desc

        # Clean the name (strip vendor, codenames)
        name = HardwareProfile._clean_gpu_name(candidate or "", vendor_prefix)
        return name

    def _detect_amd_vulkan(self) -> None:
        """Detect AMD GPU(s) via Vulkan/lspci."""
        try:
            r = subprocess.run(
                ["lspci", "-m"],
                capture_output=True, text=True, timeout=3
            )
            amd_idx = 0
            for line in r.stdout.splitlines():
                line_lower = line.lower()
                if any(cls in line_lower for cls in ["vga", "3d", "display"]):
                    if any(vendor in line_lower for vendor in ["amd", "radeon", "ati"]):
                        gpu_name = self._parse_lspci_device_name(line, "AMD")
                        gpu_info = {
                            "id": amd_idx,
                            "name": gpu_name,
                            "vendor": "amd",
                            "vram_total_mb": 0,
                            "vram_free_mb": 0,
                        }
                        self.gpus.append(gpu_info)
                        # Set primary if not already set
                        if not self.has_vulkan:
                            self.gpu_name = gpu_name
                            self.gpu_vendor = "amd"
                            self.has_vulkan = True
                        amd_idx += 1
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try to get VRAM info from AMD
        if self.has_vulkan and self.vram_total_mb == 0:
            try:
                for path in Path("/sys/class/drm").glob("card*/device/mem_info_vram_total"):
                    if path.exists():
                        vram_kb = int(path.read_text().strip())
                        vram_mb = vram_kb // 1024
                        self.vram_total_mb = vram_mb
                        self.vram_free_mb = int(vram_mb * 0.7)
                        # Update the AMD GPU entry in gpus list
                        for g in self.gpus:
                            if g["vendor"] == "amd":
                                g["vram_total_mb"] = vram_mb
                                g["vram_free_mb"] = int(vram_mb * 0.7)
                                break
                        break
            except Exception:
                pass

    def _detect_apple_silicon(self) -> None:
        """Detect Apple Silicon GPU via sysctl/platform (zero VRAM overhead).

        Avoids importing torch which would load the entire runtime.
        """
        import platform
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            return
        try:
            import subprocess
            r = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"],
                capture_output=True, text=True, timeout=3
            )
            if r.returncode == 0 and r.stdout.strip() == "1":
                self.has_mps = True
                self.gpu_vendor = "apple"
                try:
                    import psutil
                    total_ram_gb = psutil.virtual_memory().total / (1024**3)
                    self.vram_total_mb = int(total_ram_gb * 0.8 * 1024)
                    self.vram_free_mb = int(total_ram_gb * 0.6 * 1024)
                    self.gpu_name = f"Apple {'M1' if total_ram_gb < 16 else 'M2/M3' if total_ram_gb < 32 else 'M2/M3 Pro/Max'}"
                except ImportError:
                    self.vram_total_mb = 8192
                    self.vram_free_mb = 6144
                    self.gpu_name = "Apple Silicon GPU"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _detect_intel_gpu(self) -> None:
        """Detect Intel integrated/discrete GPU.

        Always runs (even when a discrete GPU is present) so the GPU list
        includes iGPUs in Vulkan enumeration order.  Intel Arc (discrete)
        GPUs are NOT tagged as integrated.
        """
        try:
            r = subprocess.run(
                ["lspci", "-m"],
                capture_output=True, text=True, timeout=3
            )
            for line in r.stdout.splitlines():
                if "Intel" in line and ("Graphics" in line or "VGA" in line):
                    gpu_name = self._parse_lspci_device_name(line, "Intel")
                    # Intel Arc is discrete; everything else from Intel is iGPU
                    is_integrated = "arc" not in gpu_name.lower()
                    if is_integrated:
                        gpu_name = "UHD Graphics"
                    gpu_info = {
                        "id": len(self.gpus),
                        "name": gpu_name,
                        "vendor": "intel",
                        "vram_total_mb": 0,
                        "vram_free_mb": 0,
                        "integrated": is_integrated,
                    }
                    self.gpus.append(gpu_info)
                    # Only set as primary if no discrete GPU detected yet
                    if not self.has_cuda and not self.has_vulkan and not self.has_mps:
                        self.gpu_name = gpu_name
                        self.gpu_vendor = "intel"
                        self.has_vulkan = True
                    break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _detect_cpu(self) -> None:
        """Detect CPU model name via lscpu (zero overhead)."""
        try:
            r = subprocess.run(
                ["lscpu"],
                capture_output=True, text=True, timeout=3
            )
            for line in r.stdout.splitlines():
                if "Model name:" in line:
                    self.cpu_name = line.split(":", 1)[1].strip()
                    break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Fallback: try /proc/cpuinfo (Linux)
        if self.cpu_name == "Unknown CPU":
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            self.cpu_name = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass

    def _detect_vulkan_ncnn(self) -> None:
        """Check if NCNN Vulkan binaries are available."""
        for name in NCNN_BINS.values():
            if find_bin(name):
                self.has_vulkan = True
                break

    def _calculate_settings(self) -> None:
        """Calculate optimal settings based on detected hardware."""
        vram_gb = self.vram_total_mb / 1024.0

        # Determine hardware tier
        if vram_gb < 0.5 or (not self.has_cuda and not self.has_mps and not self.has_vulkan):
            self.tier = HardwareTier.NONE
        elif vram_gb < 2:
            self.tier = HardwareTier.LOW
        elif vram_gb < 6:
            self.tier = HardwareTier.MEDIUM
        elif vram_gb < 12:
            self.tier = HardwareTier.HIGH
        else:
            self.tier = HardwareTier.ULTRA

        # VSR (BasicVSR++) settings
        self._calculate_vsr_settings(vram_gb)

        # NCNN upscaler settings
        self._calculate_ncnn_settings(vram_gb)

        # Precision/compilation settings
        self._calculate_precision_settings()

    def _calculate_vsr_settings(self, vram_gb: float) -> None:
        """Calculate optimal VSR batch size and model complexity.
        
        Backbone blocks: conservative baseline ~4 for midrange.
        Minimal tier variation; users can tune in Advanced settings.
        """
        if self.tier == HardwareTier.NONE:
            self.vsr_batch_size = 1
            self.vsr_backbone_blocks = 3  # Minimal backbone on CPU
        elif self.tier == HardwareTier.LOW:
            # 512MB-2GB: Very conservative, single frame
            self.vsr_batch_size = 1
            self.vsr_backbone_blocks = 4
        elif self.tier == HardwareTier.MEDIUM:
            # 2-6GB: Small batches possible, conservative backbone
            self.vsr_batch_size = 2
            self.vsr_backbone_blocks = 4
        elif self.tier == HardwareTier.HIGH:
            # 6-12GB: Good batching, slightly higher quality
            self.vsr_batch_size = 4
            self.vsr_backbone_blocks = 5
        else:  # ULTRA
            # >12GB: Aggressive batching, best quality
            self.vsr_batch_size = 8
            self.vsr_backbone_blocks = 6

    def _calculate_ncnn_settings(self, vram_gb: float) -> None:
        """Calculate optimal NCNN tile size and job count."""
        if self.tier == HardwareTier.NONE:
            # CPU-only: small tiles, single job
            self.ncnn_tile_size = 64
            self.ncnn_jobs = "1:1:1"
        elif self.tier == HardwareTier.LOW:
            # Low VRAM: conservative tiling
            self.ncnn_tile_size = 128
            self.ncnn_jobs = "1:2:2"
        elif self.tier == HardwareTier.MEDIUM:
            # Medium VRAM: balanced
            self.ncnn_tile_size = 224
            self.ncnn_jobs = "1:4:4"
        elif self.tier == HardwareTier.HIGH:
            # High VRAM: larger tiles
            self.ncnn_tile_size = 256
            self.ncnn_jobs = "2:4:4"
        else:  # ULTRA
            # Lots of VRAM: maximum throughput
            self.ncnn_tile_size = 384
            self.ncnn_jobs = "2:8:8"

    def _calculate_precision_settings(self) -> None:
        """Determine FP16/AMP availability."""
        # FP16 needs compute capability >= 5.3 (Maxwell GM20X+, Pascal+)
        self.use_fp16 = self.has_cuda and self.compute_capability >= 5.3

        # AMP is safe on any modern NVIDIA GPU or Apple Silicon
        self.amp_enabled = self.use_fp16 or self.has_mps

    def optimal_vsr_batch(self, width: int, height: int) -> int:
        """Get optimal batch size for given video resolution.

        Adjusts base batch size down for high-resolution content
        to prevent OOM on smaller GPUs.
        """
        megapixels = (width * height) / 1_000_000

        # Scale factor: 1080p = 1.0x, 4K = 4.0x
        resolution_factor = max(1.0, megapixels / 2.0)

        # Reduce batch size for high-res content
        adjusted = max(1, int(self.vsr_batch_size / resolution_factor))

        # Further limit by available VRAM
        if self.vram_free_mb > 0:
            # Estimate memory per frame at target resolution (fp16: ~6MB per MP)
            est_mem_per_frame_mb = megapixels * 6 if self.use_fp16 else megapixels * 12
            max_by_vram = max(1, int((self.vram_free_mb * 0.5) / max(est_mem_per_frame_mb, 1)))
            adjusted = min(adjusted, max_by_vram)

        return adjusted

    def adaptive_ncnn_params(self, scale_factor: int = 2) -> dict:
        """Get NCNN parameters adapted for current hardware and scale factor.

        Returns dict with keys: jobs, tile_size, gpu_id
        """
        params = {
            "jobs": self.ncnn_jobs,
            "tile_size": self.ncnn_tile_size,
            "gpu_id": 0,
        }

        # For higher scale factors, reduce tile size to manage memory
        if scale_factor >= 4 and self.tier.value <= HardwareTier.MEDIUM.value:
            params["tile_size"] = max(64, self.ncnn_tile_size // 2)

        return params

    def summary_string(self) -> str:
        """Human-readable summary of detected hardware."""
        parts = [f"{self.gpu_name}"]
        if self.vram_total_mb > 0:
            parts.append(f"{self.vram_total_mb // 1024}GB")
        parts.append(f"({self.tier.name})")
        return " ".join(parts)

    def log_details(self, user_settings: dict = None) -> str:
        """Detailed logging info for debugging.

        Args:
            user_settings: If provided, show these instead of hardware defaults.
                          This makes logs match what's actually being used.
        """
        if user_settings:
            # User has overridden settings - show THEIR values with note
            lines = [
                f"Hardware Profile (user-configured overrides active):",
                f"  GPU: {self.gpu_name} ({self.gpu_vendor})",
                f"  VRAM: {self.vram_free_mb}MB free / {self.vram_total_mb}MB total",
                f"  Tier: {self.tier.name} <- hardware tier (for reference)",
                f"  VSR: batch={user_settings.get('vsr_batch', '?')}, blocks={user_settings.get('vsr_blocks', '?')} <- USER SET",
                f"  NCNN: tile={'auto' if user_settings.get('ncnn_tile_auto', True) else user_settings.get('ncnn_tile', '?')}, jobs={user_settings.get('ncnn_jobs', '?')} <- USER SET",
                f"  Model Tier: {user_settings.get('cugan_tier', '?')} <- USER SET",
                f"  Precision: fp16={user_settings.get('use_fp16', '?')}, AMP={user_settings.get('use_amp', '?')}",
            ]
        else:
            # Auto mode - show hardware-detected values
            lines = [
                f"Hardware Profile (auto-detected optimal):",
                f"  GPU: {self.gpu_name} ({self.gpu_vendor})",
                f"  VRAM: {self.vram_free_mb}MB free / {self.vram_total_mb}MB total",
                f"  Compute Capability: {self.compute_capability}",
                f"  Tier: {self.tier.name}",
                f"  VSR: batch={self.vsr_batch_size}, blocks={self.vsr_backbone_blocks}",
                f"  NCNN: tile={self.ncnn_tile_size}, jobs={self.ncnn_jobs}",
                f"  Precision: fp16={self.use_fp16}, AMP={self.amp_enabled}",
            ]
        return "\n".join(lines)


# --- Helpers ---

def find_bin(name: str) -> Optional[Path]:
    """Find binary: check ~/.local/bin first, then PATH."""
    # Check ~/.local/bin first (user-local install, not always in PATH)
    local_bin = Path.home() / ".local" / "bin" / name
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return local_bin

    # Fall back to system PATH
    p = shutil.which(name)
    return Path(p) if p else None


def find_ncnn_bin(name: str) -> Optional[Path]:
    """Find NCNN binary. Checks NCNN_BIN_DIR (~/.local/bin) first, then system PATH."""
    bin_path = NCNN_BIN_DIR / name
    if bin_path.is_file() and os.access(bin_path, os.X_OK):
        return bin_path
    
    # Fallback to system PATH if not found in ~/.local/bin
    p = shutil.which(name)
    return Path(p) if p else None


def check_ncnn_binaries() -> Tuple[bool, bool, str]:
    """Check if both NCNN binaries are available.

    Returns:
        (has_realesrgan, has_realcugan, error_message)
    """
    realesrgan = find_ncnn_bin(NCNN_BINS["realesrgan"])
    realcugan = find_ncnn_bin(NCNN_BINS["realcugan"])

    missing = []
    if not realesrgan:
        missing.append(f"realesrgan-ncnn-vulkan not found at {NCNN_BIN_DIR}")
    if not realcugan:
        missing.append(f"realcugan-ncnn-vulkan not found at {NCNN_BIN_DIR}")

    return (bool(realesrgan), bool(realcugan), "; ".join(missing) if missing else "")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    fps_str: str
    duration: float
    codec: str
    bitrate: Optional[int]
    frames: Optional[int]
    pix_fmt: str
    color_space: str
    color_primaries: str
    color_trc: str

    @property
    def res(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def dur(self) -> str:
        s = max(0, self.duration)
        h, m = int(s // 3600), int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"


def parse_fps(raw: str) -> float:
    raw = raw.strip()
    if "/" in raw:
        p = raw.split("/")
        try:
            return float(p[0]) / float(p[1]) if float(p[1]) != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def probe_video(path: Path, timeout: float = 30.0) -> VideoInfo:
    ffprobe = find_bin("ffprobe")
    if not ffprobe:
        raise FileNotFoundError("ffprobe not found")
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    r = subprocess.run(
        [str(ffprobe), "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", "-select_streams", "v:0", str(path)],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:200]}")

    d = json.loads(r.stdout)
    streams = d.get("streams", [])
    if not streams:
        raise RuntimeError("No video streams found in file metadata")
    s = streams[0]
    fmt = d.get("format", {})

    fps_raw = s.get("r_frame_rate", s.get("avg_frame_rate", "0/1"))
    dur_s = s.get("duration") or fmt.get("duration", "0")
    dur = float(dur_s) if dur_s != "N/A" else 0.0

    br_raw = s.get("bit_rate") or fmt.get("bit_rate")
    br = int(br_raw) if br_raw and br_raw != "N/A" else None

    nb = s.get("nb_frames")
    total = int(nb) if nb and nb != "N/A" else (int(dur * parse_fps(fps_raw)) if dur > 0 else None)

    cs = s.get("color_space", "")
    cp = s.get("color_primaries", "")
    ct = s.get("color_transfer", "")

    return VideoInfo(
        width=int(s.get("width", 0)), height=int(s.get("height", 0)),
        fps=parse_fps(fps_raw), fps_str=fps_raw,
        duration=dur, codec=s.get("codec_name", "?"),
        bitrate=br, frames=total, pix_fmt=s.get("pix_fmt", "?"),
        color_space=cs, color_primaries=cp, color_trc=ct,
    )


def target_dims(w: int, h: int, pct: int) -> Tuple[int, int]:
    factor = pct / 100.0
    tw, th = int(w * factor), int(h * factor)
    return tw + tw % 2, th + th % 2  # even numbers for codecs


# --- Enums & Config ---

class Engine(Enum):
    HYBRID = "hybrid"
    ESRGAN = "esrgan"
    FFMPEG = "ffmpeg"

class GPUBackend(Enum):
    VULKAN = "vulkan"
    CUDA = "cuda"
    CPU = "cpu"

class Codec(Enum):
    H264 = "h264"
    HEVC = "hevc"
    PRORES = "prores"


class TemporalMethod(Enum):
    """Temporal processing method selection."""
    BASICVSR = "basicvsr"      # Neural network (BasicVSR++), slower but better quality
    FFMPEG = "ffmpeg"          # FFmpeg filters, faster but simpler


@dataclass
class Config:
    """User settings, persisted to disk."""
    input_path: str = ""
    output_path: str = ""
    scale_pct: int = SLIDER_CFG["resolution"]["default"]
    descale: bool = False
    deblock: int = SLIDER_CFG["deblock"]["default"]
    denoise: int = SLIDER_CFG["denoise"]["default"]
    sharpen: int = SLIDER_CFG["sharpen"]["default"]
    deblur: int = SLIDER_CFG["deblur"]["default"]
    antialias: int = SLIDER_CFG["antialias"]["default"]
    engine: Engine = Engine.HYBRID
    gpu: GPUBackend = GPUBackend.VULKAN
    codec: Codec = Codec.H264
    temporal_method: TemporalMethod = TemporalMethod.BASICVSR
    suppress_gpu_warning: bool = False  # Don't show "No GPU" popup again

    # Advanced AI settings (persisted to JSON)
    vsr_batch: int = 2           # BasicVSR++ batch size
    vsr_blocks: int = 4          # BasicVSR++ backbone blocks (conservative baseline)
    ncnn_tile: int = 224         # NCNN tile size
    ncnn_tile_auto: bool = True  # True=auto (-t 0), False=use ncnn_tile value
    ncnn_jobs: str = "1:4:4"     # NCNN thread count format
    cugan_tier: str = "se"       # Real-CUGAN model tier (se/pro/nose)
    use_fp16: bool = True        # FP16 precision
    use_amp: bool = True         # Automatic Mixed Precision
    auto_config: bool = True     # Auto-configure AI models toggle
    crf_value: int = 16          # Codec CRF (0-51, lower=better quality) for H.264/HEVC
    device_id: str = "auto"      # Device selection: "auto", "cpu", "gpu:0", "gpu:1", etc.
    upscaler_model: str = "cugan"# Upscaler: "cugan" (anime) or "esrgan" (realistic)
    post_denoise: bool = False   # Light denoising in post-filter when NOSE tier is selected

    # All known config keys for cleanup validation
    _KNOWN_KEYS = frozenset({
        "scale_pct", "descale", "deblock", "denoise", "sharpen", "deblur",
        "antialias", "engine", "gpu", "codec", "temporal_method",
        "suppress_gpu_warning", "vsr_batch", "vsr_blocks", "ncnn_tile",
        "ncnn_tile_auto", "ncnn_jobs", "cugan_tier", "use_fp16", "use_amp",
        "auto_config", "crf_value",
        "device_id", "upscaler_model", "post_denoise", "input_path", "output_path",
    })


    def save(self) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            d = {
                "scale_pct": self.scale_pct,
                "descale": self.descale,
                "deblock": self.deblock,
                "denoise": self.denoise,
                "sharpen": self.sharpen,
                "deblur": self.deblur,
                "antialias": self.antialias,
                "engine": self.engine.value,
                "gpu": self.gpu.value,
                "codec": self.codec.value,
                "temporal_method": self.temporal_method.value,
                "suppress_gpu_warning": self.suppress_gpu_warning,
                # Advanced AI settings
                "vsr_batch": self.vsr_batch,
                "vsr_blocks": self.vsr_blocks,
                "ncnn_tile": self.ncnn_tile,
                "ncnn_tile_auto": self.ncnn_tile_auto,
                "ncnn_jobs": self.ncnn_jobs,
                "cugan_tier": self.cugan_tier,
                "use_fp16": self.use_fp16,
                "use_amp": self.use_amp,
                "auto_config": self.auto_config,
                "crf_value": self.crf_value,
                "device_id": self.device_id,
                "upscaler_model": self.upscaler_model,
                "post_denoise": self.post_denoise,
            }
            with open(SETTINGS_PATH, 'w') as f:
                json.dump(d, f, indent=2)
        except Exception as e:
            print(f"[warn] couldn't save settings: {e}")

    @classmethod
    def load(cls) -> Tuple['Config', Optional[Dict[str, Any]]]:
        """Load config from disk. Returns (config, corruption_info).

        corruption_info is None if config is clean, or a dict with:
          - "preserved": list of successfully loaded key names
          - "missing": list of expected keys that were absent
          - "extra": list of unknown keys that were removed
          - "total_corrupted": bool (True if 0 values preserved)
        """
        c = cls()
        if not SETTINGS_PATH.exists():
            return c, None  # First launch, no config yet
        try:
            with open(SETTINGS_PATH) as f:
                raw = f.read().strip()
            if not raw:
                # Empty file -- treat as first launch
                print("[sharptape] config file is empty, using defaults")
                return c, None
            d = json.loads(raw)
            if not isinstance(d, dict):
                raise ValueError("config root is not a JSON object")

            # Non-invasive cleanup: remove unknown keys
            extra_keys = sorted(set(d.keys()) - cls._KNOWN_KEYS)
            if extra_keys:
                print(f"[sharptape] removing unknown config keys: {extra_keys}")
                for k in extra_keys:
                    del d[k]

            preserved = []
            missing = []

            # Enum fields
            for attr, enum_cls in [("engine", Engine), ("gpu", GPUBackend),
                                   ("codec", Codec), ("temporal_method", TemporalMethod)]:
                if attr in d:
                    try:
                        setattr(c, attr, enum_cls(d[attr]))
                        preserved.append(attr)
                    except (ValueError, KeyError):
                        missing.append(attr)
                else:
                    missing.append(attr)

            if "suppress_gpu_warning" in d:
                c.suppress_gpu_warning = bool(d["suppress_gpu_warning"])
                preserved.append("suppress_gpu_warning")
            else:
                missing.append("suppress_gpu_warning")

            if "post_denoise" in d:
                c.post_denoise = bool(d["post_denoise"])
                preserved.append("post_denoise")
            else:
                missing.append("post_denoise")

            # Simple scalar fields
            for k in ["scale_pct", "descale", "deblock", "denoise",
                      "sharpen", "deblur", "antialias"]:
                if k in d:
                    setattr(c, k, d[k])
                    preserved.append(k)
                else:
                    missing.append(k)

            # Advanced AI settings
            for k in ["vsr_batch", "vsr_blocks", "ncnn_tile", "ncnn_tile_auto", "ncnn_jobs",
                      "cugan_tier", "use_fp16", "use_amp",
                      "auto_config", "crf_value", "device_id",
                      "upscaler_model"]:
                if k in d:
                    setattr(c, k, d[k])
                    preserved.append(k)
                else:
                    missing.append(k)

            # If we cleaned up extra keys, write back the cleaned config
            if extra_keys:
                c.save()

            # Determine if config is corrupted (missing expected fields)
            # Only flag as corrupted if there are missing fields AND the file existed with content
            if missing:
                total_expected = len(preserved) + len(missing)
                info = {
                    "preserved": sorted(preserved),
                    "missing": sorted(missing),
                    "extra": sorted(extra_keys),
                    "total_corrupted": len(preserved) == 0,
                    "total_count": total_expected,
                }
                print(f"[sharptape] config has {len(missing)} missing fields, {len(preserved)} preserved out of {total_expected}")
                return c, info

            print(f"[sharptape] loaded settings from {SETTINGS_PATH}")
            return c, None

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[sharptape] config parse error: {e}")
            # Totally corrupted JSON
            info = {
                "preserved": [],
                "missing": [],
                "extra": [],
                "total_corrupted": True,
                "parse_error": str(e),
            }
            return c, info
        except Exception as e:
            print(f"[warn] couldn't load settings: {e}")
            return c, None

    # filter params computed from slider values
    @property
    def deblock_filt(self) -> Dict[str, Any]:
        t = self.deblock / 100.0
        if t < 0.05:
            return {}
        return {
            "filter": 1 if t > 0.5 else 0,
            "block": 8,
            "alpha": round(min(0.02 + t * 0.13, 1.0), 3),
            "beta": round(min(0.01 + t * 0.09, 1.0), 3),
        }

    @property
    def denoise_filt(self) -> Dict[str, float]:
        t = self.denoise / 100.0
        if t < 0.05:
            return {"luma_spatial": 0, "chroma_spatial": 0, "luma_tmp": 0, "chroma_tmp": 0}
        return {
            "luma_spatial": round(t * 6.0, 2),
            "chroma_spatial": round(t * 5.0, 2),
            "luma_tmp": round(t * 10.0, 2),
            "chroma_tmp": round(t * 7.0, 2),
        }

    @property
    def sharpen_filt(self) -> Dict[str, float]:
        t = self.sharpen / 100.0
        return {"strength": round(t, 2)}

    @property
    def unsharp_filt(self) -> Dict[str, Any]:
        t = self.deblur / 100.0
        amt = round(t * 2.5, 2)
        return {"msize_x": 5, "msize_y": 5, "luma_amount": amt, "chroma_amount": round(amt * 0.5, 2)}

    @property
    def aa_filt(self) -> Dict[str, Any]:
        v = self.antialias
        if v < 0:
            s = abs(v) / 100.0
            return {"mode": "crispen", "filter": "unsharp", "amount": round(-s * 1.5, 2), "msize": 3}
        elif v > 0:
            s = v / 100.0
            # FFmpeg bilateral filter (FFmpeg 7.x): bilateral=radius:sigmaR
            # ONLY 2 PARAMETERS:
            # - radius: spatial radius (int, default 4, range ~1-10)
            # - sigmaR: range sigma (float, MUST be in [0, 1]!)
            # Note: sigmaS is auto-calculated from radius in newer FFmpeg versions
            return {"mode": "smooth", "filter": "bilateral",
                    "radius": round(2 + s * 6),              # 2-8 range (integer-like)
                    "sigma_r": round(0.05 + s * 0.5, 2)}     # 0.05-0.55 range (MUST be < 1)
        return {"mode": "none"}

    def pre_filters(self) -> Optional[str]:
        parts = []
        db = self.deblock_filt
        if db:
            parts.append(f"deblock={db['filter']}:{db['block']}:{db['alpha']}:{db['beta']}")
        dn = self.denoise_filt
        if dn.get("luma_spatial", 0) > 0 or dn.get("luma_tmp", 0) > 0:
            parts.append(f"hqdn3d={dn['luma_spatial']}:{dn['chroma_spatial']}:{dn['luma_tmp']}:{dn['chroma_tmp']}")
        return ",".join(parts) if parts else None

    def post_filters(self) -> Optional[str]:
        parts = []
        # Post-denoise for NOSE tier: light hqdn3d denoising pass.
        # Only applies when: (1) user explicitly enabled it, (2) NOSE tier is
        # selected, (3) auto-config is OFF (auto always uses SE, not NOSE).
        if self.post_denoise and not self.auto_config and self.cugan_tier == "nose":
            parts.append("hqdn3d=2.0:1.5:3.0:2.0")
        cas = self.sharpen_filt
        if cas["strength"] > 0:
            parts.append(f"cas={cas['strength']}")
        us = self.unsharp_filt
        if us["luma_amount"] > 0:
            parts.append(f"unsharp=luma_msize_x={us['msize_x']}:luma_msize_y={us['msize_y']}:luma_amount={us['luma_amount']}:chroma_amount={us['chroma_amount']}")
        aa = self.aa_filt
        if aa["mode"] == "crispen":
            parts.append(f"unsharp=luma_msize_x={aa['msize']}:luma_msize_y={aa['msize']}:luma_amount={aa['amount']}:chroma_amount=0")
        elif aa["mode"] == "smooth":
            # FFmpeg 7.x bilateral: ONLY 2 params (radius:sigmaR)
            parts.append(f"bilateral={aa['radius']}:{aa['sigma_r']}")
        return ",".join(parts) if parts else None


# --- Progress parsing ---

PROG_RE = re.compile(
    r"Frame=\s*(?P<f>\d+)\sfps=\s*(?P<fps>[\d.]+)\sq=(?P<q>[\d.-]+)"
    r"\ssize=\s*(?P<s>[\d]+)\stime=(?P<t>[\d:.]+)\sbitrate=\s*(?P<b>[\d.]+)\sspeed=\s*(?P<sp>[\d.x]+)"
)
TIME_RE = re.compile(r"(?P<h>\d+):(?P<m>\d+):(?P<s>[\d.]+)")


@dataclass
class ProgData:
    frame: int
    fps: float
    secs: float
    pct: float


def parse_prog(line: str, dur: float) -> Optional[ProgData]:
    m = PROG_RE.search(line)
    if not m:
        return None
    d = m.groupdict()
    tm = TIME_RE.search(d["t"])
    s = 0.0
    if tm:
        s = int(tm.group("h"))*3600 + int(tm.group("m"))*60 + float(tm.group("s"))
    return ProgData(int(d["f"]), float(d["fps"]), s, min(s/dur*100, 100) if dur > 0 else 0)
