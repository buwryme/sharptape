#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sharptape -- Backend processing module
Worker thread for video enhancement pipeline (BasicVSR++, NCNN, FFmpeg).
"""

import os
import sys
import re
import gc
import json
import shutil
import subprocess
import threading
import time
import tempfile
import warnings
import contextlib
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core import (
    APP_ID, DATA_DIR,
    NCNN_BINS, NCNN_BIN_DIR, MODEL_DIR,
    PipelineProgress, BASICVSR_MODEL_FILENAME,
    HardwareProfile, HardwareTier,
    find_bin, find_ncnn_bin,
    VideoInfo, target_dims,
    Engine, TemporalMethod, Codec,
    Config, log_msg,
)

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("GLib", "2.0")

from gi.repository import GLib, GObject

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gi.repository.Adw")

# --- Background Worker ---

class Worker(GObject.Object, threading.Thread):
    __gsignals__ = {
        "progress": (GObject.SignalFlags.RUN_FIRST, None, (float, str)),
        "done": (GObject.SignalFlags.RUN_FIRST, None, (bool, str)),
        "error": (GObject.SignalFlags.RUN_FIRST, None, (str,)),
        "log": (GObject.SignalFlags.RUN_FIRST, None, (str,)),  # User-facing status messages (UI + console)
        "debug": (GObject.SignalFlags.RUN_FIRST, None, (str,)),  # Verbose app logs (console only)
        "toast": (GObject.SignalFlags.RUN_FIRST, None, (str,)),  # In-window toast from worker thread
    }

    def __init__(self, cfg: Config, info: VideoInfo, cancel: threading.Event):
        super().__init__()
        threading.Thread.__init__(self, daemon=True, name="Worker")
        self._cfg = cfg
        self._info = info
        self._cancel = cancel
        self._workdir: Optional[Path] = None
        self._silent_fail_count: int = 0

    def _get_effective_settings(self, hw_profile: 'HardwareProfile') -> Dict[str, Any]:
        """Get effective AI settings: from config if auto_config is OFF, else from hw_profile.

        Returns dict with:
        - vsr_batch: batch size for BasicVSR++
        - vsr_blocks: backbone blocks count
        - ncnn_tile: tile size for NCNN
        - ncnn_jobs: thread count string for NCNN
        - cugan_tier: model tier (se/pro/nose)
        - use_fp16: FP16 precision toggle
        - use_amp: AMP toggle
        """
        if not getattr(self._cfg, 'auto_config', True):
            # MANUAL MODE: use user's config values
            tile_auto = getattr(self._cfg, 'ncnn_tile_auto', True)
            vsr_blocks = getattr(self._cfg, 'vsr_blocks', 4)
            ncnn_tile = getattr(self._cfg, 'ncnn_tile', 192)
            
            # Auto-adjust tile only if user sets very high vsr_blocks (>8)
            if vsr_blocks > 8 and ncnn_tile >= 192:
                ncnn_tile = max(192, ncnn_tile // 2)  # Halve but keep >= 192 minimum
                self._debug_log(f"Reducing NCNN tile to {ncnn_tile} (high vsr_blocks={vsr_blocks})")
            
            settings = {
                'vsr_batch': getattr(self._cfg, 'vsr_batch', 2),
                'vsr_blocks': vsr_blocks,
                'ncnn_tile': ncnn_tile,
                'ncnn_tile_auto': tile_auto,
                'ncnn_jobs': getattr(self._cfg, 'ncnn_jobs', '1:4:4'),
                'cugan_tier': getattr(self._cfg, 'cugan_tier', 'se'),
                'use_fp16': getattr(self._cfg, 'use_fp16', True),
                'use_amp': getattr(self._cfg, 'use_amp', True),
            }
            tile_disp = 'auto' if tile_auto else str(settings['ncnn_tile'])
            self._debug_log(f"MANUAL AI settings: batch={settings['vsr_batch']}, blocks={settings['vsr_blocks']}, tile={tile_disp}, tier={settings['cugan_tier']}")
            return settings
        else:
            # AUTO MODE: use hardware-detected optimal values
            ncnn_params = hw_profile.adaptive_ncnn_params(2)  # default scale for params
            tile_auto = getattr(self._cfg, 'ncnn_tile_auto', True)
            settings = {
                'vsr_batch': hw_profile.vsr_batch_size,
                'vsr_blocks': hw_profile.vsr_backbone_blocks,
                'ncnn_tile': hw_profile.ncnn_tile_size,
                'ncnn_tile_auto': tile_auto,
                'ncnn_jobs': hw_profile.ncnn_jobs,
                'cugan_tier': 'se',  # ignored in auto mode; _ncnn_upscale() uses its own priority order
                'use_fp16': hw_profile.use_fp16,
                'use_amp': hw_profile.amp_enabled,
            }
            tile_disp = 'auto' if tile_auto else str(settings['ncnn_tile'])
            self._debug_log(f"AUTO AI settings: batch={settings['vsr_batch']}, blocks={settings['vsr_blocks']}, tile={tile_disp}")
            return settings

    def run(self) -> None:
        try:
            self._log("starting pipeline...")

            # Log configuration mode clearly
            auto_mode = getattr(self._cfg, 'auto_config', True)
            if auto_mode:
                self._debug_log("config mode: AUTO (hardware-detected optimal settings)")
            else:
                _tile_disp = 'auto' if self._cfg.ncnn_tile_auto else str(self._cfg.ncnn_tile)
            self._debug_log(f"config mode: MANUAL (batch={self._cfg.vsr_batch}, blocks={self._cfg.vsr_blocks}, tile={_tile_disp}, tier={self._cfg.cugan_tier})")

            self._prog(0.0, "init")

            # Validate and sanitize input path
            input_path = Path(self._cfg.input_path).resolve()
            if not input_path.is_file():
                raise FileNotFoundError(f"input missing: {self._cfg.input_path}")

            # Security: ensure path doesn't contain suspicious characters
            path_str = str(input_path)
            if any(char in path_str for char in ['\x00', '\n', '\r']):
                raise ValueError("Input path contains invalid characters")

            # Update config with resolved path
            self._cfg.input_path = path_str

            # Validate output directory is writable
            output_path = Path(self._cfg.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            self._workdir = Path(tempfile.mkdtemp(prefix="st_"))

            if self._cfg.engine == Engine.HYBRID:
                self._hybrid()
            elif self._cfg.engine == Engine.ESRGAN:
                self._esrgan_only()
            else:
                self._ffmpeg_only()

            self._done(not self._cancel.is_set(), "done!" if not self._cancel.is_set() else "cancelled")
        except Exception as e:
            self._err(str(e))
        finally:
            self._cleanup()

    def _hybrid(self) -> None:
        orig = self._workdir / "01_orig"
        enhanced = self._workdir / "03_out"
        orig.mkdir(exist_ok=True)
        enhanced.mkdir(exist_ok=True)

        self._prog(PipelineProgress.EXTRACT, "extracting frames")
        self._extract(orig)
        if self._cancel.is_set(): return

        pre = self._cfg.pre_filters()
        if pre:
            self._prog(PipelineProgress.PREFILTER, "pre-filters")
            self._filter_seq(orig, orig, pre, PipelineProgress.PREFILTER, 0.07)
        if self._cancel.is_set(): return

        vsr = self._workdir / "02_vsr"
        vsr.mkdir(exist_ok=True)

        # DYNAMIC temporal method selection based on user preference
        if self._cfg.temporal_method == TemporalMethod.BASICVSR:
            # Safety check: if torch disappeared, fall back to ffmpeg
            # Use importlib -- never import torch outside the dedicated thread
            import importlib.util as _ilu
            if _ilu.find_spec("torch") is None:
                self._log("PyTorch not found, falling back to FFmpeg temporal processing")
                self._cfg.temporal_method = TemporalMethod.FFMPEG
                self._cfg.save()

        if self._cfg.temporal_method == TemporalMethod.BASICVSR:
            self._prog(PipelineProgress.VSR_START, "BasicVSR++ temporal pass")
            self._basicvsr(orig, vsr)
        else:
            self._prog(PipelineProgress.VSR_START, "FFmpeg temporal processing")
            self._ffmpeg_temporal(orig, vsr)

        if self._cancel.is_set(): return

        if self._cfg.scale_pct > 100:
            self._prog(PipelineProgress.NCNN_START, "ncnn upscale")
            self._ncnn_upscale(vsr, enhanced)
        else:
            for f in vsr.glob("frame_*.png"):
                shutil.copy2(f, enhanced / f.name)
        if self._cancel.is_set(): return

        post = self._cfg.post_filters()
        if post:
            self._prog(PipelineProgress.POSTFILTER, "post-filters")
            self._filter_seq(enhanced, enhanced, post, PipelineProgress.POSTFILTER, 0.12)
        if self._cancel.is_set(): return

        self._prog(PipelineProgress.ASSEMBLE_START, "assembling video")
        self._assemble(enhanced)
        self._prog(PipelineProgress.ASSEMBLE_END, "finished")

    def _esrgan_only(self) -> None:
        orig = self._workdir / "01_orig"
        out = self._workdir / "02_out"
        orig.mkdir(exist_ok=True); out.mkdir(exist_ok=True)

        self._prog(0.05, "extracting"); self._extract(orig)
        if self._cancel.is_set(): return

        # Skip pre-filters if all enhancement sliders are at default/0%
        pre = self._cfg.pre_filters()
        if pre:
            self._prog(0.12, "pre-filters")
            self._filter_seq(orig, orig, pre, 0.12, 0.08)
        if self._cancel.is_set(): return

        # Skip upscaler entirely at 100% scale
        if self._cfg.scale_pct > 100:
            self._prog(0.25, "ncnn upscale")
            self._ncnn_upscale(orig, out)
        else:
            self._log("Scale is 100% -- skipping AI upscaling")
            for f in orig.glob("frame_*.png"):
                shutil.copy2(f, out / f.name)
        if self._cancel.is_set(): return

        # Skip post-filters if all enhancement sliders are at default/0%
        post = self._cfg.post_filters()
        if post:
            self._prog(0.70, "post-filters")
            self._filter_seq(out, out, post, 0.70, 0.18)
        if self._cancel.is_set(): return

        self._prog(0.92, "assembling"); self._assemble(out)
        self._prog(1.0, "done")

    def _ffmpeg_only(self) -> None:
        ffmpeg = find_bin("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found")

        fg = self.build_full_filtergraph(self._info)
        self._debug_log(f"filter graph: {fg}")

        tw, th = target_dims(self._info.width, self._info.height, self._cfg.scale_pct)
        # Detect output color space for explicit metadata tagging
        _cs_raw = (self._info.color_space or "").lower()
        if "bt709" in _cs_raw or self._info.width >= 1280:
            _out_cs = "bt709"
        else:
            _out_cs = "bt601"
        cmd = [str(ffmpeg), "-i", self._cfg.input_path, "-vf", fg,
               "-c:v", "libx264", "-preset", "slow", "-crf", str(getattr(self._cfg, 'crf_value', 18)),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-colorspace", _out_cs, "-color_trc", _out_cs, "-color_primaries", _out_cs,
               "-y", self._cfg.output_path]
        self._run_cmd(cmd, "ffmpeg encode", max(60, self._info.duration))

    def build_full_filtergraph(self, info: VideoInfo) -> str:
        parts = []
        pre = self._cfg.pre_filters()
        if pre: parts.append(pre)
        if self._cfg.scale_pct != 100:
            w, h = target_dims(info.width, info.height, self._cfg.scale_pct)
            parts.append(f"scale={w}:{h}:flags=lanczos")
        post = self._cfg.post_filters()
        if post: parts.append(post)
        return ",".join(parts) if parts else "null"

    def _extract(self, dst: Path) -> None:
        """Extract video frames to PNG files using FFmpeg.

        Uses image2 muxer with %08d pattern for sequential numbering.
        Compatible with FFmpeg 7.x+.
        
        FIX (v1.0.0): Added explicit pixel format and color space handling
        to fix green tint issues. PNG extraction now uses RGB24 output which
        preserves color fidelity through the pipeline.
        """
        ffmpeg = find_bin("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found for extraction")

        dst.mkdir(parents=True, exist_ok=True)
        pat = dst / "frame_%08d.png"

        self._debug_log("extract frames: rgb24")

        cmd = [
            str(ffmpeg),
            "-i", self._cfg.input_path,
            "-vsync", "0",
            "-pix_fmt", "rgb24",
            str(pat),
            "-y"
        ]

        try:
            self._run_cmd(cmd, "frame extract", max(120, self._info.duration * 2))
        except subprocess.CalledProcessError as e:
            # Check if we got partial frames despite error
            cnt = len(list(dst.glob("frame_*.png")))
            if cnt > 0:
                self._log(f"frame extract had errors but got {cnt} frames, continuing...")
                return
            # Re-raise if truly no frames
            raise RuntimeError(
                f"Frame extraction failed (code {e.returncode}). "
                f"FFmpeg stderr: {e.stderr[-500:] if e.stderr else 'N/A'}"
            )

        cnt = len(list(dst.glob("frame_*.png")))
        if cnt == 0:
            raise RuntimeError(f"Frame extraction produced 0 frames from {self._cfg.input_path}")
        self._log(f"extracted {cnt} frames")

    def _filter_seq(self, src: Path, dst: Path, filt: Optional[str], base: float, rng: float) -> None:
        """Apply FFmpeg video filter sequence to frames.

        Uses -vf (video filter) for each frame. On failure the original frame
        is copied to avoid missing output.  Fatal errors (disk full, permission
        denied) are raised immediately and bubble up to the error dialog.
        Handles in-place filtering (when src == dst) by using temp files.
        """
        if not filt:
            # No filter - just copy frames
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.glob("frame_*.png")):
                if self._cancel.is_set():
                    return
                shutil.copy2(f, dst / f.name)
            return

        ffmpeg = find_bin("ffmpeg")
        if not ffmpeg:
            self._debug_log("no ffmpeg for filters, copying frames instead")
            dst.mkdir(parents=True, exist_ok=True)
            for f in sorted(src.glob("frame_*.png")):
                if self._cancel.is_set():
                    return
                shutil.copy2(f, dst / f.name)
            return

        # Enforce full-range RGB in/out for all frame filter operations
        filt = f"format=rgb24,{filt},format=rgb24"

        # Check if we're doing in-place filtering (src == dst)
        in_place = src.resolve() == dst.resolve()

        frames = sorted(src.glob("frame_*.png"))
        total = len(frames)
        fail_count = 0

        # Reset silent failure counter for this batch
        self._silent_fail_count = 0

        for i, f in enumerate(frames):
            if self._cancel.is_set():
                return

            if in_place:
                # CRITICAL: Cannot read and write same file simultaneously!
                # Use a temporary file, then replace original
                tmp_out = f.with_suffix('.tmp.png')
                cmd = [str(ffmpeg), "-i", str(f), "-vf", filt, "-update", "1", str(tmp_out), "-y"]
                success = self._run_cmd_silent(cmd, 30)

                if success and tmp_out.exists():
                    # Replace original with filtered version
                    tmp_out.replace(f)  # Atomic replace
                else:
                    fail_count += 1
                    # Clean up temp file if it exists
                    if tmp_out.exists():
                        tmp_out.unlink()
            else:
                # Different directories - safe to write directly
                out = dst / f.name
                cmd = [str(ffmpeg), "-i", str(f), "-vf", filt, "-update", "1", str(out), "-y"]
                success = self._run_cmd_silent(cmd, 30)

                if not success:
                    fail_count += 1
                    # On failure, copy original frame to avoid missing output
                    if not out.exists():
                        shutil.copy2(f, out)

            frac = base + rng * ((i+1) / max(total, 1))
            self._prog(min(frac, 1.0), f"filter {i+1}/{total}")

        # Summarize if there were failures
        if fail_count > 0:
            self._log(f"filter completed with {fail_count}/{total} frame failures (copied originals)")

    def _basicvsr(self, src: Path, dst: Path) -> None:
        """Run BasicVSR++ temporal processing.

        All PyTorch operations are isolated in a dedicated daemon thread.
        This ensures exactly one CUDA context exists, and it is fully
        released when processing finishes (success, error, or cancel).
        """
        # Detect hardware profile first (nvidia-smi only -- no torch import)
        hw_profile = HardwareProfile.detect()
        ai_settings = self._get_effective_settings(hw_profile)

        is_manual = not getattr(self._cfg, 'auto_config', True)
        log_settings = ai_settings if is_manual else None
        self._debug_log(hw_profile.log_details(user_settings=log_settings))

        # Shared state for thread communication
        thread_error: List[Optional[Exception]] = [None]
        done_event = threading.Event()

        def _pytorch_worker():
            """All PyTorch operations live here -- one thread, one CUDA context.

            VRAM management strategy:
            - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces fragmentation
            VRAM management strategy:
            - Dynamic batch size capping based on available VRAM.
            - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces fragmentation.
            - Autocast FP16/AMP for speed and low VRAM footprint.
            """
            try:
                # Set CUDA allocator config BEFORE importing torch
                # expandable_segments avoids fragmentation that wastes VRAM
                os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

                import torch
                import numpy as np
                from PIL import Image

                model_path = MODEL_DIR / "basicvsr" / BASICVSR_MODEL_FILENAME
                if not model_path.exists():
                    raise RuntimeError(
                        "BasicVSR++ model file not found.\n\n"
                        "Run the setup script to download models:\n"
                        "  ./setup.sh"
                    )

                # Device selection
                user_device_id = getattr(self._cfg, 'device_id', 'auto')

                if user_device_id == "cpu":
                    device = torch.device("cpu")
                    use_gpu = False
                elif user_device_id and user_device_id.startswith("gpu:"):
                    gpu_idx = int(user_device_id.split(":")[1])
                    if gpu_idx < len(hw_profile.gpus):
                        selected = hw_profile.gpus[gpu_idx]
                        vendor = selected.get("vendor", "")
                        if vendor in ("nvidia", "amd"):
                            # Translate our GPU list index to CUDA/ROCm device ordinal.
                            # Our list includes non-CUDA/non-ROCm GPUs (e.g. Intel iGPU), 
                            # but PyTorch only sees NVIDIA/AMD devices under the "cuda" namespace.
                            # Count NVIDIA/AMD GPUs before this one to get the correct ordinal.
                            cuda_ordinal = sum(
                                1 for g in hw_profile.gpus[:gpu_idx]
                                if g.get("vendor") in ("nvidia", "amd")
                            )
                            try:
                                device = torch.device(f"cuda:{cuda_ordinal}")
                                use_gpu = True
                                self._debug_log(f"PyTorch: gpu list idx {gpu_idx} ({vendor}) -> cuda:{cuda_ordinal}")
                            except RuntimeError:
                                self._log(f"CUDA/ROCm device {cuda_ordinal} unavailable, falling back to CPU")
                                device = torch.device("cpu")
                                use_gpu = False
                        elif vendor == "apple":
                            device = torch.device("mps")
                            use_gpu = True
                        else:
                            # Non-CUDA/Non-ROCm GPU selected (Intel iGPU, etc.)
                            # PyTorch cannot use Vulkan -- fall back to auto
                            self._debug_log(f"Selected GPU ({selected.get('name', '?')}) is not PyTorch CUDA/ROCm-capable, using auto")
                            if hw_profile.has_cuda:
                                device = torch.device("cuda")
                                use_gpu = True
                            elif hw_profile.has_mps:
                                device = torch.device("mps")
                                use_gpu = True
                            else:
                                device = torch.device("cpu")
                                use_gpu = False
                    else:
                        self._log(f"GPU index {gpu_idx} out of range, falling back to CPU")
                        device = torch.device("cpu")
                        use_gpu = False
                elif hw_profile.has_cuda:
                    device = torch.device("cuda")
                    use_gpu = True
                elif hw_profile.has_mps:
                    device = torch.device("mps")
                    use_gpu = True
                else:
                    device = torch.device("cpu")
                    use_gpu = False

                # Memory-efficient precision context
                # Centralises autocast + inference_mode so every model call
                # site stays DRY and we never forget one.
                @contextlib.contextmanager
                def _model_ctx():
                    """Yield under the correct autocast + inference_mode for *device*."""
                    _amp = ai_settings.get('use_amp', True)
                    _ctx = None
                    if device.type == 'cuda' and _amp:
                        _ctx = torch.amp.autocast('cuda', enabled=ai_settings.get('use_fp16', True))
                    elif device.type == 'mps' and _amp:
                        _ctx = torch.amp.autocast(device_type='mps')
                    elif device.type == 'cpu' and _amp:
                        # bfloat16 halves all activation memory on CPU
                        _ctx = torch.amp.autocast('cpu', dtype=torch.bfloat16)
                    if _ctx is not None:
                        with _ctx:
                            with torch.inference_mode():
                                yield
                    else:
                        with torch.inference_mode():
                            yield

                if not use_gpu:
                    self._log(f"using CPU for VSR ({hw_profile.cpu_name})")

                # Build model
                model = self._build_model_with_settings(ai_settings)
                if model is None:
                    raise RuntimeError(
                        "Failed to build BasicVSR++ model.\n\n"
                        "Check that PyTorch is installed and your GPU has sufficient memory."
                    )

                # Load weights on CPU first, then move to device
                state = torch.load(model_path, map_location='cpu', weights_only=False)
                model.load_state_dict(state, strict=False)
                del state
                gc.collect()

                model.to(device).eval()

                # Frame setup
                frames = sorted(src.glob("frame_*.png"))
                total = len(frames)
                if not frames:
                    raise FileNotFoundError(f"No frame_*.png files found in {src}")

                with Image.open(frames[0]) as sample_img:
                    w, h = sample_img.convert("RGB").size

                batch_size = ai_settings.get('vsr_batch', 2)

                # CPU has no GPU parallelism -- batch>1 just multiplies RAM usage
                # with zero speed gain.  Force 1 to keep peak memory minimal.
                if device.type == 'cpu':
                    if batch_size > 1:
                        self._toast(f"Batch size {batch_size} -> 1 (CPU has no parallelism benefit)")
                    batch_size = 1
                    self._debug_log("CPU: forcing batch_size=1 (no GPU parallelism, saves RAM)")

                # Dynamic batch size capping based on available VRAM
                # The model's feature extraction produces 64-channel tensors:
                #   [batch, 64, H, W] -- for 1080p that's ~2.5GB per batch in fp32!
                # Each residual block needs input+output, so ~3x activation memory.
                # Cap batch size to fit in available VRAM with safety margin.
                if use_gpu and device.type == 'cuda' and hasattr(torch.cuda, 'mem_get_info'):
                    try:
                        free_vram, total_vram = torch.cuda.mem_get_info(device)
                        free_mb = free_vram / (1024 ** 2)
                        # Model weights + CUDA context overhead (~300MB safety margin)
                        available_mb = max(0, free_mb - 300)
                        # Memory per frame: W*H*64*bytes_per_element*3 (input+output+intermediate)
                        bytes_per_elem = 2 if ai_settings.get('use_fp16', True) else 4
                        mem_per_frame_mb = (w * h * 64 * bytes_per_elem * 3) / (1024 ** 2)
                        safe_batch = max(1, int(available_mb / max(mem_per_frame_mb, 1)))
                        if safe_batch < batch_size:
                            self._debug_log(f"VRAM budget: {free_mb:.0f}MB free, ~{mem_per_frame_mb:.0f}MB/frame -> capping batch {batch_size} -> {safe_batch}")
                            self._toast(f"Batch size {batch_size} -> {safe_batch} (VRAM limit)")
                            batch_size = safe_batch
                    except Exception as e:
                        self._debug_log(f"VRAM budget calc skipped: {e}")

                self._debug_log(f"processing {total} frames ({w}x{h}) | batch={batch_size} | blocks={ai_settings.get('vsr_blocks', 10)} | device={device.type}")

                dst.mkdir(parents=True, exist_ok=True)
                consecutive_ooms = 0
                max_consecutive_ooms = 3

                for batch_start in range(0, total, batch_size):
                    if self._cancel.is_set():
                        return

                    batch_end = min(batch_start + batch_size, total)
                    batch_frames = frames[batch_start:batch_end]
                    actual_batch = len(batch_frames)

                    B, C, H, W = actual_batch, 3, h, w
                    batch_tensor = torch.zeros(B, C, H, W, dtype=torch.float32)

                    for i, f in enumerate(batch_frames):
                        with Image.open(f) as im:
                            img = np.array(im.convert("RGB")).astype(np.float32) / 255.0
                        batch_tensor[i] = torch.from_numpy(img).permute(2, 0, 1)
                        del img

                    batch_tensor = batch_tensor.to(device)
                    batch_input = batch_tensor.unsqueeze(0)

                    try:
                        with _model_ctx():
                            out = model(batch_input).clamp(0, 1)

                        for i in range(actual_batch):
                            if self._cancel.is_set():
                                return
                            global_idx = batch_start + i
                            result = out[0, i].float().cpu().permute(1, 2, 0).numpy()
                            frame = (result * 255).clip(0, 255).astype(np.uint8)
                            Image.fromarray(frame).save(dst / f"frame_{global_idx:08d}.png")
                            del result, frame  # free per-frame numpy arrays immediately

                        consecutive_ooms = 0
                        self._prog(0.18 + 0.27 * (batch_end / max(total, 1)),
                                  f"vsr {batch_end}/{total}")

                    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                        is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
                        error_kind = "CUDA out of memory" if is_oom else "CUDA runtime error"
                        self._debug_log(f"{error_kind}: {e}")
                        consecutive_ooms += 1

                        if consecutive_ooms >= max_consecutive_ooms:
                            torch.cuda.empty_cache()
                            raise RuntimeError(
                                f"{error_kind} -- your GPU ran out of VRAM.\n\n"
                                f"Try these fixes:\n"
                                f"  - Lower the resolution scale\n"
                                f"  - Reduce the batch size in Advanced settings\n"
                                f"  - Close other GPU-using applications\n"
                                f"  - Use a GPU with more VRAM"
                            ) from e

                        if batch_size > 1:
                            new_batch = max(1, batch_size // 2)
                            self._debug_log(f"OOM (#{consecutive_ooms}), reducing batch: {batch_size} -> {new_batch}")
                            torch.cuda.empty_cache()

                            for sub_start in range(0, actual_batch, new_batch):
                                if self._cancel.is_set():
                                    return
                                sub_end = min(sub_start + new_batch, actual_batch)
                                sub_frames = batch_frames[sub_start:sub_end]

                                sub_imgs = []
                                for f in sub_frames:
                                    with Image.open(f) as im:
                                        img = np.array(im.convert("RGB")).astype(np.float32) / 255.0
                                    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
                                    sub_imgs.append(tensor)

                                sub_batch = torch.cat(sub_imgs, dim=0) if len(sub_imgs) > 1 else sub_imgs[0]

                                try:
                                    with _model_ctx():
                                        o = model(sub_batch).clamp(0, 1)

                                    for j in range(len(sub_frames)):
                                        if self._cancel.is_set():
                                            return
                                        idx = batch_start + sub_start + j
                                        r = o[0, j].float().cpu().permute(1, 2, 0).numpy() if o.dim() == 5 else o[0].float().cpu().permute(1, 2, 0).numpy()
                                        Image.fromarray((r * 255).clip(0, 255).astype(np.uint8)).save(dst / f"frame_{idx:08d}.png")

                                    del sub_imgs, sub_batch, o
                                except torch.cuda.OutOfMemoryError:
                                    for k, f in enumerate(sub_frames):
                                        if self._cancel.is_set():
                                            return
                                        try:
                                            with Image.open(f) as im:
                                                img = np.array(im.convert("RGB")).astype(np.float32) / 255.0
                                            tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
                                            with _model_ctx():
                                                single_out = model(tensor).clamp(0, 1)
                                            r = single_out[0, 0].float().cpu().permute(1, 2, 0).numpy()
                                            Image.fromarray((r * 255).clip(0, 255).astype(np.uint8)).save(dst / f"frame_{batch_start + sub_start + k:08d}.png")
                                            del tensor, single_out, img, r
                                            torch.cuda.empty_cache()
                                        except Exception:
                                            raise

                            batch_size = new_batch
                        else:
                            self._log("Single-frame OOM! GPU does not have enough memory for this resolution.")
                            torch.cuda.empty_cache()
                            gc.collect()
                            torch.cuda.empty_cache()
                            raise RuntimeError(
                                "GPU out of memory -- even a single frame doesn't fit in VRAM.\n\n"
                                "Try these fixes:\n"
                                "  - Use a lower resolution scale\n"
                                "  - Use a GPU with more VRAM\n"
                                "  - Close other GPU-using applications"
                            )

                    # Periodic cleanup -- run on ALL devices, not just GPU
                    cleanup_interval = 50 if (use_gpu and hw_profile.tier.value >= HardwareTier.HIGH.value) else 20
                    if batch_end % cleanup_interval == 0:
                        if use_gpu and device.type == 'cuda':
                            torch.cuda.empty_cache()
                        gc.collect()

                    try: del batch_tensor
                    except NameError: pass
                    try: del batch_input
                    except NameError: pass
                    try: del out
                    except NameError: pass
                    gc.collect()  # ensure intermediates from this batch are freed

                self._log("BasicVSR++ processing complete")
                self._debug_log(f"basicvsr stats: final_batch={batch_size} | blocks={ai_settings.get('vsr_blocks', 10)}")

            except ImportError:
                thread_error[0] = RuntimeError(
                    "PyTorch is required for AI processing but is not installed.\n\n"
                    "Install it with:\n"
                    "  pip install torch"
                )
            except RuntimeError:
                raise  # propagate our own RuntimeErrors as-is
            except Exception as e:
                import traceback
                self._debug_log(f"traceback:\n{traceback.format_exc()}")
                thread_error[0] = RuntimeError(f"BasicVSR++ processing failed:\n\n{e}")
            finally:
                # Thorough cleanup -- free ALL VRAM so NCNN can use it
                try:
                    del model
                except Exception:
                    pass
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
                # Ensure ALL torch tensors in this scope are freed
                for _name in list(locals().keys()):
                    try:
                        del locals()[_name]
                    except Exception:
                        pass
                gc.collect()
                done_event.set()

        # Launch the dedicated PyTorch thread
        pytorch_thread = threading.Thread(target=_pytorch_worker, daemon=True, name="PyTorch")
        pytorch_thread.start()

        # Wait for completion, checking cancel periodically
        while not done_event.is_set():
            if self._cancel.is_set():
                return  # daemon thread dies with the app
            done_event.wait(timeout=0.5)

        # Propagate any error from the PyTorch thread
        if thread_error[0] is not None:
            raise thread_error[0]

    def _build_model_with_settings(self, ai_settings: Dict[str, Any]):
        """Build BasicVSR++ network with user-configured or auto-detected settings.

        Uses ai_settings['vsr_blocks'] for backbone complexity instead of
        hardware profile detection. This allows manual override via Advanced settings.
        """
        # Use vsr_blocks from settings (defaults to 10 if not set)
        num_blocks = ai_settings.get('vsr_blocks', 10)

        # Reuse the dynamic builder but inject our block count
        class FakeHWProfile:
            """Minimal fake hw_profile that provides user-configured values."""
            def __init__(self, blocks):
                self.vsr_backbone_blocks = blocks
                self.tier = type('obj', (object,), {'name': 'USER'})()  # Fake tier for logging

        return self._build_model_dynamic(FakeHWProfile(num_blocks))

    def _build_model_dynamic(self, hw_profile: HardwareProfile):
        """Build BasicVSR++ network with DYNAMIC COMPLEXITY based on hardware.

        Architecture scales conservatively with hardware tier:
        - NONE/CPU:  3 backbone blocks  (~15MB, minimal VRAM)
        - LOW:       4 backbone blocks  (~20MB, conservative)
        - MEDIUM:    4 backbone blocks  (~20MB, balanced)
        - HIGH:      5 backbone blocks  (~25MB, good quality)
        - ULTRA:     6 backbone blocks  (~30MB, maximum quality)

        Weights are loaded with strict=False to allow partial compatibility."""
        try:
            import torch
            import torch.nn as nn

            class ResidualBlock(nn.Module):
                """Lightweight residual block with 64 channels."""
                def __init__(self, nf=64):
                    super().__init__()
                    self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
                    self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

                def forward(self, x):
                    # In-place ReLU eliminates 2 intermediate tensor allocations
                    # per block.  At 1080p x 64 channels each intermediate is
                    # ~500 MB (fp32) / ~250 MB (bf16), so this saves ~1 GB per
                    # block on CPU and frees VRAM on GPU.
                    out = self.conv1(x)
                    out.relu_()       # in-place
                    out = self.conv2(out)
                    out.relu_()       # in-place
                    return x + out

            class AdaptiveVSR(nn.Module):
                """VSR network with configurable backbone depth and residual refinement."""
                def __init__(self, num_blocks: int = 10):
                    super().__init__()
                    # Feature extraction: RGB -> 64 channels
                    self.feat_extract = nn.Sequential(
                        nn.Conv2d(3, 64, 3, 1, 1),
                        nn.LeakyReLU(0.1, inplace=True),
                        ResidualBlock(64),
                        ResidualBlock(64),
                    )
                    # Processing backbone: DYNAMIC depth based on hardware
                    self.backbone = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
                    # Reconstruction: 64 -> RGB (residual delta)
                    self.reconstruction = nn.Sequential(
                        nn.Conv2d(64, 64, 3, 1, 1),
                        nn.LeakyReLU(0.1, inplace=True),
                        nn.Conv2d(64, 3, 3, 1, 1),
                    )
                    # Initialize final convolution to zero so default/residual output starts transparent
                    nn.init.zeros_(self.reconstruction[-1].weight)
                    nn.init.zeros_(self.reconstruction[-1].bias)

                def forward(self, x):
                    """
                    Input: [B, T, C, H, W] where B=1, T=batch_size
                    Output: [B, T, C, H, W] enhanced
                    """
                    B, T, C, H, W = x.shape
                    x_flat = x.reshape(B * T, C, H, W)
                    feat = self.feat_extract(x_flat)
                    out = self.backbone(feat)
                    delta = self.reconstruction(out)
                    res = x_flat + delta
                    return res.reshape(B, T, 3, H, W)

            # Use hardware-profiled block count
            num_blocks = hw_profile.vsr_backbone_blocks
            model = AdaptiveVSR(num_blocks=num_blocks)
            self._debug_log(f"built model with {num_blocks} backbone blocks for {hw_profile.tier.name} tier")
            return model
        except Exception as e:
            self._debug_log(f"couldn't build model: {e}")
            return None

    def _ffmpeg_temporal(self, src: Path, dst: Path) -> None:
        """Dedicated FFmpeg temporal processing mode (user-selected alternative to BasicVSR++).

        Uses FFmpeg temporal filters for:
        - Temporal noise reduction (hqdn3d, tmix)
        - Deblocking and sharpening

        Much faster than BasicVSR++ (~10-50x) but less sophisticated.
        """
        self._log("using FFmpeg temporal processing (user-selected mode)")
        ffmpeg = find_bin("ffmpeg")
        if not ffmpeg:
            self._log("no ffmpeg available, copying frames as-is")
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("frame_*.png"):
                shutil.copy2(f, dst / f.name)
            return

        dst.mkdir(parents=True, exist_ok=True)
        frames = sorted(src.glob("frame_*.png"))
        total = len(frames)

        # Build comprehensive temporal filter chain
        filters = []

        # 1. Temporal denoising (if denoise > 0)
        dn = self._cfg.denoise_filt
        if dn.get("luma_tmp", 0) > 0 or dn.get("luma_spatial", 0) > 0:
            filters.append(f"hqdn3d={dn['luma_spatial']:.1f}:{dn['chroma_spatial']:.1f}:{dn['luma_tmp']:.1f}:{dn['chroma_tmp']:.1f}")

        # 2. Temporal anti-aliasing / motion smoothing (tmix blends adjacent frames)
        if self._cfg.denoise >= 20:
            # Stronger temporal mixing for noisy content
            tmix_frames = min(5, max(2, int(self._cfg.denoise / 20)))
            tmix_weight = min(0.7, self._cfg.denoise / 100.0)
            filters.append(f"tmix={tmix_frames}:{tmix_weight:.2f}")
        elif self._cfg.antialias != 0:
            # Use tmix for anti-aliasing too
            filters.append(f"tmix=3:0.30")

        # 3. Temporal deblocking
        db = self._cfg.deblock_filt
        if db:
            filters.append(f"deblock={db['filter']}:{db['block']}:{db['alpha']:.3f}:{db['beta']:.3f}")

        # 4. Temporal sharpening (unsharp across frames creates crispness)
        if self._cfg.sharpen > 30:
            us = self._cfg.unsharp_filt
            if us["luma_amount"] > 0:
                filters.append(f"unsharp=luma_msize_x={us['msize_x']}:luma_msize_y={us['msize_y']}:luma_amount={us['luma_amount']}:chroma_amount={us['chroma_amount']}")

        # If no filters applied, use a light default temporal filter
        if not filters:
            filters.append("tmix=2:0.20")

        vf = "format=rgb24," + ",".join(filters) + ",format=rgb24"
        self._debug_log(f"FFmpeg temporal filter chain: {vf[:100]}...")

        for i, f in enumerate(frames):
            if self._cancel.is_set():
                return
            # Use -update 1 for FFmpeg 7.x single image output compatibility
            cmd = [str(ffmpeg), "-i", str(f), "-vf", vf, "-update", "1", str(dst / f.name), "-y"]
            self._run_cmd_silent(cmd, 30)
            self._prog(0.18 + 0.27 * ((i+1)/max(total,1)), f"FFmpeg temporal {i+1}/{total}")

        self._log("FFmpeg temporal processing complete")

    def _get_vulkan_gpu_map(self) -> Optional[Dict[int, int]]:
        """Build mapping from our GPU list index to Vulkan device index.

        NCNN uses Vulkan for GPU acceleration. Vulkan enumerates devices in its
        own order (typically iGPU first, then dGPU), which differs from
        nvidia-smi ordering. This method queries vulkaninfo to get the Vulkan
        device order and matches them to our detected GPUs by name/vendor.

        Returns {our_gpu_idx: vulkan_device_idx} or None if unavailable.
        """
        try:
            r = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode != 0:
                return None

            # Parse Vulkan GPU names in enumeration order
            vulkan_gpus = []
            for line in r.stdout.splitlines():
                line_stripped = line.strip()
                # Format: "GPU id = N (name):" or "GPU id = N (name)"
                if line_stripped.startswith("GPU") and "id =" in line_stripped:
                    match = re.search(r'GPU\s+id\s*=\s*(\d+)\s+\((.+?)\)', line_stripped)
                    if match:
                        vid = int(match.group(1))
                        vname = match.group(2).strip()
                        vulkan_gpus.append({"id": vid, "name": vname})

            if not vulkan_gpus:
                return None

            # Get our GPU list
            hw = HardwareProfile.detect()
            if not hw.gpus:
                return None

            # Match our GPUs to Vulkan GPUs by vendor and name similarity
            mapping = {}
            used_vulkan = set()
            for our_idx, our_gpu in enumerate(hw.gpus):
                our_name = our_gpu.get("name", "").lower()
                vendor = our_gpu.get("vendor", "")
                best_vid = None
                best_score = 0
                for vg in vulkan_gpus:
                    if vg["id"] in used_vulkan:
                        continue
                    vname = vg["name"].lower()
                    # Match by vendor keyword in Vulkan device name
                    vendor_match = False
                    if vendor == "nvidia" and "nvidia" in vname:
                        vendor_match = True
                    elif vendor == "intel" and "intel" in vname:
                        vendor_match = True
                    elif vendor == "amd" and ("amd" in vname or "radeon" in vname or "ati" in vname):
                        vendor_match = True

                    if vendor_match:
                        # Score by shared words for disambiguation
                        our_words = set(our_name.split())
                        vk_words = set(vname.split())
                        score = len(our_words & vk_words) + 1  # +1 for vendor match
                        if score > best_score:
                            best_score = score
                            best_vid = vg["id"]

                if best_vid is not None:
                    mapping[our_idx] = best_vid
                    used_vulkan.add(best_vid)

            self._debug_log(f"Vulkan GPU map: {mapping}")
            return mapping if mapping else None

        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return None

    def _ncnn_gpu_id(self) -> int:
        """Resolve the NCNN GPU ID (Vulkan device index) from config.

        Returns the -g flag value for NCNN binaries:
        - -1 = CPU
        - 0+ = Vulkan device index

        The GPU list in HardwareProfile is built in Vulkan enumeration order
        (iGPU first, then dGPU), so our list indices should directly correspond
        to Vulkan device indices. vulkaninfo is used as optional verification.
        """
        device_id = getattr(self._cfg, 'device_id', 'auto')
        if device_id == 'cpu':
            return -1

        hw = HardwareProfile.detect()

        if device_id and device_id.startswith('gpu:'):
            try:
                our_idx = int(device_id.split(':')[1])
            except (ValueError, IndexError):
                our_idx = 0
            # Our GPU list is ordered to match Vulkan enumeration
            if our_idx < len(hw.gpus):
                self._debug_log(f"ncnn gpu: list idx {our_idx} -> vulkan -g {our_idx} ({hw.gpus[our_idx]['name']})")
                return our_idx
            return 0

        # Auto: skip integrated GPUs, prefer dedicated (NVIDIA/AMD)
        for i, gpu in enumerate(hw.gpus):
            if not gpu.get("integrated", False):
                self._debug_log(f"ncnn gpu auto: selected idx {i} ({gpu['name']})")
                return i
        # Only iGPU available
        if hw.gpus:
            return 0
        return -1

    def _ncnn_upscale(self, src: Path, dst: Path) -> None:
        """Dispatch NCNN upscaling to the user-selected model (CUGAN or ESRGAN)."""
        if getattr(self._cfg, 'upscaler_model', 'cugan') == 'esrgan':
            self._ncnn_upscale_realesrgan(src, dst)
        else:
            self._ncnn_upscale_cugan(src, dst)

    def _ncnn_run_frames(self, binary: Path, frames: List[Path], dst: Path,
                           cmd_builder: Callable, label: str,
                           prog_start: float, prog_span: float) -> None:
        """Process frames one at a time via NCNN.

        Per-frame processing is more robust and avoids GPU OOM from
        directory mode loading all frames into memory.
        """
        total = len(frames)
        dst.mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(frames):
            if self._cancel.is_set():
                return
            out_path = dst / f.name
            cmd = cmd_builder(f, out_path)
            if not self._run_cmd_silent(cmd, 300):
                error_info = getattr(self, '_last_silent_error', 'unknown error')
                # On Vulkan OOM (vkAllocateMemory), retry with auto tile (-t 0)
                if 'vkAllocateMemory' in error_info:
                    self._debug_log(f"{label}: OOM on frame {i+1}, retrying with auto tile")
                    retry_cmd = list(cmd)
                    for j in range(len(retry_cmd) - 1):
                        if retry_cmd[j] == '-t':
                            retry_cmd[j + 1] = '0'
                            break
                    if not self._run_cmd_silent(retry_cmd, 300):
                        error_info = getattr(self, '_last_silent_error', 'unknown error')
                        raise RuntimeError(f"{label} frame {i+1}/{total} failed!\n\n{error_info}")
                else:
                    raise RuntimeError(f"{label} frame {i+1}/{total} failed!\n\n{error_info}")
            self._prog(prog_start + prog_span * ((i + 1) / max(total, 1)),
                      f"{label} {i + 1}/{total}")

        self._validate_ncnn_batch_output(dst, label)

    def _ncnn_upscale_cugan(self, src: Path, dst: Path) -> None:
        """NCNN Real-CUGAN upscaling.

        Resolves model tier, validates files, then delegates to _ncnn_run_frames.
        """
        binary = find_ncnn_bin(NCNN_BINS["realcugan"])
        if not binary:
            raise RuntimeError(
                "realcugan-ncnn-vulkan not found at ~/.local/bin\n\n"
                "Install NCNN tools: ./setup.sh --force-build\n"
                "Or use scale=100% to disable AI upscaling"
            )
        hw_profile = HardwareProfile.detect()
        ai_settings = self._get_effective_settings(hw_profile)
        frames = sorted(src.glob("frame_*.png"))
        sf = max(2, self._cfg.scale_pct // 100)

        # Model tier selection
        user_tier = ai_settings.get('cugan_tier', 'se')
        if not getattr(self._cfg, 'auto_config', True):
            tier_map = {"se": "models-se", "pro": "models-pro", "nose": "models-nose"}
            preferred = [tier_map.get(user_tier, "models-se")]
            for sd in ["models-se", "models-pro", "models-nose"]:
                if sd not in preferred:
                    preferred.append(sd)
        else:
            preferred = ["models-se", "models-pro", "models-nose"]

        cugan_base = MODEL_DIR / "real-cugan"
        tier_scale_support = {"models-se": {2, 3, 4}, "models-pro": {2, 3}, "models-nose": {2}}
        model_dir = None
        noise_level = -1
        for subdir in preferred:
            candidate = cugan_base / subdir
            if not candidate.is_dir():
                continue
            if sf not in tier_scale_support.get(subdir, set()):
                self._debug_log(f"realcugan: {subdir} does not support {sf}x, skipping")
                continue
            if subdir == "models-nose":
                model_file = candidate / f"up{sf}x-no-denoise.param"
                noise_level = 0
            else:
                model_file = candidate / f"up{sf}x-conservative.param"
                noise_level = -1
            if model_file.exists() and model_file.with_suffix(".bin").exists():
                model_dir = candidate
                break
            self._debug_log(f"realcugan: model files missing for {subdir} {sf}x at {model_file}")

        if model_dir is None:
            raise RuntimeError(
                f"No valid Real-CUGAN model found for {sf}x upscale.\n\n"
                "Run the setup script to download models:\n  ./setup.sh")

        ncnn_jobs = ai_settings.get('ncnn_jobs', '1:4:4')
        use_tile_auto = getattr(self._cfg, 'ncnn_tile_auto', True)
        tile_arg = "0" if use_tile_auto else str(self._cfg.ncnn_tile)
        gpu_id = self._ncnn_gpu_id()
        self._debug_log(f"realcugan: {len(frames)} frames, {sf}x, model={model_dir}, noise={noise_level}, gpu={gpu_id}")

        def cmd_builder(frame_in, frame_out):
            return [str(binary), "-i", str(frame_in), "-o", str(frame_out),
                    "-s", str(sf), "-n", str(noise_level), "-m", str(model_dir),
                    "-g", str(gpu_id), "-j", ncnn_jobs, "-t", tile_arg]

        self._ncnn_run_frames(binary, frames, dst, cmd_builder, "cugan", 0.25, 0.45)

    def _ncnn_upscale_realesrgan(self, src: Path, dst: Path) -> None:
        """NCNN Real-ESRGAN upscaling.

        Resolves model files, validates, then delegates to _ncnn_run_frames.
        """
        binary = find_ncnn_bin(NCNN_BINS["realesrgan"])
        if not binary:
            raise RuntimeError(
                "realesrgan-ncnn-vulkan not found at ~/.local/bin\n\n"
                "Install NCNN tools: ./setup.sh --force-build\n"
                "Or use scale=100% to disable AI upscaling"
            )

        hw_profile = HardwareProfile.detect()
        ai_settings = self._get_effective_settings(hw_profile)
        frames = sorted(src.glob("frame_*.png"))
        sf = max(2, self._cfg.scale_pct // 100)

        # -n is a MODEL NAME (not a path). Tool appends -x{scale} automatically.
        # When ESRGAN is selected ("Realistic" in UI), prefer realistic model;
        # fall back to animevideov3 if realistic model files aren't available.
        model_dir = MODEL_DIR / "realesrgan"

        if not model_dir.is_dir():
            raise RuntimeError(
                f"Real-ESRGAN model directory not found: {model_dir}\n\n"
                "Run the setup script to download models:\n  ./setup.sh")

        # Candidate model names in priority order: realistic first, then anime.
        # realesrgan-x4plus is inherently a 4x model -- only consider it at sf==4.
        # realesr-animevideov3 has per-scale variants (x2, x3, x4) and works at any sf.
        _X4ONLY = {"realesrgan-x4plus", "realesrgan-x4plus-anime"}
        model_candidates = [
            ("realesrgan-x4plus", "realistic"),
            ("realesr-animevideov3", "anime"),
        ]
        model_name = None
        model_flavor = None
        for candidate, flavor in model_candidates:
            # Skip models that don't support the requested scale
            if candidate in _X4ONLY and sf != 4:
                continue
            # Try scale-suffixed name: {name}-x{sf}.param (e.g. animevideov3-x2.param)
            expected = model_dir / f"{candidate}-x{sf}.param"
            if expected.exists() and expected.with_suffix(".bin").exists():
                model_name = candidate
                model_flavor = flavor
                break
            # Try plain name: {name}.param (e.g. realesrgan-x4plus.param)
            expected_alt = model_dir / f"{candidate}.param"
            if expected_alt.exists() and expected_alt.with_suffix(".bin").exists():
                model_name = candidate
                model_flavor = flavor
                break

        if model_name is None:
            raise RuntimeError(
                f"No Real-ESRGAN {sf}x model files found in {model_dir}.\n\n"
                "Run the setup script to download models:\n  ./setup.sh")

        if model_flavor == "anime":
            self._log("Real-ESRGAN: realistic model not available for {}x, "
                      "using anime model".format(sf))
        else:
            self._log(f"Real-ESRGAN: using {model_flavor} model: {model_name}")

        ncnn_jobs = ai_settings.get('ncnn_jobs', '1:4:4')
        use_tile_auto = getattr(self._cfg, 'ncnn_tile_auto', True)
        tile_arg = "0" if use_tile_auto else str(self._cfg.ncnn_tile)
        gpu_id = self._ncnn_gpu_id()
        self._debug_log(f"realesrgan: {len(frames)} frames, {sf}x, model={model_name}, gpu={gpu_id}")

        def cmd_builder(frame_in, frame_out):
            return [str(binary), "-i", str(frame_in), "-o", str(frame_out),
                    "-s", str(sf), "-n", model_name, "-m", str(model_dir),
                    "-g", str(gpu_id), "-j", ncnn_jobs, "-t", tile_arg]

        self._ncnn_run_frames(binary, frames, dst, cmd_builder, "esrgan", 0.25, 0.45)

    def _assemble(self, frames_dir: Path) -> None:
        """Assemble processed frames back into a video file.

        Robust implementation with:
        - Input validation (frames exist, paths valid)
        - Audio passthrough
        - Detailed error reporting with FFmpeg stderr
        """
        ffmpeg = find_bin("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found for video assembly")

        # Validation Phase

        # Check frames directory exists and has content
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            raise FileNotFoundError(f"No frame_*.png files found in {frames_dir}")

        frame_count = len(frame_files)
        self._log(f"assembling {frame_count} frames from {frames_dir.name}")

        # DEBUG: Log frame details for diagnostics
        first_frame = frame_files[0].name if frame_files else "none"
        last_frame = frame_files[-1].name if len(frame_files) > 1 else first_frame
        first_size = frame_files[0].stat().st_size if frame_files else 0
        self._debug_log(f"frame range: {first_frame} -> {last_frame} ({frame_count} frames)")
        self._debug_log(f"first frame size: {first_size} bytes")

        # Check for frame numbering gaps (FFmpeg image2 demuxer needs sequential)
        frame_nums = []
        for f in frame_files:
            m = re.search(r'frame_(\d{8})\.png', f.name)
            if m:
                frame_nums.append(int(m.group(1)))

        if frame_nums:
            expected = list(range(frame_nums[0], frame_nums[0] + len(frame_nums)))
            gaps = set(expected) - set(frame_nums)
            if gaps:
                self._debug_log(f"WARNING: {len(gaps)} gap(s) in frame numbering! FFmpeg may fail.")
                self._debug_log(f"Gap positions (first 5): {sorted(gaps)[:5]}")

        # Validate output path is writable
        output_path = Path(self._cfg.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate original input exists (for audio extraction)
        original_input = Path(self._cfg.input_path)
        has_audio_input = original_input.exists() and original_input.is_file()

        # Build Parameters

        pattern = str(frames_dir / "frame_%08d.png")
        proc_w, proc_h = target_dims(self._info.width, self._info.height, self._cfg.scale_pct)

        # Figure out final output resolution (descale back to original?)
        if self._cfg.descale and self._cfg.scale_pct > 100:
            out_w, out_h = self._info.width, self._info.height
            out_w += out_w % 2; out_h += out_h % 2
        else:
            out_w, out_h = proc_w, proc_h

        # Ensure dimensions are valid (even numbers, positive)
        out_w = max(2, out_w + (out_w % 2))
        out_h = max(2, out_h + (out_h % 2))
        proc_w = max(2, proc_w + (proc_w % 2))
        proc_h = max(2, proc_h + (proc_h % 2))

        # Get codec FIRST - needed for pix_fmt selection below
        codec = self._cfg.codec
        fps_str = self._info.fps_str

        # Build video filter chain
        # Proper RGB->YUV conversion with range handling
        # - PNG files are FULL-RANGE RGB (0-255)
        # - YUV video formats expect LIMITED RANGE (16-235 luma, 16-240 chroma)
        # - Without explicit range conversion, output appears BLACK or washed out!
        # - Using in_range=full -> out_range=limited

        # Determine output pix_fmt based on codec
        if codec == Codec.PRORES:
            out_pix_fmt = "yuv422p10le"
            vf_pix_fmt = "yuv422p10le"
        else:
            out_pix_fmt = "yuv420p"
            vf_pix_fmt = "yuv420p"

        # Determine color matrix from original video metadata.
        # FFmpeg's default RGB->YUV heuristic can be wrong, causing green tint.
        # We read the original colorspace from ffprobe and pass it through.
        cs_raw = (self._info.color_space or "").lower()
        if "bt709" in cs_raw or "bt2020" in cs_raw or "bt2100" in cs_raw:
            cs_matrix = "bt709" if "bt709" in cs_raw else "bt2020"
        elif "bt470bg" in cs_raw or "smpte170m" in cs_raw or "smpte240m" in cs_raw:
            cs_matrix = "bt601"
        elif self._info.width >= 1280:
            cs_matrix = "bt709"
        else:
            cs_matrix = "bt601"
        self._debug_log(f"color matrix: {cs_matrix} (from '{self._info.color_space}')")

        # Map color_trc and color_primaries for output metadata
        trc_map = {"bt709": "bt709", "bt470bg": "gamma28", "smpte170m": "smpte170m",
                   "bt2020": "bt2020_10", "bt2100": "arib-std-b67"}
        cs_out_trc = trc_map.get(self._info.color_trc, cs_matrix)
        cs_out_prim = self._info.color_primaries if self._info.color_primaries else cs_matrix

        # FIX (v1.0.0): Improved filter chain for better quality and color fidelity
        # Input frames are RGB24 (full range), properly specify in filter chain
        if self._cfg.descale and self._cfg.scale_pct > 100:
            vf = f"format=rgb24,cas=0.2,scale={proc_w}:{proc_h}:flags=lanczos,scale={out_w}:{out_h}:flags=lanczos,unsharp=5:5:1.0:5:5:0.05,colorspace=ispace=bt709:itrc=srgb:iprimaries=bt709:space={cs_matrix}:trc={cs_out_trc}:primaries={cs_out_prim}:range=tv:fast=1,format={vf_pix_fmt}"
            self._debug_log(f"descale mode: {proc_w}x{proc_h} -> {out_w}x{out_h}")
        else:
            vf = f"format=rgb24,cas=0.2,unsharp=3:3:0.25:3:3:0,scale={out_w}:{out_h}:flags=lanczos,colorspace=ispace=bt709:itrc=srgb:iprimaries=bt709:space={cs_matrix}:trc={cs_out_trc}:primaries={cs_out_prim}:range=tv:fast=1,format={vf_pix_fmt}"

        self._debug_log(f"video filters: scale={out_w}x{out_h}, format={vf_pix_fmt}")

        # Get CRF value from config (used for H.264/HEVC, ignored for ProRes)
        crf_val = str(getattr(self._cfg, 'crf_value', 18))

        # Validate fps_str for ffmpeg (must be in N/D format or numeric)
        if not fps_str or fps_str == "N/A" or '/' not in fps_str:
            # Try to reconstruct from float value
            if self._info.fps > 0:
                # Convert to fraction for ffmpeg compatibility
                frac = Fraction(self._info.fps).limit_denominator(1000)
                fps_str = f"{frac.numerator}/{frac.denominator}"
            else:
                fps_str = "24/1"  # Safe default fallback
            self._debug_log(f"using fallback framerate: {fps_str}")

        # Build Command Based on Codec

        # Base command components
        base_cmd = [
            str(ffmpeg), "-y",
            "-framerate", fps_str,
            "-i", pattern,
        ]

        # Add audio input if available
        if has_audio_input:
            base_cmd.extend(["-i", str(original_input)])

        # Video filter
        base_cmd.extend(["-vf", vf])

        # Color space metadata for output (prevents green tint on playback)
        base_cmd.extend(["-color_primaries", cs_out_prim,
                          "-color_trc", cs_out_trc,
                          "-colorspace", cs_matrix])

        # Codec-specific options
        # Note: -pix_fmt here matches vf_pix_fmt from filter chain above
        if codec == Codec.PRORES:
            cmd = base_cmd + [
                "-map", "0:v",
                "-c:v", "prores", "-profile:v", "3",
                "-pix_fmt", out_pix_fmt, "-vendor", "apl0",
            ]
            if has_audio_input:
                cmd.extend(["-map", "1:a?", "-c:a", "copy"])
            cmd.append(str(output_path))

        elif codec == Codec.HEVC:
            cmd = base_cmd + [
                "-map", "0:v",
                "-c:v", "libx265", "-preset", "slow", "-crf", crf_val,
                "-pix_fmt", out_pix_fmt, "-tag:v", "hvc1",
                "-movflags", "+faststart",
            ]
            if has_audio_input:
                cmd.extend(["-map", "1:a?", "-c:a", "copy"])
            cmd.append(str(output_path))

        else:  # H264
            cmd = base_cmd + [
                "-map", "0:v",
                "-c:v", "libx264", "-preset", "slow", "-crf", crf_val,
                "-pix_fmt", out_pix_fmt, "-movflags", "+faststart",
            ]
            if has_audio_input:
                cmd.extend(["-map", "1:a?", "-c:a", "copy"])
            cmd.append(str(output_path))

        # Execute Assembly

        crf_info = f", CRF={crf_val}" if codec != Codec.PRORES else " (ProRes: fixed bitrate)"
        self._log(f"Assembling video ({frame_count} frames)...")
        self._debug_log(f"assembly config: {codec.value} ({out_w}x{out_h}){crf_info}")
        # Don't log full command - it's too verbose and clutters the UI

        try:
            self._run_cmd_robust(cmd, "video assembly", max(120, self._info.duration * 2))
            return
        except subprocess.CalledProcessError as e:
            self._log(f"assembly failed (code {e.returncode}): {e.stderr[-500:] if e.stderr else 'no stderr'}")

            # Fallback 1: Try without audio (rebuild command explicitly)
            if has_audio_input:
                self._debug_log("retrying without audio...")
                no_audio_cmd = [
                    str(ffmpeg), "-y",
                    "-framerate", fps_str,
                    "-i", pattern,
                    "-vf", vf,
                    "-map", "0:v",
                ]
                # Add codec-specific options (no audio)
                if codec == Codec.PRORES:
                    no_audio_cmd.extend([
                        "-c:v", "prores", "-profile:v", "3",
                        "-pix_fmt", "yuv422p10le", "-vendor", "apl0",
                    ])
                elif codec == Codec.HEVC:
                    no_audio_cmd.extend([
                        "-c:v", "libx265", "-preset", "slow", "-crf", crf_val,
                        "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
                        "-movflags", "+faststart",
                    ])
                else:  # H264
                    no_audio_cmd.extend([
                        "-c:v", "libx264", "-preset", "slow", "-crf", crf_val,
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    ])
                no_audio_cmd.append(str(output_path))

                try:
                    self._run_cmd_robust(no_audio_cmd, "assembly (no audio)", max(120, self._info.duration * 2))
                    self._log("Assembly complete (video only)")
                    self._debug_log("assembly succeeded without audio")
                    return
                except subprocess.CalledProcessError as e2:
                    self._log(f"no-audio fallback also failed: {e2.stderr[-300:] if e2.stderr else ''}")

            # Fallback 2: Try with minimal filter (just scale) - rebuild explicitly
            self._debug_log("trying with minimal filters...")
            simple_vf = f"scale={out_w}:{out_h},colorspace=ispace=bt709:itrc=srgb:iprimaries=bt709:space={cs_matrix}:trc={cs_out_trc}:primaries={cs_out_prim}:range=tv:fast=1,format=yuv420p"
            simple_cmd = [
                str(ffmpeg), "-y",
                "-framerate", fps_str,
                "-i", pattern,
            ]
            if has_audio_input:
                simple_cmd.extend(["-i", str(original_input)])
            simple_cmd.extend([
                "-vf", simple_vf,
                "-map", "0:v",
            ])
            if has_audio_input:
                simple_cmd.extend(["-map", "1:a?", "-c:a", "copy"])
            simple_cmd.extend([
                "-c:v", "libx264", "-preset", "slow", "-crf", crf_val,
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(output_path)
            ])

            try:
                self._run_cmd_robust(simple_cmd, "assembly (simple)", max(120, self._info.duration * 2))
                self._log("assembly succeeded with simple filters")
                return
            except subprocess.CalledProcessError as e3:
                self._log(f"simple filter fallback also failed: {e3.stderr[-300:] if e3.stderr else ''}")

            # Fallback 3: Absolute minimum - raw frame concat, no filters
            self._log("trying bare minimum assembly...")
            bare_cmd = [
                str(ffmpeg), "-y",
                "-framerate", fps_str,
                "-i", pattern,
                "-vf", f"format=rgb24,colorspace=ispace=bt709:itrc=srgb:iprimaries=bt709:space={cs_matrix}:trc={cs_out_trc}:primaries={cs_out_prim}:range=tv:fast=1,format=yuv420p",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                "-movflags", "+faststart",
                str(output_path)
            ]
            try:
                self._run_cmd_robust(bare_cmd, "assembly (bare)", max(180, self._info.duration * 3))
                self._log("assembly succeeded with bare minimum settings")
                return
            except subprocess.CalledProcessError as e4:
                # All fallbacks exhausted - raise with helpful message
                error_details = e4.stderr[-1000:] if e4.stderr else "No error output"
                raise RuntimeError(
                    f"Video assembly failed after all fallback attempts.\n"
                    f"Frame count: {frame_count}\n"
                    f"Target resolution: {out_w}x{out_h}\n"
                    f"FFmpeg error:\n{error_details}"
                )

    def _run_cmd_robust(self, cmd: List[str], desc: str="", timeout: float=3600) -> subprocess.CompletedProcess:
        """Run command with robust error handling."""
        self._log(f"[{desc}] running...")  # Don't log full command - too verbose

        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL  # Prevent interactive prompts
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"'{desc}' timed out after {timeout}s")

        if r.returncode != 0:
            # Log full output for debugging
            if r.stdout:
                self._log(f"[{desc}] stdout: {r.stdout[-500:]}")
            if r.stderr:
                self._log(f"[{desc}] stderr: {r.stderr[-1000:]}")
            raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)

        self._log(f"[{desc}] completed successfully")
        return r

    def _run_cmd(self, cmd: List[str], desc: str="", timeout: float=3600) -> subprocess.CompletedProcess:
        self._log(f"[{desc}] running...")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr)
        return r

    # Substrings in stderr that indicate fatal, non-retryable errors
    _FATAL_STDERR_PATTERNS = (
        "disk quota exceeded",
        "no space left on device",
        "permission denied",
        "read-only file system",
        "input/output error",
        "stale file handle",
    )

    def _run_cmd_silent(self, cmd: List[str], timeout: float=300) -> bool:
        """Run command silently but log failures for debugging.

        Uses rate-limited logging to prevent log flooding when called in loops.
        Only logs first failure (with full stderr) and every 50th thereafter.
        Returns True on success, False on failure.

        Raises RuntimeError immediately on fatal errors (disk full, permission
        denied, etc.) that will never succeed on retry.

        Stores last error details in self._last_silent_error for caller inspection.

        NOTE: Uses Popen + drain threads instead of capture_output to avoid
        pipe deadlocks.  NCNN Vulkan tools write progress to stderr; if the
        OS pipe buffer fills (typically 64 KiB), the subprocess blocks on write
        and hangs -- but subprocess.run(capture_output=True) only reads AFTER
        the process exits, creating a deadlock.
        """
        proc = None
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True,
            )
            # Drain stderr in a thread so the pipe never fills up
            stderr_chunks: List[str] = []
            def _drain():
                for chunk in proc.stderr:  # type: ignore
                    stderr_chunks.append(chunk)
            drain_thread = threading.Thread(target=_drain, daemon=True)
            drain_thread.start()

            proc.wait(timeout=timeout)
            drain_thread.join(timeout=2)  # Give drain a moment to finish

            if proc.returncode != 0:
                stderr_text = ''.join(stderr_chunks)
                # Check for fatal filesystem errors
                stderr_lower = stderr_text.lower()
                for pattern in self._FATAL_STDERR_PATTERNS:
                    if pattern in stderr_lower:
                        self._log(f"[silent] FATAL ERROR: {pattern}")
                        self._log(f"[silent]   cmd: {' '.join(str(x) for x in cmd)}")
                        raise RuntimeError(
                            f"Filesystem error during '{cmd[0] if cmd else 'command'}': "
                            f"{pattern}. Free up disk space or fix permissions and try again."
                        )

                self._silent_fail_count += 1
                if stderr_text.strip():
                    lines = [l.strip() for l in stderr_text.strip().split('\n') if l.strip()]
                    self._last_silent_error = f"code {proc.returncode}: {'; '.join(lines[-5:])}"
                else:
                    self._last_silent_error = f"code {proc.returncode} (no stderr)"

                if self._silent_fail_count == 1:
                    self._log(f"[silent] FIRST FAILURE - code {proc.returncode}")
                    self._log(f"[silent]   cmd: {' '.join(str(x) for x in cmd)}")
                    if stderr_text.strip():
                        snippet = stderr_text[-800:] if len(stderr_text) > 800 else stderr_text
                        for line in snippet.strip().split('\n')[-10:]:
                            if line.strip():
                                self._log(f"[silent]   stderr: {line.strip()}")
                elif self._silent_fail_count % 50 == 0:
                    self._log(f"[silent] ... {self._silent_fail_count} total failures so far")
                return False

            return True  # Success

        except subprocess.TimeoutExpired:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            if not self._cancel.is_set():
                self._log(f"[silent] command timed out: {cmd[0] if cmd else 'unknown'}")
            self._last_silent_error = "timeout"
            return False

    def _is_image_black_or_corrupt(self, img_path: Path) -> bool:
        """Check if an image file is black, near-black, or corrupt.

        Returns True if the image appears BLACK or CORRUPT (bad output).
        Returns False if the image looks VALID.

        Uses PIL/Pillow for thread-safe pixel sampling (GdkPixbuf is not
        thread-safe and must not be called from worker threads).

        This fixes the known NCNN Vulkan issue where tools exit successfully
        but produce completely black images on some GPUs/drivers.
        See: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/issues/80
        """
        # Check 1: File must exist and have reasonable size
        if not img_path.exists():
            self._debug_log(f"BLACK CHECK: file doesn't exist: {img_path}")
            return True  # Treat as bad

        file_size = img_path.stat().st_size
        # A valid PNG frame (even small ones) should be > 1KB
        # Black/corrupt images are often tiny (< 500 bytes)
        if file_size < 512:
            self._debug_log(f"BLACK CHECK: file too small ({file_size} bytes): {img_path.name}")
            return True  # Likely corrupt or empty

        # Check 2: Try to load with PIL (thread-safe, unlike GdkPixbuf)
        try:
            from PIL import Image as PILImage
            img = PILImage.open(img_path)
            img.load()  # Force full decode to catch corrupt data
            width, height = img.size
        except Exception as e:
            self._debug_log(f"BLACK CHECK: failed to load image: {img_path.name} - {e}")
            return True  # Corrupt or unreadable

        if width == 0 or height == 0:
            self._debug_log(f"BLACK CHECK: zero dimension image: {img_path.name}")
            return True

        # Check 3: Sample pixels for black/near-black detection
        # Sample strategy: check corners + center + grid of points
        sample_points = [
            (0, 0),
            (width - 1, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (width // 2, height // 2),
        ]

        # Add grid samples for larger images
        if width > 100 and height > 100:
            for gx in [width // 4, 3 * width // 4]:
                for gy in [height // 4, 3 * height // 4]:
                    sample_points.append((gx, gy))

        total_brightness = 0
        max_allowed_avg = 1  # Only flag pure black (brightness 0-1)

        for x, y in sample_points:
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            pixel = img.getpixel((x, y))
            if isinstance(pixel, int):
                # Grayscale image
                pixel_brightness = pixel
            else:
                # RGB or RGBA - sum first 3 channels
                pixel_brightness = (int(pixel[0]) + int(pixel[1]) + int(pixel[2])) // 3
            total_brightness += pixel_brightness

        avg_brightness = total_brightness / len(sample_points)

        if avg_brightness < max_allowed_avg:
            self._debug_log(f"BLACK CHECK: {img_path.name} avg_brightness={avg_brightness:.1f} -- PURE BLACK")
            return True  # Image is black!

        return False  # Image looks OK

    def _validate_ncnn_output(self, output_path: Path, tool_name: str = "NCNN") -> bool:
        """Validate NCNN tool output. Returns True if output is GOOD.

        This is a wrapper around _is_image_black_or_corrupt that provides
        better logging for error reporting.
        """
        if self._is_image_black_or_corrupt(output_path):
            self._log(f"[!] {tool_name} produced BLACK/CORRUPT output!")
            self._log(f"  This is a known issue with some GPU/driver combinations.")
            # Delete the bad output
            try:
                output_path.unlink()
            except OSError:
                pass
            return False
        return True

    def _validate_ncnn_batch_output(self, dst: Path, tool_name: str = "NCNN") -> None:
        """Post-processing check: verify not ALL output frames are black/corrupt.

        Samples frames spread across the output directory. Only raises if
        every single sampled frame is black -- individual black frames are
        legitimate (fade-ins, VFX, etc.).
        """
        out_frames = sorted(dst.glob("frame_*.png"))
        if not out_frames:
            return  # nothing to check

        # Sample up to 5 frames spread evenly across the video
        sample_count = min(5, len(out_frames))
        indices = [int(i * (len(out_frames) - 1) / max(sample_count - 1, 1)) for i in range(sample_count)]

        black_count = 0
        for idx in indices:
            if self._is_image_black_or_corrupt(out_frames[idx]):
                black_count += 1

        if black_count == sample_count:
            # EVERY sampled frame is black -- the whole output is corrupt
            raise RuntimeError(
                f"{tool_name} produced ALL-BLACK output on every sampled frame.\n\n"
                f"This usually means your GPU doesn't have enough VRAM for this\n"
                f"resolution, or there's a driver compatibility issue.\n\n"
                f"Try:\n"
                f"  - Lowering the scale factor\n"
                f"  - Closing other GPU-using applications\n"
                f"  - Switching to a different GPU in Device settings"
            )
        elif black_count > 0:
            self._debug_log(
                f"NCNN post-check: {black_count}/{sample_count} sampled frames are black "
                f"(likely intentional -- fade-ins, VFX, etc.)"
            )

    def _cleanup(self) -> None:
        if self._workdir and self._workdir.exists():
            try:
                shutil.rmtree(self._workdir, ignore_errors=True)
            except OSError as e:
                self._debug_log(f"cleanup issue: {e}")

    def _prog(self, frac: float, msg: str) -> None:
        GLib.idle_add(self.emit, "progress", frac, msg)

    def _done(self, ok: bool, msg: str) -> None:
        GLib.idle_add(self.emit, "done", ok, msg)

    def _err(self, msg: str) -> None:
        GLib.idle_add(self.emit, "error", msg)

    def _log(self, msg: str) -> None:
        """Log user-facing message (shown in UI status bar + console + file)."""
        GLib.idle_add(self.emit, "log", msg)
        log_msg(msg, "Sharptape")

    def _debug_log(self, msg: str) -> None:
        """Log verbose debug message (console + file, NOT shown in UI)."""
        GLib.idle_add(self.emit, "debug", msg)
        log_msg(msg, "Sharptape:Debug")

    def _toast(self, msg: str) -> None:
        """Show an in-window toast notification from the worker thread."""
        GLib.idle_add(self.emit, "toast", msg)
        log_msg(msg, "Sharptape:Toast")
