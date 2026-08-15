#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sharptape -- GTK4/Libadwaita application
Main window, UI, and entry point.
"""

import os
import sys
import re
import json
import shutil
import signal
import subprocess
import threading
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core import (
    APP_ID, DATA_DIR, SETTINGS_PATH, VERSION_PATH, __version__,
    VIDEO_EXTS,
    NCNN_BINS, NCNN_BIN_DIR, MODEL_DIR,
    SPACING_XS, SPACING_SM, SPACING_MD, SPACING_LG,
    SLIDER_CFG, VALUE_LABEL_WIDTH,
    PipelineProgress, BASICVSR_MODEL_FILENAME,
    HardwareProfile, HardwareTier,
    find_bin,
    VideoInfo, probe_video, target_dims,
    Engine, GPUBackend, Codec, TemporalMethod,
    Config, log_msg,
)
from backend import Worker
from languages import init_languages, get_lang, _, _f

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("GLib", "2.0")

from gi.repository import Gtk, Adw, GLib, Gio, GObject, Pango, Gdk, GdkPixbuf

warnings.filterwarnings("ignore", category=DeprecationWarning, module="gi.repository.Adw")

# --- Main Window ---

class Window(Adw.ApplicationWindow):
    """
    Main application window.

    GNOME HIG compliance notes:
    - Uses Adw.PreferencesPage/Group for consistent layout
    - Spacing follows 6px increment system
    - All sliders have fixed-width value labels for alignment
    - Consistent margin/padding throughout
    """

    def __init__(self, app: Adw.Application):
        super().__init__(application=app)
        self.set_title("Sharptape")
        self.set_default_size(1100, 750)  # Wider to fit long subtitles without truncation

        # Icon is set via application_icon in About dialog (uses APP_ID)

        # Load config with corruption detection
        cfg, corruption_info = Config.load()

        # Handle config corruption BEFORE building UI
        if corruption_info is not None:
            action = self._show_config_corrupted_dialog(corruption_info)
            if action == "quit":
                print("[sharptape] user chose to quit due to config corruption")
                self.close()
                return
            elif action == "fresh":
                # Start fresh with defaults
                cfg = Config()
                cfg.save()
                print("[sharptape] config reset to defaults per user choice")
            # else: "load_preserved" -- keep cfg as-is (partial load)

        self._cfg = cfg
        self._info: Optional[VideoInfo] = None
        self._worker: Optional[Worker] = None
        self._cancel = threading.Event()
        self._busy = False
        self._atomic_overwrite_original: Optional[str] = None  # transient, not persisted

        # Size group for perfect slider alignment (GNOME HIG compliant)
        self._slider_size_group = Gtk.SizeGroup.new(Gtk.SizeGroupMode.HORIZONTAL)

        self._build_ui()
        self.connect("close-request", self._on_close)

    def _log(self, msg: str) -> None:
        if hasattr(self, '_status'):
            self._status.set_text(msg)
        log_msg(msg, "Sharptape")

    def _on_close(self, *a) -> None:
        """Handle window close: cancel worker, cleanup, save config."""
        # Cancel any running worker thread
        if self._cancel and not self._cancel.is_set():
            self._cancel.set()
            self._log("Cancelling processing...")

        # Kill any lingering child processes (ffmpeg, ncnn, etc.)
        self._kill_child_processes()

        self._cfg.save()

        # Give worker a moment to clean up, then return False to allow close
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=0.5)  # Wait up to 500ms for graceful shutdown

        return False  # Allow window to close

    def _show_config_corrupted_dialog(self, info: Dict[str, Any]) -> str:
        """Show config corruption dialog with vertical buttons. Returns 'load_preserved', 'fresh', or 'quit'."""
        result = ["quit"]  # mutable container for nested lambda

        preserved = info.get("preserved", [])
        missing = info.get("missing", [])
        total_count = info.get("total_count", len(preserved) + len(missing))
        total_corrupted = info.get("total_corrupted", False)

        if not total_corrupted and len(preserved) > 0:
            body = (
                f"Your config is missing some values that were previously "
                f"saved.\n\n"
                f"We can try to load what's still there, but the rest "
                f"will be reset to defaults.\n\n"
                f"Preserved values: {len(preserved)} out of {total_count} total"
            )
            buttons = [
                ("Load what's still saved", "suggested-action", "load_preserved"),
                ("Start fresh", "warning", "fresh"),
                ("Quit", "destructive-action", "quit"),
            ]
        else:
            body = (
                "Your config has been corrupted or malformed.\n\n"
                "Sharptape will continue with default settings."
            )
            buttons = [
                ("Continue", "warning", "fresh"),
                ("Quit", "destructive-action", "quit"),
            ]

        # Build custom dialog with vertical button layout
        custom = Gtk.Dialog(transient_for=self, modal=True)
        custom.set_title("Config corrupted/invalid")

        content_area = custom.get_content_area()
        content_area.set_margin_start(24)
        content_area.set_margin_end(24)
        content_area.set_margin_top(24)
        content_area.set_margin_bottom(24)
        content_area.set_spacing(12)

        heading = Gtk.Label(label="Config corrupted/invalid")
        heading.add_css_class("title-2")
        heading.add_css_class("bold")
        heading.set_xalign(0)
        content_area.append(heading)

        body_label = Gtk.Label(label=body)
        body_label.set_wrap(True)
        body_label.set_xalign(0)
        body_label.add_css_class("body")
        content_area.append(body_label)

        btn_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        btn_box.set_margin_top(12)
        btn_box.set_halign(Gtk.Align.CENTER)
        content_area.append(btn_box)

        for btn_label, css_class, action in buttons:
            btn = Gtk.Button(label=btn_label)
            btn.add_css_class(css_class)
            btn.set_size_request(250, -1)
            btn.connect("clicked", lambda b, a=action: (result.__setitem__(0, a), custom.close()))
            btn_box.append(btn)

        custom.present()

        # Block until dialog closes
        loop = GLib.MainLoop()
        def on_close_request(*_a):
            loop.quit()
            return False  # Allow default destroy to proceed
        custom.connect("close-request", on_close_request)
        loop.run()

        return result[0]

    def _kill_child_processes(self) -> None:
        """Force-kill any child processes spawned by Sharptape.

        This ensures ffmpeg, ncnn-vulkan, etc. don't keep running
        after the window is closed.
        """
        try:
            # Get our PID and kill all our children
            my_pid = os.getpid()

            # On Linux, use /proc to find and kill children
            if sys.platform.startswith('linux'):
                try:
                    # Find all child PIDs
                    result = subprocess.run(
                        ['pgrep', '-P', str(my_pid)],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.stdout.strip():
                        for pid_str in result.stdout.strip().split('\n'):
                            try:
                                child_pid = int(pid_str)
                                os.kill(child_pid, signal.SIGTERM)
                                print(f"[sharptape] Killed child process {child_pid}")
                            except (ProcessLookupError, ValueError):
                                pass  # Process already dead
                except Exception as e:
                    print(f"[sharptape] Warning: couldn't kill children: {e}")

            # Also try killing by process group (more thorough)
            try:
                os.killpg(os.getpgid(0), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass  # No process group or no permission

        except Exception as e:
            print(f"[sharptape] Cleanup error: {e}")

    def _build_ui(self) -> None:
        # Create ToastOverlay as root for in-window notifications
        self._toast_overlay = Adw.ToastOverlay()
        self.set_content(self._toast_overlay)

        # Apply custom CSS to fix text truncation issues
        self._apply_custom_css()

        # Build main content
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content.set_hexpand(True)  # Allow content to fill window width
        content.set_vexpand(True)
        self._toast_overlay.set_child(content)

        content.append(self._header())

        scroll = Gtk.ScrolledWindow(hexpand=True, vexpand=True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)  # No horizontal scroll, let content expand
        scroll.set_min_content_width(900)  # Ensure minimum width for text display

        # Wrap prefs page in a Box that forces width expansion
        prefs_wrapper = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        prefs_wrapper.set_hexpand(True)
        prefs_wrapper.set_halign(Gtk.Align.FILL)
        prefs_wrapper.append(self._prefs_page())
        scroll.set_child(prefs_wrapper)
        content.append(scroll)

        content.append(self._footer())

    def _apply_custom_css(self) -> None:
        """Apply custom CSS for visual tweaks.

        Note: Text ellipsization must be done PROGRAMMATICALLY (via set_ellipsize),
        NOT via CSS - GTK4 doesn't support -gtk-ellipsize or ellipsize as CSS properties.
        The programmatic fix is in _fix_combo_row_text() and _disable_ellipsize_recursive().

        This CSS only handles valid visual properties like sizing/spacing.
        """
        css_data = """
        /* Make ComboRow dropdown popover wider so text fits better */
        popover.combo list > row {
            padding: 8px 12px;
            min-width: 320px;
        }

        /* Ensure rows have enough room for content */
        actionrow,
        comborow {
            min-height: 52px;
        }
        """

        css_provider = Gtk.CssProvider()
        try:
            css_provider.load_from_data(css_data.encode('utf-8'))
            Gtk.StyleContext.add_provider_for_display(
                Gdk.Display.get_default(),
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
        except Exception as e:
            print(f"[sharptape] Warning: Could not load custom CSS: {e}")

    def _fix_combo_row_text(self, combo_row: Adw.ComboRow) -> None:
        """Programmatically fix text truncation in a ComboRow.

        GTK4's ComboRow has internal labels that get ellipsized by default.
        This method finds those labels and disables ellipsization.
        """
        # Use idle_add to ensure widget is realized before we access children
        def fix_labels():
            try:
                # The ComboRow's selected-item label is usually in a specific position
                # We need to find and update all labels inside
                self._disable_ellipsize_recursive(combo_row)
            except Exception as e:
                print(f"[sharptape] Warning: Could not fix combo row labels: {e}")
            return False  # Don't repeat

        GLib.idle_add(fix_labels)

    def _disable_ellipsize_recursive(self, widget) -> None:
        """Recursively disable ellipsization on all Label widgets."""
        if isinstance(widget, Gtk.Label):
            widget.set_ellipsize(Pango.EllipsizeMode.NONE)
            widget.set_wrap(True)
            widget.set_xalign(0.0)
            widget.set_max_width_chars(-1)  # Remove max width constraint

        # Recurse into child widgets
        if hasattr(widget, 'get_first_child'):
            child = widget.get_first_child()
            while child:
                self._disable_ellipsize_recursive(child)
                child = child.get_next_sibling()

    def _header(self) -> Adw.HeaderBar:
        bar = Adw.HeaderBar()
        title_label = Gtk.Label()
        title_label.set_markup('<b>Sharptape</b>')
        bar.set_title_widget(title_label)

        open_btn = Gtk.Button(css_classes=["flat"])
        open_btn.set_icon_name("document-open-symbolic")
        open_btn.set_tooltip_text("Open Video")
        open_btn.connect("clicked", self._on_open)
        bar.pack_start(open_btn)

        about_btn = Gtk.Button(css_classes=["flat"])
        about_btn.set_icon_name("help-about-symbolic")
        about_btn.connect("clicked", lambda *a: self.get_application()._about())
        bar.pack_end(about_btn)

        self._cancel_btn = Gtk.Button(label="Cancel", css_classes=["flat"])
        self._cancel_btn.set_visible(False)
        self._cancel_btn.connect("clicked", self._on_cancel)
        bar.pack_end(self._cancel_btn)

        return bar

    def _prefs_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage()
        page.set_hexpand(True)  # Allow page to expand to full width
        page.set_halign(Gtk.Align.FILL)
        page.add(self._grp_files())
        page.add(self._grp_resolution())
        page.add(self._grp_enhance())
        page.add(self._grp_device())
        page.add(self._grp_advanced())

        # Silent hardware detection (no UI -- runs for auto-config only)
        import importlib.util as _ilu
        self._pytorch_available = _ilu.find_spec("torch") is not None
        if not self._pytorch_available:
            self._cfg.temporal_method = TemporalMethod.FFMPEG
            self._cfg.save()
        GLib.timeout_add(100, self._detect_hw_silent)

        # Apply initial scale cap based on saved upscaler/tier
        GLib.idle_add(self._update_scale_max)

        return page

    def _footer(self) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=SPACING_SM)
        box.add_css_class("toolbar")
        box.set_margin_start(SPACING_SM); box.set_margin_end(SPACING_SM)
        box.set_margin_top(SPACING_XS); box.set_margin_bottom(SPACING_XS)

        spin = Gtk.Spinner(visible=False)
        self._spinner = spin
        box.append(spin)

        status = Gtk.Label(label="Ready", hexpand=True, xalign=0)
        status.add_css_class("caption")
        self._status = status
        box.append(status)

        prog = Gtk.ProgressBar(hexpand=True, visible=False)
        self._prog_bar = prog
        box.append(prog)

        enhance = Gtk.Button(label="Enhance Video", css_classes=["suggested-action"], height_request=42)
        enhance.set_valign(Gtk.Align.CENTER)  # Center vertically
        enhance.set_halign(Gtk.Align.END)     # Keep on the right
        enhance.connect("clicked", self._on_enhance)
        self._enhance_btn = enhance
        box.append(enhance)

        return box

    # File Selection Group

    def _grp_files(self) -> Adw.PreferencesGroup:
        grp = Adw.PreferencesGroup()
        grp.set_title("Files")

        self._in_row = Adw.ActionRow(title="Input Video", subtitle="Select a video file...",
                                      icon_name="video-x-generic-symbolic")
        self._in_row.set_activatable(True)
        self._in_row.connect("activated", self._on_open)
        grp.add(self._in_row)

        self._out_row = Adw.ActionRow(title="Output Video", subtitle="Save location...",
                                       icon_name="folder-download-symbolic")
        self._out_row.set_activatable(True)
        self._out_row.connect("activated", self._on_save)
        grp.add(self._out_row)

        # codec selector
        self._codec_combo = Adw.ComboRow(title="Output Codec", subtitle="Encoding format")
        self._codec_combo.set_subtitle_lines(2)  # Allow wrapping for longer codec names
        codec_model = Gtk.StringList.new([
            "H.264 (CRF 18)",
            "HEVC/H.265 (CRF 18)",
            "ProRes 422 HQ (lossless-ish)"
        ])
        self._codec_combo.set_model(codec_model)
        ci = list(Codec).index(self._cfg.codec)
        self._codec_combo.set_selected(ci)
        self._codec_combo.connect("notify::selected", self._on_codec_changed)
        self._fix_combo_row_text(self._codec_combo)  # Fix text truncation
        grp.add(self._codec_combo)

        return grp

    # Resolution Group

    def _grp_resolution(self) -> Adw.PreferencesGroup:
        grp = Adw.PreferencesGroup()
        grp.set_title("Resolution")
        grp.set_description("Output size (100% keeps original)")

        # Use same slider_row pattern as enhancement group for consistency
        row = Adw.ActionRow(title="Scale Factor", subtitle="Resize output")

        # Build the suffix widget (value label + slider + info + descale)
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=SPACING_XS)

        # Value header row (aligned with enhancement sliders)
        hdr = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=SPACING_SM)
        val_lbl = Gtk.Label(label=f"{self._cfg.scale_pct}%")
        val_lbl.add_css_class("dim-label")
        val_lbl.set_halign(Gtk.Align.END)
        val_lbl.set_xalign(1.0)
        val_lbl.set_size_request(VALUE_LABEL_WIDTH, -1)  # Fixed width for alignment
        self._scale_val = val_lbl
        hdr.append(Gtk.Label(hexpand=True))  # spacer
        hdr.append(val_lbl)
        vbox.append(hdr)

        # Slider (hexpand to fill available space)
        cfg = SLIDER_CFG["resolution"]
        slider = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, cfg["min"], cfg["max"], cfg["step"])
        slider.set_value(self._cfg.scale_pct)
        slider.set_draw_value(False)
        slider.set_hexpand(True)
        slider.connect("value-changed", self._on_scale_changed)
        self._scale_slider = slider
        vbox.append(slider)

        # Dimension preview label
        dim_lbl = Gtk.Label(label="")
        dim_lbl.add_css_class("caption")
        dim_lbl.set_wrap(True)
        dim_lbl.set_xalign(0.0)
        dim_lbl.set_size_request(VALUE_LABEL_WIDTH, -1)  # Match label width
        self._dim_lbl = dim_lbl
        vbox.append(dim_lbl)

        # De-scale toggle row
        ds_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=SPACING_SM)
        ds_lbl = Gtk.Label(label="De-scale back to original after processing")
        ds_lbl.set_halign(Gtk.Align.START)
        ds_lbl.set_hexpand(True)
        ds_lbl.set_wrap(True)
        ds_lbl.set_xalign(0.0)
        ds_box.append(ds_lbl)
        ds_sw = Gtk.Switch(active=self._cfg.descale, valign=Gtk.Align.CENTER)
        ds_sw.connect("notify::active", self._on_descale_toggled)
        self._descale_sw = ds_sw
        ds_box.append(ds_sw)
        vbox.append(ds_box)

        row.add_suffix(vbox)
        grp.add(row)

        self._refresh_dims()
        return grp

    # Enhancement Group

    def _grp_enhance(self) -> Adw.PreferencesGroup:
        grp = Adw.PreferencesGroup()
        grp.set_title("Enhancement")
        grp.set_description("Tweak these to taste")

        self._row_deblock = self._slider_row("Deblock", "Remove compression artifacts", "deblock", self._on_deblock, self._cfg.deblock)
        self._row_denoise = self._slider_row("Denoise", "Reduce noise", "denoise", self._on_denoise, self._cfg.denoise)
        self._row_sharpen = self._slider_row("Sharpen", "Enhance details", "sharpen", self._on_sharpen, self._cfg.sharpen)
        self._row_deblur = self._slider_row("De-blur", "Fix softness", "deblur", self._on_deblur, self._cfg.deblur)
        self._row_aa = self._slider_row("Anti-alias", "Negative=crisp, positive=smooth", "antialias", self._on_aa, self._cfg.antialias)

        for r in [self._row_deblock, self._row_denoise, self._row_sharpen, self._row_deblur, self._row_aa]:
            grp.add(r)

        # Upscaler model selector
        self._upscaler_combo = Adw.ComboRow(
            title="Upscaler",
            subtitle="Anime-styled cartoonish upscaling." if self._cfg.upscaler_model == "cugan" else "Realism-focused upscaling.\nMay produce oversharpened artifacts."
        )
        self._upscaler_combo.set_subtitle_lines(2)
        upscaler_model = Gtk.StringList.new([
            "Anime (Real-CUGAN)",
            "Realistic (Real-ESRGAN)",
        ])
        self._upscaler_combo.set_model(upscaler_model)
        upscaler_idx = 0 if self._cfg.upscaler_model == "cugan" else 1
        self._upscaler_combo.set_selected(upscaler_idx)
        self._upscaler_combo.connect("notify::selected", self._on_upscaler_changed)
        self._fix_combo_row_text(self._upscaler_combo)
        # Add tooltips per-item
        self._upscaler_combo.set_tooltip_text("Select AI upscaling model")
        grp.add(self._upscaler_combo)

        # Temporal processing method selector
        self._temporal_combo = Adw.ComboRow(
            title="Temporal Processing",
            subtitle="BasicVSR++ = better quality, slower | FFmpeg = faster, simpler"
        )
        self._temporal_combo.set_subtitle_lines(2)
        temporal_model = Gtk.StringList.new([
            "BasicVSR++ (AI)",
            "FFmpeg"
        ])
        self._temporal_combo.set_model(temporal_model)
        temporal_idx = 0 if self._cfg.temporal_method == TemporalMethod.BASICVSR else 1
        self._temporal_combo.set_selected(temporal_idx)
        self._temporal_combo.connect("notify::selected", self._on_temporal_changed)
        self._fix_combo_row_text(self._temporal_combo)
        # If PyTorch is not installed, force FFmpeg temporal and gray out
        if not getattr(self, '_pytorch_available', True):
            self._temporal_combo.set_sensitive(False)
            self._temporal_combo.set_opacity(0.5)
            self._temporal_combo.set_subtitle("Install PyTorch to enable neural temporal processing")
            self._temporal_combo.set_selected(1)
        grp.add(self._temporal_combo)

        return grp

    def _slider_row(self, title: str, sub: str, key: str, cb: Callable, init_val: int) -> Adw.ActionRow:
        """
        Create a perfectly aligned slider row using GNOME HIG Box container pattern.

        Key insight: Put slider AND value label in a SINGLE Box container,
        then add that box as ONE suffix. This ensures they stay together
        and align properly per GNOME HIG guidelines.
        """
        cfg = SLIDER_CFG[key]
        row = Adw.ActionRow(title=title, subtitle=sub)

        # Create horizontal box to hold slider + value label together
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=SPACING_SM)

        # Slider: expands to fill available space
        sl = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, cfg["min"], cfg["max"], cfg["step"])
        sl.set_value(init_val)
        sl.set_draw_value(False)
        sl.set_hexpand(True)
        sl.set_valign(Gtk.Align.CENTER)

        # Add to SizeGroup for consistent width across ALL sliders
        self._slider_size_group.add_widget(sl)

        # Value label: fixed width, right-aligned
        vl = Gtk.Label(label=cfg["fmt"].format(init_val))
        vl.add_css_class("dim-label")
        vl.set_halign(Gtk.Align.END)
        vl.set_xalign(1.0)
        vl.set_size_request(VALUE_LABEL_WIDTH, -1)
        vl.set_valign(Gtk.Align.CENTER)

        def on_val(s):
            v = int(s.get_value())
            vl.set_label(cfg["fmt"].format(v))
            cb(v)

        sl.connect("value-changed", on_val)

        # Assemble: slider (expanding) + label (fixed width)
        hbox.append(sl)
        hbox.append(vl)

        # Add the BOX as single suffix widget (keeps them aligned together)
        row.add_suffix(hbox)

        # Store references
        row._slider = sl
        row._val_lbl = vl
        row._hbox = hbox

        return row

    # Silent Hardware Detection (no UI)

    def _detect_hw_silent(self) -> bool:
        """Run hardware detection silently for auto-config, without any UI."""
        hw_profile = HardwareProfile.detect()
        print(f"[sharptape] HW: {hw_profile.summary_string()}")
        # Show GPU warning only if NO GPU backend at all (not even Vulkan)
        if not hw_profile.has_cuda and not hw_profile.has_mps and not hw_profile.has_vulkan:
            GLib.idle_add(self._maybe_show_gpu_warning)
        return False

    def _grp_device(self) -> Adw.PreferencesGroup:
        """Device selection group -- sits above Advanced Settings."""
        grp = Adw.PreferencesGroup()
        grp.set_title("Device")
        grp.set_description("GPU/CPU selection for AI processing")

        hw = HardwareProfile.detect()

        # Build device list: Auto, then each GPU, then CPU
        device_strings = ["Auto"]
        device_values = ["auto"]
        if hw.gpus:
            # Query system memory for integrated GPU VRAM estimation (usually uses shared RAM)
            total_ram_mb = 0
            try:
                import psutil
                total_ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
            except ImportError:
                pass
            for i, gpu in enumerate(hw.gpus):
                name = gpu.get("name", f"GPU {i}")
                is_igpu = gpu.get("integrated", False)
                if is_igpu:
                    if total_ram_mb > 0:
                        # Integrated GPUs share system memory dynamically (typically up to 50% max)
                        igpu_ram = total_ram_mb // 2
                        label = f"UHD Graphics ({igpu_ram} MB shared VRAM) (integrated)"
                    else:
                        label = "UHD Graphics (integrated)"
                else:
                    vram = gpu.get("vram_total_mb", 0)
                    label = f"{name}"
                    if vram > 0:
                        label += f" ({vram} MB VRAM)"
                device_strings.append(label)
                device_values.append(f"gpu:{i}")
        device_strings.append("CPU (slowest)")
        device_values.append("cpu")

        model = Gtk.StringList.new(device_strings)
        device_combo = Adw.ComboRow(title="Processing Device",
                                    subtitle="Where AI upscaling runs")
        device_combo.set_model(model)
        device_combo.set_subtitle_lines(2)

        # Set current selection from config
        # Migration: old "gpu:0" may now point to iGPU on hybrid systems;
        # if the saved device is an iGPU and a dGPU exists, reset to Auto.
        current = getattr(self._cfg, 'device_id', 'auto')
        if current and current.startswith('gpu:'):
            try:
                saved_idx = int(current.split(':')[1])
                if saved_idx < len(hw.gpus) and hw.gpus[saved_idx].get('integrated', False):
                    # Saved device is iGPU but dGPU exists -- reset to auto
                    if any(not g.get('integrated', False) for g in hw.gpus):
                        current = 'auto'
                        self._cfg.device_id = 'auto'
                        self._cfg.save()
            except (ValueError, IndexError):
                pass
        try:
            idx = device_values.index(current)
        except ValueError:
            idx = 0
        device_combo.set_selected(idx)
        device_combo.connect("notify::selected", self._on_device_changed)
        self._fix_combo_row_text(device_combo)

        # Store references for later use
        self._device_combo = device_combo
        self._device_values = device_values

        grp.add(device_combo)
        return grp

    def _on_device_changed(self, combo: Adw.ComboRow, *_args) -> None:
        """Handle device selection change."""
        idx = combo.get_selected()
        if idx < 0 or not hasattr(self, '_device_values'):
            return
        self._cfg.device_id = self._device_values[idx]
        self._cfg.save()
        self._log(f"Device set to: {self._cfg.device_id}")

    def _grp_advanced(self) -> Adw.PreferencesGroup:
        """Advanced settings with auto-configure toggle and AI model parameters.

        Features:
        - Foldable section (ExpanderRow)
        - Auto-configure toggle that grays out manual controls
        - AI model parameters (CRF slider is also included but unaffected by auto-configure)
        """
        grp = Adw.PreferencesGroup()
        grp.set_title("Advanced")
        grp.set_description("Fine-tuning for advanced users")

        # Create main expander row for foldable behavior
        expander = Adw.ExpanderRow()
        expander.set_title("Advanced Settings")
        expander.set_subtitle("Advanced settings for Sharptape")
        expander.set_enable_expansion(True)
        expander.set_expanded(True)  # Start expanded so users can see it

        # Auto-Detected Settings Info Button

        detected_row = Adw.ActionRow(
            title="Auto-Detected Settings",
            subtitle="View hardware-recommended values"
        )
        detected_btn = Gtk.Button(label="View", css_classes=["flat"])
        detected_btn.set_valign(Gtk.Align.CENTER)
        detected_btn.connect("clicked", self._on_show_detected_settings)
        detected_row.add_suffix(detected_btn)
        expander.add_row(detected_row)

        # Auto-Configure Toggle (always visible)

        auto_row = Adw.ActionRow(
            title="Auto-Configure AI Models",
            subtitle="Let Sharptape optimize settings for your hardware"
        )

        auto_switch = Gtk.Switch(active=self._cfg.auto_config, valign=Gtk.Align.CENTER)
        auto_switch.connect("notify::active", self._on_auto_config_toggled)
        self._auto_config_switch = auto_switch
        auto_row.add_suffix(auto_switch)

        # Add to expander as first child (always visible even when collapsed)
        expander.add_row(auto_row)

        # Separator
        sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sep.set_margin_top(SPACING_SM)
        expander.add_row(sep)

        # Manual Configuration Controls (grayed out when auto is ON)

        # Store all manually-configurable widgets for enable/disable
        self._advanced_widgets = []

        # 1. VSR Batch Size
        batch_row = Adw.ActionRow(
            title="VSR Batch Size",
            subtitle="Frames per BasicVSR++ batch (higher = faster but more VRAM).\nBe careful with the values of this setting in case your VRAM is limited."
        )
        batch_row.set_subtitle_lines(2)  # Allow 2-line subtitle
        batch_spin = Gtk.SpinButton.new_with_range(1, 16, 1)
        batch_spin.set_value(self._cfg.vsr_batch)
        batch_spin.set_valign(Gtk.Align.CENTER)
        batch_spin.connect("value-changed", self._on_vsr_batch_changed)
        batch_row.add_suffix(batch_spin)
        self._vsr_batch_spin = batch_spin
        expander.add_row(batch_row)
        self._advanced_widgets.extend([batch_row, batch_spin])

        # 2. Backbone Blocks
        blocks_row = Adw.ActionRow(
            title="Backbone Blocks",
            subtitle="BasicVSR++ backbone complexity (more = better quality, slower)"
        )
        blocks_row.set_subtitle_lines(2)  # Allow 2-line subtitle
        blocks_spin = Gtk.SpinButton.new_with_range(1, 20, 1)
        blocks_spin.set_value(self._cfg.vsr_blocks)
        blocks_spin.set_valign(Gtk.Align.CENTER)
        blocks_spin.connect("value-changed", self._on_vsr_blocks_changed)
        blocks_row.add_suffix(blocks_spin)
        self._vsr_blocks_spin = blocks_spin
        expander.add_row(blocks_row)
        self._advanced_widgets.extend([blocks_row, blocks_spin])

        # 3. Custom Tile Size -- toggle + conditional slider
        tile_toggle_row = Adw.ActionRow(
            title="Custom Tile Size",
            subtitle="Override automatic tile size for NCNN upscalers"
        )
        tile_toggle_row.set_subtitle_lines(2)
        tile_sw = Gtk.Switch(
            active=not self._cfg.ncnn_tile_auto,
            valign=Gtk.Align.CENTER,
        )
        tile_sw.connect("notify::active", self._on_ncnn_tile_toggled)
        tile_toggle_row.add_suffix(tile_sw)
        self._ncnn_tile_toggle_sw = tile_sw
        self._ncnn_tile_toggle_row = tile_toggle_row
        expander.add_row(tile_toggle_row)
        self._advanced_widgets.append(tile_toggle_row)

        # 3b. Tile size slider (only visible when toggle is ON)
        tile_row = Adw.ActionRow(
            title="NCNN Tile Size",
            subtitle="GPU processing tile size (larger = faster but more VRAM)"
        )
        tile_row.set_subtitle_lines(2)
        tile_spin = Gtk.SpinButton.new_with_range(64, 512, 32)
        tile_spin.set_value(self._cfg.ncnn_tile)
        tile_spin.set_valign(Gtk.Align.CENTER)
        tile_spin.connect("value-changed", self._on_ncnn_tile_changed)
        tile_row.add_suffix(tile_spin)
        self._ncnn_tile_spin = tile_spin
        self._ncnn_tile_row = tile_row
        expander.add_row(tile_row)
        self._advanced_widgets.extend([tile_row, tile_spin])

        # Set initial visibility based on saved setting
        tile_row.set_visible(not self._cfg.ncnn_tile_auto)

        # 4. NCNN Thread Count
        jobs_row = Adw.ActionRow(
            title="NCNN Jobs",
            subtitle="Thread count format: load:proc:save"
        )
        jobs_entry = Gtk.Entry()
        jobs_entry.set_text(self._cfg.ncnn_jobs)
        jobs_entry.set_width_chars(10)
        jobs_entry.set_valign(Gtk.Align.CENTER)
        jobs_entry.connect("changed", self._on_ncnn_jobs_changed)
        jobs_row.add_suffix(jobs_entry)
        self._ncnn_jobs_entry = jobs_entry
        expander.add_row(jobs_row)
        self._advanced_widgets.extend([jobs_row, jobs_entry])

        # 5. Model Tier Selection
        tier_model = Gtk.StringList.new([
            "SE - Balanced quality/speed",
            "PRO - Conservative (best detail)",
            "NOSE - No denoising (fastest)"
        ])
        tier_combo = Adw.ComboRow(title="AI Model Tier", subtitle="Upscaling model variant")
        tier_combo.set_subtitle_lines(2)  # Allow 2-line for long model names
        tier_combo.set_model(tier_model)
        tier_idx = {"se": 0, "pro": 1, "nose": 2}.get(self._cfg.cugan_tier, 0)
        tier_combo.set_selected(tier_idx)
        tier_combo.connect("notify::selected", self._on_cugan_tier_changed)
        self._fix_combo_row_text(tier_combo)  # Fix text truncation
        self._cugan_tier_combo = tier_combo
        tier_combo.set_visible(self._cfg.upscaler_model == "cugan")
        expander.add_row(tier_combo)
        self._advanced_widgets.append(tier_combo)

        # Post-denoise toggle (only visible when NOSE tier is selected)
        post_dn_row = Adw.ActionRow(
            title="Post-denoise",
            subtitle="Apply light denoising after NOSE upscaling"
        )
        post_dn_row.set_subtitle_lines(2)
        post_dn_sw = Gtk.Switch(active=self._cfg.post_denoise, valign=Gtk.Align.CENTER)
        post_dn_sw.connect("notify::active", self._on_post_denoise_toggled)
        post_dn_row.add_suffix(post_dn_sw)
        self._post_denoise_row = post_dn_row
        self._post_denoise_switch = post_dn_sw
        expander.add_row(post_dn_row)
        self._advanced_widgets.append(post_dn_row)
        # Show/hide based on current tier and upscaler
        post_dn_row.set_visible(self._cfg.upscaler_model == "cugan" and self._cfg.cugan_tier == "nose")

        # Separator before toggles
        sep2 = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        sep2.set_margin_top(SPACING_SM)
        expander.add_row(sep2)
        self._advanced_widgets.append(sep2)

        # 6. FP16 Toggle
        fp16_row = Adw.ActionRow(
            title="Use FP16 Precision",
            subtitle="Half-precision floats (faster, less VRAM on supported GPUs)"
        )
        fp16_row.set_subtitle_lines(2)  # Allow 2-line subtitle
        fp16_sw = Gtk.Switch(active=self._cfg.use_fp16, valign=Gtk.Align.CENTER)
        fp16_sw.connect("notify::active", self._on_fp16_toggled)
        fp16_row.add_suffix(fp16_sw)
        self._fp16_switch = fp16_sw
        expander.add_row(fp16_row)
        self._advanced_widgets.extend([fp16_row, fp16_sw])

        # 7. AMP Toggle
        amp_row = Adw.ActionRow(
            title="Use AMP (Mixed Precision)",
            subtitle="Automatic Mixed Precision training/inference"
        )
        amp_row.set_subtitle_lines(2)  # Allow 2-line subtitle
        amp_sw = Gtk.Switch(active=self._cfg.use_amp, valign=Gtk.Align.CENTER)
        amp_sw.connect("notify::active", self._on_amp_toggled)
        amp_row.add_suffix(amp_sw)
        self._amp_switch = amp_sw
        expander.add_row(amp_row)
        self._advanced_widgets.extend([amp_row, amp_sw])

        # Codec Quality (CRF) Slider
        # INSIDE expander but NOT affected by Auto-Configure toggle (not in _advanced_widgets)

        # Separator before CRF
        crf_sep = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        crf_sep.set_margin_top(SPACING_SM)
        expander.add_row(crf_sep)

        # CRF row with slider
        crf_row = Adw.ActionRow(
            title="Codec Quality (CRF)",
            subtitle="H.264/HEVC quality (0=lossless, 18=high, 23=default, 51=worst). Grayed for ProRes."
        )
        crf_row.set_subtitle_lines(2)  # Allow 2-line subtitle

        # Build CRF slider + value label
        crf_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=SPACING_SM)

        # Value label
        crf_val_lbl = Gtk.Label(label=str(self._cfg.crf_value))
        crf_val_lbl.add_css_class("dim-label")
        crf_val_lbl.set_halign(Gtk.Align.END)
        crf_val_lbl.set_xalign(1.0)
        crf_val_lbl.set_size_request(40, -1)
        crf_val_lbl.set_valign(Gtk.Align.CENTER)
        self._crf_value_label = crf_val_lbl

        # CRF slider: 0-51 range
        crf_slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL,
            0,  # min (lossless)
            51,  # max (worst quality)
            1    # step
        )
        crf_slider.set_value(self._cfg.crf_value)
        crf_slider.set_draw_value(False)
        crf_slider.set_hexpand(True)
        crf_slider.set_valign(Gtk.Align.CENTER)

        def on_crf_changed(slider):
            val = int(slider.get_value())
            self._cfg.crf_value = val
            crf_val_lbl.set_label(str(val))
            self._cfg.save()

        crf_slider.connect("value-changed", on_crf_changed)
        self._crf_slider = crf_slider

        crf_hbox.append(crf_slider)
        crf_hbox.append(crf_val_lbl)
        crf_row.add_suffix(crf_hbox)

        self._crf_row = crf_row
        expander.add_row(crf_row)  # NOTE: NOT in _advanced_widgets -> not grayed by auto-toggle

        grp.add(expander)

        # Initialize CRF sensitivity based on current codec
        GLib.idle_add(lambda: self._update_crf_sensitivity())

        # Initialize state: if auto is ON, gray out controls
        GLib.idle_add(lambda: self._on_auto_config_toggled(None, None))

        return grp

    def _on_auto_config_toggled(self, switch, pspec):
        """Handle auto-configure toggle: gray out/enable manual controls."""
        if not hasattr(self, '_auto_config_switch'):
            return

        is_auto = self._auto_config_switch.get_active()

        for widget in getattr(self, '_advanced_widgets', []):
            widget.set_sensitive(not is_auto)

            # Additional visual feedback: reduce opacity when disabled
            if is_auto:
                widget.set_opacity(0.5)
            else:
                widget.set_opacity(1.0)

        if is_auto:
            self._log("Auto-configure ON: using hardware-detected optimal settings")
        else:
            self._log("Auto-configure OFF: using manual AI settings")

        # Save the toggle state
        self._cfg.auto_config = is_auto
        self._cfg.save()

    def _on_show_detected_settings(self, *_a) -> None:
        """Show a dialog with hardware-detected recommended settings."""
        hw = HardwareProfile.detect()
        ncnn = hw.adaptive_ncnn_params(2)

        # Build readable lines
        lines = []
        lines.append(f"Hardware: {hw.gpu_name}")
        if hw.vram_total_mb > 0:
            vram_gb = hw.vram_total_mb / 1024
            lines.append(f"VRAM: {vram_gb:.1f} GB" if vram_gb >= 1 else f"VRAM: {hw.vram_total_mb} MB")
        lines.append(f"Tier: {hw.tier.name}")
        lines.append("")
        lines.append(f"VSR Batch Size: {hw.vsr_batch_size}")
        lines.append(f"Backbone Blocks: {hw.vsr_backbone_blocks}")
        lines.append(f"NCNN Tile Size: {hw.ncnn_tile_size}")
        lines.append(f"NCNN Jobs: {hw.ncnn_jobs}")
        lines.append(f"FP16: {'On' if hw.use_fp16 else 'Off'}")
        lines.append(f"AMP: {'On' if hw.amp_enabled else 'Off'}")

        # Vendor/backend info
        lines.append("")
        backends = []
        if hw.has_cuda:
            backends.append("PyTorch CUDA")
        if hw.has_mps:
            backends.append("Apple Metal")
        if hw.has_vulkan:
            backends.append("Vulkan (NCNN)")
        if not backends:
            backends.append("CPU only")
        lines.append(f"Backends: {', '.join(backends)}")

        body = "\n".join(lines)

        dlg = Adw.MessageDialog(
            transient_for=self,
            heading="Auto-Detected Settings",
            body=body,
        )
        dlg.add_response("close", "Close")
        dlg.connect("response", lambda d, *a: d.destroy())
        dlg.present()

    def _maybe_show_gpu_warning(self) -> None:
        """Show 'No GPU Found' warning if user hasn't dismissed it before."""
        if self._cfg.suppress_gpu_warning:
            return

        # Build dialog
        dlg = Adw.MessageDialog(
            transient_for=self,
            heading="No GPU Found by PyTorch",
            body=(
                "PyTorch couldn't find a CUDA or Metal GPU for BasicVSR++ temporal processing.\n\n"
                "Note: Vulkan-based upscaling (Real-CUGAN/ESRGAN) may still work \n"
                "if you have an AMD, Intel, or other Vulkan-capable GPU.\n\n"
                "You can:"
                "\n- Install the PyTorch backend matching your GPU (CUDA for NVIDIA, etc.)"
                "\n- Switch temporal processing to FFmpeg for faster CPU-only mode."
            ),
        )
        dlg.add_response("continue", "Continue")
        dlg.set_response_appearance("continue", Adw.ResponseAppearance.DESTRUCTIVE)


        # "Don't show this again" checkbox
        cb = Gtk.CheckButton(label="Don't show this again")
        cb.set_halign(Gtk.Align.CENTER)
        cb.set_margin_top(12)
        cb.connect("toggled", self._on_suppress_gpu_warning_toggled)
        dlg.set_extra_child(cb)

        dlg.connect("response", lambda d, *a: d.destroy())
        dlg.present()

    def _on_suppress_gpu_warning_toggled(self, cb: Gtk.CheckButton) -> None:
        """Persist the user's 'don't show again' choice."""
        self._cfg.suppress_gpu_warning = cb.get_active()
        self._cfg.save()

    # Callbacks

    def _on_open(self, *_a) -> None:
        dlg = Gtk.FileDialog(title="Open Video", accept_label="Open")
        filters = Gio.ListStore(item_type=Gtk.FileFilter)
        f = Gtk.FileFilter(name="Videos")
        for ext in VIDEO_EXTS: f.add_pattern(f"*{ext}")
        filters.append(f)
        dlg.set_default_filter(f)
        dlg.open(self, None, self._open_cb)

    def _open_cb(self, dlg, result) -> None:
        try:
            f = dlg.open_finish(result)
            if f:
                self._load_input(f.get_path())
        except Exception:
            pass  # cancelled

    def _load_input(self, path: str) -> None:
        """Load video info in background thread so UI doesn't freeze."""
        self._status.set_text("Loading video...")
        t = threading.Thread(target=self._probe_video_bg, args=(path,), daemon=True)
        t.start()

    def _probe_video_bg(self, path: str) -> None:
        """Background thread: probe video and update UI."""
        try:
            info = probe_video(Path(path))
            GLib.idle_add(self._on_video_loaded, path, info)
        except Exception as e:
            GLib.idle_add(self._show_err, f"Couldn't load video: {e}")

    def _on_video_loaded(self, path: str, info: VideoInfo) -> None:
        """Called after successful background probe."""
        self._info = info
        self._cfg.input_path = path
        self._in_row.set_subtitle(os.path.basename(path))
        self._status.set_text(f"Loaded: {info.res} @ {info.fps:.1f}fps, {info.dur}")
        self._refresh_dims()
        self._update_state()

    def _on_save(self, *_a) -> None:
        dlg = Gtk.FileDialog(title="Save Output", accept_label="Save")
        filters = Gio.ListStore(item_type=Gtk.FileFilter)
        f = Gtk.FileFilter(name="Video")
        for ext in [".mp4", ".mkv", ".mov"]: f.add_pattern(f"*{ext}")
        filters.append(f)
        dlg.set_default_filter(f)
        dlg.save(self, None, self._save_cb)

    def _save_cb(self, dlg, result) -> None:
        try:
            f = dlg.save_finish(result)
            if f:
                p = f.get_path()
                self._cfg.output_path = p
                self._out_row.set_subtitle(os.path.basename(p))
                self._status.set_text(f"Output: {os.path.basename(p)}")
                self._update_state()
                self._cfg.save()  # Persist output path
        except Exception:
            pass

    def _on_scale_changed(self, slider) -> None:
        v = int(slider.get_value())
        self._cfg.scale_pct = v
        self._scale_val.set_label(f"{v}%")
        self._refresh_dims()
        self._cfg.save()  # Persist scale setting immediately

    def _on_descale_toggled(self, sw, *a) -> None:
        self._cfg.descale = sw.get_active()
        self._refresh_dims()
        self._cfg.save()  # Persist descale setting

    def _refresh_dims(self) -> None:
        if not self._info:
            self._dim_lbl.set_label("Load a video first")
            return
        pct = self._cfg.scale_pct
        tw, th = target_dims(self._info.width, self._info.height, pct)
        if pct == 100 and not self._cfg.descale:
            self._dim_lbl.set_label(f"Output: {self._info.res} (native)")
        elif self._cfg.descale:
            self._dim_lbl.set_label(f"Process at: {tw}x{th} -> Descale to: {self._info.res}")
        else:
            self._dim_lbl.set_label(f"Output: {tw}x{th}")

    def _on_codec_changed(self, combo, *a) -> None:
        idx = combo.get_selected()
        codecs = list(Codec)
        if 0 <= idx < len(codecs):
            self._cfg.codec = codecs[idx]
            self._log(f"codec: {combo.get_selected_item().get_string()}")
            self._cfg.save()  # Persist codec setting immediately
            # Update CRF slider sensitivity (disabled for ProRes)
            self._update_crf_sensitivity()

    def _update_crf_sensitivity(self) -> None:
        """Enable/disable CRF slider based on selected codec.

        H.264 and HEVC support CRF encoding.
        ProRes uses fixed bitrate profiles (not CRF-based).
        """
        if not hasattr(self, '_crf_row'):
            return

        is_prores = (self._cfg.codec == Codec.PRORES)

        # Gray out row and disable slider for ProRes
        self._crf_row.set_sensitive(not is_prores)
        if hasattr(self, '_crf_slider'):
            self._crf_slider.set_sensitive(not is_prores)

        # Visual feedback
        if is_prores:
            self._crf_row.set_opacity(0.5)
            self._crf_row.set_subtitle("ProRes uses fixed bitrate -- CRF not applicable")
        else:
            self._crf_row.set_opacity(1.0)
            self._crf_row.set_subtitle(f"Quality for {self._cfg.codec.name} (0=lossless, 18=high, 23=default, 51=worst)")

    def _update_scale_max(self) -> None:
        """Dynamically cap the scale slider based on upscaler model and CUGAN tier.

        Real-CUGAN tier scale support:
          - SE:   2x/3x/4x -> max 400%
          - Pro:  2x/3x    -> max 300%
          - Nose: 2x       -> max 200%
        Real-ESRGAN: supports 2x/3x/4x -> max 400%
        """
        upscaler = getattr(self._cfg, 'upscaler_model', 'cugan')
        if upscaler == 'esrgan':
            max_pct = 400
        else:
            tier = getattr(self._cfg, 'cugan_tier', 'se')
            max_pct = {"se": 400, "pro": 300, "nose": 200}.get(tier, 400)

        slider = getattr(self, '_scale_slider', None)
        if slider:
            slider.set_range(100, max_pct)
            # If current value exceeds new max, clamp it down
            current = int(slider.get_value())
            if current > max_pct:
                slider.set_value(max_pct)
                self._cfg.scale_pct = max_pct
                self._scale_val.set_label(f"{max_pct}%")
                self._refresh_dims()
                self._cfg.save()

    def _on_upscaler_changed(self, combo, *a) -> None:
        """Handle upscaler model selection change."""
        idx = combo.get_selected()
        labels = ["cugan", "esrgan"]
        descs = [
            "Anime-styled cartoonish upscaling.",
            "Realism-focused upscaling.\nMay produce oversharpened artifacts.",
        ]
        if 0 <= idx < len(labels):
            self._cfg.upscaler_model = labels[idx]
            combo.set_subtitle(descs[idx])
            self._cfg.save()
            self._update_scale_max()
            # Model tier and post-denoise only apply to Real-CUGAN
            is_cugan = self._cfg.upscaler_model == "cugan"
            if hasattr(self, '_cugan_tier_combo'):
                self._cugan_tier_combo.set_visible(is_cugan)
            if hasattr(self, '_post_denoise_row'):
                self._post_denoise_row.set_visible(is_cugan and self._cfg.cugan_tier == "nose")

    def _on_temporal_changed(self, combo, *a) -> None:
        """Handle temporal processing method selection change."""
        idx = combo.get_selected()
        methods = [TemporalMethod.BASICVSR, TemporalMethod.FFMPEG]
        if 0 <= idx < len(methods):
            self._cfg.temporal_method = methods[idx]
            method_name = "BasicVSR++ (AI)" if methods[idx] == TemporalMethod.BASICVSR else "FFmpeg"
            self._log(f"temporal method: {method_name}")
            self._cfg.save()  # Persist temporal method setting

    def _on_deblock(self, v: float) -> None:
        self._cfg.deblock = int(v)
        self._cfg.save()

    def _on_denoise(self, v: float) -> None:
        self._cfg.denoise = int(v)
        self._cfg.save()

    def _on_sharpen(self, v: float) -> None:
        self._cfg.sharpen = int(v)
        self._cfg.save()

    def _on_deblur(self, v: float) -> None:
        self._cfg.deblur = int(v)
        self._cfg.save()

    def _on_aa(self, v: float) -> None:
        self._cfg.antialias = int(v)
        self._cfg.save()

    # Advanced Settings Change Handlers

    def _on_vsr_batch_changed(self, spin) -> None:
        self._cfg.vsr_batch = int(spin.get_value())
        self._cfg.save()

    def _on_vsr_blocks_changed(self, spin) -> None:
        self._cfg.vsr_blocks = int(spin.get_value())
        self._cfg.save()

    def _on_ncnn_tile_toggled(self, sw, *a) -> None:
        """Handle custom tile size toggle: show/hide the slider row."""
        is_manual = sw.get_active()  # ON = manual, OFF = auto
        self._cfg.ncnn_tile_auto = not is_manual
        self._cfg.save()
        # Show/hide the tile slider row cleanly
        if hasattr(self, '_ncnn_tile_row'):
            self._ncnn_tile_row.set_visible(is_manual)

    def _on_ncnn_tile_changed(self, spin) -> None:
        self._cfg.ncnn_tile = int(spin.get_value())
        self._cfg.save()

    def _on_ncnn_jobs_changed(self, entry) -> None:
        self._cfg.ncnn_jobs = entry.get_text()
        self._cfg.save()

    def _on_cugan_tier_changed(self, combo, *a) -> None:
        idx = combo.get_selected()
        tier_map = {0: "se", 1: "pro", 2: "nose"}
        self._cfg.cugan_tier = tier_map.get(idx, "se")
        self._cfg.save()
        self._update_scale_max()
        # Show post-denoise toggle only for NOSE tier (and only when CUGAN is active)
        if hasattr(self, '_post_denoise_row'):
            is_cugan = self._cfg.upscaler_model == "cugan"
            self._post_denoise_row.set_visible(is_cugan and self._cfg.cugan_tier == "nose")

    def _on_post_denoise_toggled(self, sw, *a) -> None:
        self._cfg.post_denoise = sw.get_active()
        self._cfg.save()

    def _on_fp16_toggled(self, sw, *a) -> None:
        self._cfg.use_fp16 = sw.get_active()
        self._cfg.save()

    def _on_amp_toggled(self, sw, *a) -> None:
        self._cfg.use_amp = sw.get_active()
        self._cfg.save()

    def _on_enhance(self, *_a) -> None:
        if not self._cfg.input_path:
            self._show_err("Pick an input video first")
            return
        if not self._cfg.output_path:
            self._show_err("Pick where to save output")
            return
        if self._busy:
            return  # Guard against double-start

        in_resolved = Path(self._cfg.input_path).resolve()
        out_resolved = Path(self._cfg.output_path).resolve()

        # Same-file dialog: input and output point to the same path
        if in_resolved == out_resolved:
            stem = out_resolved.stem
            ext = out_resolved.suffix or ".mov"
            temp_name = f"{stem}_sharptaped{ext}"
            dlg = Adw.MessageDialog(
                transient_for=self,
                heading="Input and output are the same file",
                body=(
                    f"'{out_resolved.name}' is both input and output.\n\n"
                    f"Choose how to handle this:"
                ),
            )
            dlg.add_response("cancel", "Cancel")
            dlg.add_response("save_as", f"Save to {temp_name}")
            dlg.add_response("overwrite", "Overwrite original")
            dlg.set_response_appearance("overwrite", Adw.ResponseAppearance.DESTRUCTIVE)
            dlg.connect("response", self._on_same_path_response)
            # Stash the temp output path on the dialog for the handler
            dlg._sharptape_temp_output = str(out_resolved.parent / temp_name)
            dlg._sharptape_original_output = self._cfg.output_path
            dlg.present()
            return

        # Different-file overwrite dialog (existing behaviour, unchanged)
        if Path(self._cfg.output_path).exists():
            dlg = Adw.MessageDialog(
                transient_for=self,
                heading="Overwrite existing file?",
                body=f"'{os.path.basename(self._cfg.output_path)}' already exists. Replace it?",
            )
            dlg.add_response("cancel", "Cancel")
            dlg.add_response("overwrite", "Overwrite")
            dlg.set_response_appearance("overwrite", Adw.ResponseAppearance.DESTRUCTIVE)
            dlg.connect("response", self._on_overwrite_response)
            dlg.present()
            return

        self._start_processing()

    def _on_overwrite_response(self, dlg, response: str) -> None:
        if response == "overwrite":
            self._start_processing()

    def _on_same_path_response(self, dlg, response: str) -> None:
        """Handle the same-input-output-path dialog."""
        if response == "cancel":
            return  # just close the dialog, window stays
        if response == "save_as":
            # Process to _sharptaped file, no atomic swap
            self._cfg.output_path = dlg._sharptape_temp_output
            self._atomic_overwrite_original = None
            self._start_processing()
        elif response == "overwrite":
            # Process to _sharptaped temp file, then atomically swap on success
            self._cfg.output_path = dlg._sharptape_temp_output
            self._atomic_overwrite_original = dlg._sharptape_original_output
            self._start_processing()

    def _start_processing(self) -> None:
        """Actually kick off the worker thread."""
        self._busy = True
        self._cancel.clear()
        self._start_ui()

        self._worker = Worker(self._cfg, self._info, self._cancel)
        self._worker.connect("progress", self._on_prog)
        self._worker.connect("done", self._on_done)
        self._worker.connect("error", self._on_err)
        self._worker.connect("log", self._on_worker_log)
        self._worker.connect("debug", self._on_worker_debug)  # Console-only logs
        self._worker.connect("toast", self._on_worker_toast)
        self._worker.start()

    def _on_cancel(self, *_a) -> None:
        self._cancel.set()
        self._status.set_text("Cancelling...")

    def _on_prog(self, worker: 'Worker', frac: float, msg: str) -> None:
        self._prog_bar.set_fraction(frac)
        self._status.set_text(msg)

    def _on_done(self, worker: 'Worker', success: bool, msg: str) -> None:
        self._restore_ui()
        if success:
            # Atomic overwrite: swap temp file -> original (same-path mode)
            if self._atomic_overwrite_original:
                try:
                    temp_path = Path(self._cfg.output_path)
                    orig_path = Path(self._atomic_overwrite_original)
                    if temp_path.exists():
                        orig_path.unlink(missing_ok=True)
                        temp_path.rename(orig_path)
                        self._cfg.output_path = str(orig_path)
                except OSError as e:
                    self._show_err(f"Couldn't replace original: {e}")
                self._atomic_overwrite_original = None

            output_name = os.path.basename(self._cfg.output_path)
            self._status.set_text(f"Done! -> {output_name}")

            # Show in-window toast notification (stays inside window!)
            toast = Adw.Toast.new("Video restored successfully!")
            toast.set_button_label("Open File")
            toast.connect("button-clicked", lambda *a: self._open_output_file())
            self._toast_overlay.add_toast(toast)

            # Send desktop notification (system-level)
            self._send_desktop_notification(
                "Sharptape Complete",
                f"Video saved to {output_name}",
                success=True
            )
        elif msg == "cancelled":
            self._status.set_text("Cancelled")
            toast = Adw.Toast.new("Processing cancelled")
            toast.set_timeout(3)
            self._toast_overlay.add_toast(toast)
            # Clean up temp file if atomic overwrite was planned
            self._cleanup_atomic_overwrite()
        else:
            self._status.set_text(msg)
            toast = Adw.Toast.new("Processing completed with warnings")
            toast.set_timeout(5)
            self._toast_overlay.add_toast(toast)
            self._cleanup_atomic_overwrite()

    def _cleanup_atomic_overwrite(self) -> None:
        """Clean up temp file and reset atomic overwrite state on failure/cancel."""
        if self._atomic_overwrite_original:
            # If the temp file was created, remove it so the original stays untouched
            try:
                Path(self._cfg.output_path).unlink(missing_ok=True)
            except OSError:
                pass
            # Restore the original output path in the UI
            self._cfg.output_path = self._atomic_overwrite_original
            self._atomic_overwrite_original = None

    def _open_output_file(self) -> None:
        """Open the output file with the system's default MIME handler.

        Uses start_new_session=True so the opened process (e.g. nautilus,
        video player) is fully detached -- closing Sharptape will NOT kill it.
        Opens the FILE directly, not the folder, so it launches the video
        player instead of the file manager.
        """
        output_path = Path(self._cfg.output_path)
        if not output_path.exists():
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(output_path)],
                                start_new_session=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", str(output_path)],
                                start_new_session=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            else:  # Linux / BSD
                subprocess.Popen(["xdg-open", str(output_path)],
                                start_new_session=True,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        except Exception as e:
            self._log(f"Could not open file: {e}")

    def _send_desktop_notification(self, title: str, body: str, success: bool = True) -> None:
        """Send a desktop notification using Gio.Notification or fallback."""
        try:
            # Try GLib/Gio notification first (works on GNOME/KDE/etc.)
            notif = Gio.Notification.new(title)
            notif.set_body(body)

            # Set icon based on success/failure
            if success:
                notif.set_icon(Gio.ThemedIcon.new("emblem-default-symbolic"))
            else:
                notif.set_icon(Gio.ThemedIcon.new("dialog-warning-symbolic"))

            # Set priority
            notif.set_priority(Gio.NotificationPriority.NORMAL)

            # Send via application
            app = self.get_application()
            if app:
                app.send_notification(None, notif)  # None = default ID
        except Exception as e:
            # Fallback: try notify-send on Linux
            self._log(f"Desktop notification failed: {e}")
            try:
                icon = "--icon=emblem-default" if success else "--icon=dialog-warning"
                subprocess.run(
                    ["notify-send", icon, title, body],
                    check=False,
                    timeout=2
                )
            except Exception:
                pass  # Silently fail - notification is optional

    def _on_err(self, worker: 'Worker', msg: str) -> None:
        self._restore_ui()
        self._cleanup_atomic_overwrite()

        # Truncate very long error messages for display
        display_msg = msg[:500] if len(msg) <= 500 else msg[:497] + "..."

        # Build a human-readable model label for the error dialog
        upscaler = getattr(self._cfg, 'upscaler_model', 'cugan')
        model_label = "Real-CUGAN" if upscaler == "cugan" else "Real-ESRGAN"
        temporal = getattr(self._cfg, 'temporal_method', TemporalMethod.BASICVSR)
        temporal_label = "BasicVSR++ (AI)" if temporal == TemporalMethod.BASICVSR else "FFmpeg"
        pipeline = f"{model_label} + {temporal_label}"

        dlg = Adw.MessageDialog(
            transient_for=self,
            heading=f"Processing Failed ({pipeline})",
            body=f"An error occurred during video processing:\n\n{display_msg}",
        )
        dlg.add_response("close", "Cancel")

        dlg.connect("response", lambda d, *a: d.destroy())
        dlg.present()

        self._send_desktop_notification("Sharptape Error", msg[:200], success=False)

    def _on_worker_log(self, worker: 'Worker', msg: str) -> None:
        """Handle user-facing log messages (shown in UI status bar)."""
        if hasattr(self, '_status'):
            self._status.set_text(msg)

    def _on_worker_debug(self, worker: 'Worker', msg: str) -> None:
        """Handle debug log messages (console only, NOT shown in UI)."""
        pass

    def _start_ui(self) -> None:
        self._enhance_btn.set_sensitive(False)
        self._enhance_btn.set_visible(False)
        self._cancel_btn.set_visible(True)
        self._spinner.start(); self._spinner.set_visible(True)
        self._prog_bar.set_visible(True)
        self._set_sensitive(False)

    def _restore_ui(self) -> None:
        self._busy = False
        self._enhance_btn.set_sensitive(True)
        self._enhance_btn.set_visible(True)
        self._cancel_btn.set_visible(False)
        self._spinner.stop(); self._spinner.set_visible(False)
        self._prog_bar.set_visible(False)
        self._set_sensitive(True)
        self._update_state()

    def _set_sensitive(self, on: bool) -> None:
        self._scale_slider.set_sensitive(on)
        for r in [self._row_deblock, self._row_denoise, self._row_sharpen, self._row_deblur, self._row_aa]:
            s = getattr(r, "_slider", None)
            if s: s.set_sensitive(on)

    def _update_state(self) -> None:
        ok = (bool(self._cfg.input_path) and bool(self._cfg.output_path)
              and Path(self._cfg.input_path).is_file() and not self._busy)
        self._enhance_btn.set_sensitive(ok)

    def _show_err(self, msg: str) -> None:
        """Show error as in-window toast (not popup dialog)."""
        # Truncate long messages for display
        display_msg = msg if len(msg) <= 100 else msg[:97] + "..."

        toast = Adw.Toast.new(display_msg)
        toast.set_timeout(5)
        self._toast_overlay.add_toast(toast)

    def _on_worker_toast(self, worker: 'Worker', msg: str) -> None:
        """Handle toast signal from the worker thread."""
        toast = Adw.Toast.new(msg)
        toast.set_timeout(5)
        self._toast_overlay.add_toast(toast)


# --- App Class ---

class App(Adw.Application):
    """GTK4 Application with single-instance support.

    Uses D-Bus application ID to ensure only one instance runs.
    When a second instance is launched, it focuses the existing window
    and exits cleanly.
    """
    def __init__(self) -> None:
        # Use HANDLES_OPEN flag for single-instance via D-Bus
        # When second instance starts, GTK forwards to existing instance's do_open()
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.HANDLES_OPEN)
        GLib.set_application_name("Sharptape")
        self._primary_window = None  # Track the main window

        quit_act = Gio.SimpleAction.new("quit", None)
        quit_act.connect("activate", lambda *a: self.quit())
        self.add_action(quit_act)
        self.set_accels_for_action("app.quit", ["<Control>Q"])

        open_act = Gio.SimpleAction.new("open", None)
        open_act.connect("activate", lambda *a: self._do_open())
        self.add_action(open_act)
        self.set_accels_for_action("app.open", ["<Control>O"])

    def do_activate(self) -> None:
        """Called when app is activated (first launch or from dock).

        For single-instance: if window already exists, focus it and quit this instance.
        For first launch: create and show main window.
        """
        # If we already have a window (second instance), focus it and exit
        if self._primary_window:
            self._log("Single-instance: focusing existing window and exiting")
            self._primary_window.present()
            self._present_with_urgency()
            # CRITICAL: Quit this second instance after focusing the primary
            self.quit()
            return

        win = Window(self)
        self._primary_window = win
        win.present()

    def do_window_removed(self, window):
        """Called when a window is destroyed.

        Reset primary_window reference and quit application when last window closes.
        CRITICAL: Must explicitly quit because HANDLES_OPEN flag keeps D-Bus alive.
        """
        if window == self._primary_window:
            self._primary_window = None
        # Explicitly quit -- HANDLES_OPEN keeps D-Bus alive after window closes
        # The HANDLES_OPEN flag creates a D-Bus connection that keeps the process alive
        # even after all windows are destroyed. Without this, the process hangs.
        self.quit()

    def _present_with_urgency(self) -> None:
        """Try multiple methods to bring window to front."""
        if not self._primary_window:
            return
        try:
            # Method 1: Present and activate (standard GTK4 way)
            self._primary_window.present()
            self._primary_window.activate()

            # Method 2: Try GdkWindow hint if available (X11)
            surface = self._primary_window.get_surface()
            if surface:
                surface.raise_()  # X11 specific, may not work on Wayland
                # Set urgency hint (taskbar flash)
                if hasattr(surface, 'set_urgency_hint'):
                    surface.set_urgency_hint(True)
        except Exception:
            pass  # Best effort - don't crash if focus fails

    def do_open(self, files, n_files, hint) -> None:
        """Called when second instance tries to open files or activate.

        GTK4 D-Bus single-instance: this runs in the PRIMARY instance,
        not the one that was just launched.
        """
        # Ensure we have a window and present it
        self.do_activate()

        w = self.get_active_window()
        if w and n_files > 0:
            p = files[0].get_path()
            if p and any(p.lower().endswith(e) for e in VIDEO_EXTS):
                w._load_input(p)

    def _do_open(self) -> None:
        w = self.get_active_window()
        if w: w._on_open()

    def _check_required_models(self) -> Tuple[bool, List[str]]:
        """Verify that all required model files exist on disk."""
        missing = []

        # 1. basicvsr++ model
        basicvsr_path = MODEL_DIR / "basicvsr" / BASICVSR_MODEL_FILENAME
        if not basicvsr_path.is_file():
            missing.append("BasicVSR++ (.pth)")

        # 2. real-cugan model directories (must contain actual .bin files)
        cugan_base = MODEL_DIR / "real-cugan"
        cugan_found = False
        for subdir in ["models-se", "models-pro", "models-nose"]:
            cugan_subdir = cugan_base / subdir
            if cugan_subdir.is_dir() and list(cugan_subdir.glob("*.bin")):
                cugan_found = True
                break
        if not cugan_found:
            missing.append("Real-CUGAN (models-se/pro/nose with *.bin files)")

        # 3. realesrgan model files (must have .bin files)
        esrgan_dir = MODEL_DIR / "realesrgan"
        if not esrgan_dir.is_dir() or not list(esrgan_dir.glob("*.bin")):
            missing.append("Real-ESRGAN (*.bin)")

        return len(missing) == 0, missing

    def _about(self, *_a) -> None:
        dlg = Adw.AboutDialog(
            application_name="Sharptape",
            application_icon="net.buwryy.Sharptape",
            version=__version__,
            copyright="© 2025 buwryme",
            license_type=Gtk.License.APACHE_2_0,
            developer_name="buwryme",
            developers=[
                "buwryme",
            ],
        )
        dlg.present()


# --- Entry Point ---

def main():
    """Main entry point with single-instance handling."""
    # Initialize multi-language support (auto-detects system locale)
    init_languages()
    
    Adw.init()
    app = App()

    # NOTE: Do NOT call app.hold() - it prevents the app from exiting!
    # GTK4 default behavior: exits when last window closes (what we want)

    # Ensure clean exit on signals
    def _signal_handler(signum, frame):
        print(f"[sharptape] Received signal {signum}, quitting...")

        # Kill any child processes first
        try:
            my_pid = os.getpid()

            if sys.platform.startswith('linux'):
                try:
                    result = subprocess.run(
                        ['pgrep', '-P', str(my_pid)],
                        capture_output=True, text=True, timeout=1
                    )
                    if result.stdout.strip():
                        for pid_str in result.stdout.strip().split('\n'):
                            try:
                                os.kill(int(pid_str), signal.SIGKILL)  # Force kill on signal
                            except (ProcessLookupError, ValueError):
                                pass
                except Exception:
                    pass

            # Also try process group
            try:
                os.killpg(os.getpgid(0), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        except Exception:
            pass

        app.quit()
        exit(1)  # Force exit if quit() doesn't work

    signal.signal(signal.SIGINT, _signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, _signal_handler)  # kill

    # Check for required models before showing window
    ok, missing = app._check_required_models()
    if not ok:
        missing_str = "\n- " + "\n- ".join(missing)
        error_msg = f"Missing required model files:\n{missing_str}\n\nHave you run the setup script?"

        print(f"[sharptape] ERROR: {error_msg}")

        # Show a GTK dialog to inform the user, then exit with error
        # Note: Gtk/Adw already imported at module level - don't re-import here!

        # Create a simple error dialog
        dialog = Adw.MessageDialog()
        dialog.set_heading("Missing Model Files")
        dialog.set_body(error_msg)
        dialog.add_response("quit", "Quit")
        dialog.set_response_appearance("quit", Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.connect("response", lambda *a: exit(1))
        dialog.present()

        # Run the dialog in its own mini-loop
        # Note: GLib already imported at module level (line 38)
        GLib.MainLoop().run()

    # Run the application - this will either:
    # 1. Start as primary instance (first launch)
    # 2. Forward to existing instance and exit (second launch)
    rc = app.run(sys.argv)
    sys.exit(rc)

if __name__ == "__main__":
    main()
