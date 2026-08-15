#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-language support system for Sharptape.

Auto-detects system locale from Linux/Freedesktop environment variables
and loads appropriate language file.

Supports: en_US, en_GB, ru_RU, sv_SE, de_DE, fr_FR, and more
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional

# Default language file
DEFAULT_LANGUAGE = "en_US"


def detect_system_locale() -> str:
    """
    Detect system locale from Linux/Freedesktop environment variables.
    
    Priority order (standard Linux locale detection):
    1. LC_ALL (highest priority, but rarely set)
    2. LC_MESSAGES (messages/UI language)
    3. LANG (fallback to system language)
    4. LANGUAGE (list of fallback languages)
    
    Returns locale code like 'en_US', 'de_DE', 'ru_RU', etc.
    Falls back to en_US if unable to detect.
    """
    locale_code = None
    
    # Check LC_ALL (highest priority)
    lc_all = os.getenv("LC_ALL", "").strip()
    if lc_all and lc_all != "C":
        locale_code = _parse_locale_code(lc_all)
        if locale_code:
            return locale_code
    
    # Check LC_MESSAGES (UI language)
    lc_messages = os.getenv("LC_MESSAGES", "").strip()
    if lc_messages:
        locale_code = _parse_locale_code(lc_messages)
        if locale_code:
            return locale_code
    
    # Check LANG (system default)
    lang = os.getenv("LANG", "").strip()
    if lang and lang != "C":
        locale_code = _parse_locale_code(lang)
        if locale_code:
            return locale_code
    
    # Check LANGUAGE (fallback list, colon-separated)
    language = os.getenv("LANGUAGE", "").strip()
    if language:
        # LANGUAGE can contain multiple locales separated by colons
        for loc in language.split(":"):
            loc = loc.strip()
            if loc:
                locale_code = _parse_locale_code(loc)
                if locale_code:
                    return locale_code
    
    # Fallback to English US
    return DEFAULT_LANGUAGE


def _parse_locale_code(locale_str: str) -> Optional[str]:
    """
    Parse locale string and extract language_COUNTRY code.
    
    Handles formats like:
    - en_US.UTF-8 -> en_US
    - de_DE -> de_DE
    - ru_RU.utf8 -> ru_RU
    - en (just language) -> en_US (returns with country fallback)
    
    Returns None if unable to parse.
    """
    if not locale_str:
        return None
    
    # Remove encoding suffix (e.g., .UTF-8, .utf8)
    if "." in locale_str:
        locale_str = locale_str.split(".")[0]
    
    # Handle language_COUNTRY format (e.g., en_US, de_DE)
    if "_" in locale_str and len(locale_str) >= 5:
        parts = locale_str.split("_")
        if len(parts[0]) == 2 and len(parts[1]) == 2:
            return f"{parts[0].lower()}_{parts[1].upper()}"
    
    # Handle hyphenated format (e.g., en-US) - convert to underscore
    if "-" in locale_str and len(locale_str) >= 5:
        locale_str = locale_str.replace("-", "_")
        parts = locale_str.split("_")
        if len(parts[0]) == 2 and len(parts[1]) == 2:
            return f"{parts[0].lower()}_{parts[1].upper()}"
    
    return None


class LanguageManager:
    """Manages language translations for the application."""
    
    SUPPORTED_LANGUAGES = {
        "en_US": "English (US)",
        "en_GB": "English (UK)",
        "ru_RU": "Русский (Россия)",
        "sv_SE": "Svenska (Sverige)",
        "de_DE": "Deutsch (Deutschland)",
        "fr_FR": "Français (France)",
        "es_ES": "Español (España)",
    }
    
    def __init__(self, language_code: Optional[str] = None):
        """
        Initialize language manager.
        
        Args:
            language_code: Specific language to load (e.g., 'en_US').
                          If None, auto-detects from system locale.
        """
        self.current_language = language_code or detect_system_locale()
        self.translations: Dict[str, str] = {}
        self._load_language()
    
    def _load_language(self) -> None:
        """Load language file for current language."""
        lang_file = self._get_language_file_path(self.current_language)
        
        if lang_file and lang_file.exists():
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self.translations = json.load(f)
                print(f"[sharptape:i18n] loaded language: {self.current_language}")
                return
            except (IOError, json.JSONDecodeError) as e:
                print(f"[sharptape:i18n] error loading {lang_file}: {e}")
        
        # Fallback to English if language not available
        if self.current_language != DEFAULT_LANGUAGE:
            print(f"[sharptape:i18n] falling back to {DEFAULT_LANGUAGE}")
            self.current_language = DEFAULT_LANGUAGE
            lang_file = self._get_language_file_path(DEFAULT_LANGUAGE)
            if lang_file and lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations = json.load(f)
                    return
                except (IOError, json.JSONDecodeError):
                    pass
        
        print("[sharptape:i18n] WARNING: no language files found, using hardcoded English")
        self.translations = self._get_builtin_english()
    
    @staticmethod
    def _get_language_file_path(language_code: str) -> Optional[Path]:
        """Get path to language file for given language code."""
        # Try multiple possible locations for language files
        possible_paths = [
            Path(__file__).parent / "assets" / "languages" / f"{language_code}.json",
            Path("/usr/share/sharptape/languages") / f"{language_code}.json",
            Path.home() / ".local" / "share" / "sharptape" / "languages" / f"{language_code}.json",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    @staticmethod
    def _get_builtin_english() -> Dict[str, str]:
        """Return hardcoded English strings as fallback."""
        return {
            "app_name": "Sharptape",
            "window_title": "Sharptape",
            "open_video": "Open Video",
            "enhance_video": "Enhance Video",
            "processing": "Processing...",
            "cancel": "Cancel",
            "done": "Done",
            "failed": "Failed",
            "error_during_processing": "An error occurred during video processing:\n\n{error}",
        }
    
    def get(self, key: str, fallback: Optional[str] = None) -> str:
        """
        Get translation for key.
        
        Args:
            key: Translation key (e.g., 'open_video')
            fallback: Fallback text if key not found
        
        Returns:
            Translated string, or fallback/key if not found
        """
        if key in self.translations:
            return self.translations[key]
        
        if fallback:
            return fallback
        
        print(f"[sharptape:i18n] missing translation key: {key}")
        return key
    
    def get_formatted(self, key: str, **kwargs) -> str:
        """
        Get translation and format with keyword arguments.
        
        Args:
            key: Translation key
            **kwargs: Format arguments
        
        Returns:
            Formatted translated string
        """
        text = self.get(key, key)
        try:
            return text.format(**kwargs)
        except KeyError as e:
            print(f"[sharptape:i18n] missing format key {e} in: {text}")
            return text
    
    def set_language(self, language_code: str) -> bool:
        """
        Change active language.
        
        Args:
            language_code: Language code (e.g., 'en_US', 'de_DE')
        
        Returns:
            True if language was successfully changed
        """
        if language_code not in self.SUPPORTED_LANGUAGES:
            print(f"[sharptape:i18n] unsupported language: {language_code}")
            return False
        
        self.current_language = language_code
        self._load_language()
        return True
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Return dict of supported languages {code: display_name}."""
        return self.SUPPORTED_LANGUAGES.copy()


# Global language manager instance
_lang_manager: Optional[LanguageManager] = None


def init_languages(language_code: Optional[str] = None) -> LanguageManager:
    """Initialize global language manager."""
    global _lang_manager
    _lang_manager = LanguageManager(language_code)
    return _lang_manager


def get_lang() -> LanguageManager:
    """Get global language manager instance."""
    global _lang_manager
    if _lang_manager is None:
        _lang_manager = LanguageManager()
    return _lang_manager


def _(key: str, fallback: Optional[str] = None) -> str:
    """
    Translate key using global language manager.
    
    This is a convenience function for UI code:
    
        label.set_text(_("open_video"))
    
    Args:
        key: Translation key
        fallback: Fallback text if not found
    
    Returns:
        Translated string
    """
    return get_lang().get(key, fallback)


def _f(key: str, **kwargs) -> str:
    """
    Translate and format key using global language manager.
    
    Example:
        self._log(_f("extracted_frames", count=42))
    
    Args:
        key: Translation key
        **kwargs: Format arguments
    
    Returns:
        Formatted translated string
    """
    return get_lang().get_formatted(key, **kwargs)
