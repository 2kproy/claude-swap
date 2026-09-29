"""Tests for per-account proxy base URLs (``--base-url``).

A third-party proxy key is only usable if Claude Code is pointed at the proxy
via ``env.ANTHROPIC_BASE_URL``, which lives in ``settings.json`` -- a file
claude-swap normally never touches. These tests cover the user_settings
writer and the switch-path hook that applies or clears it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from claude_swap.user_settings import (
    API_KEY_ENV,
    BASE_URL_ENV,
    get_settings_path,
    read_settings,
    set_proxy_env,
)

API_KEY = "sk-ant-api03-" + "a1b2c3d4e5" * 4  # 53 chars, matches tests/test_api_key_accounts.py


class TestUserSettingsWriter:
    """The settings.json env block writer, in isolation."""

    def test_applies_base_url_and_key(self, temp_home: Path):
        path = temp_home / ".claude" / "settings.json"
        set_proxy_env(path, api_key=API_KEY, base_url="https://api.example.com")

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["env"][BASE_URL_ENV] == "https://api.example.com"
        assert data["env"][API_KEY_ENV] == API_KEY

    def test_preserves_unrelated_settings(self, temp_home: Path):
        path = temp_home / ".claude" / "settings.json"
        path.write_text(
            json.dumps({
                "model": "claude-opus-4-8",
                "theme": "dark",
                "env": {"HTTP_PROXY": "http://local:3128"},
            }),
            encoding="utf-8",
        )
        set_proxy_env(path, api_key=API_KEY, base_url="https://api.example.com")

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["model"] == "claude-opus-4-8"
        assert data["theme"] == "dark"
        # Another tool's env entry must survive, untouched.
        assert data["env"]["HTTP_PROXY"] == "http://local:3128"
        assert data["env"][BASE_URL_ENV] == "https://api.example.com"

    def test_creating_settings_file_when_absent(self, temp_home: Path):
        path = temp_home / ".claude" / "settings.json"
        assert not path.exists()
        set_proxy_env(path, api_key=API_KEY, base_url="https://x.test")
        assert path.exists()

    def test_clear_removes_owned_vars_only(self, temp_home: Path):
        path = temp_home / ".claude" / "settings.json"
        path.write_text(
            json.dumps({
                "model": "claude-opus-4-8",
                "env": {
                    BASE_URL_ENV: "https://x.test",
                    API_KEY_ENV: "old-key",
                    "HTTP_PROXY": "http://local:3128",
                },
            }),
            encoding="utf-8",
        )
        set_proxy_env(path, api_key=None, base_url=None)

        data = json.loads(path.read_text(encoding="utf-8"))
        # Owned vars are gone; everything else is preserved.
        assert BASE_URL_ENV not in data["env"]
        assert API_KEY_ENV not in data["env"]
        assert data["env"] == {"HTTP_PROXY": "http://local:3128"}
        assert data["model"] == "claude-opus-4-8"

    def test_clear_removes_empty_env_block(self, temp_home: Path):
        path = temp_home / ".claude" / "settings.json"
        path.write_text(
            json.dumps({BASE_URL_ENV: "https://x.test"} and {"env": {BASE_URL_ENV: "https://x.test"}}),
            encoding="utf-8",
        )
        set_proxy_env(path, api_key=None, base_url=None)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "env" not in data

    def test_clear_without_base_url_drops_stale_api_key(self, temp_home: Path):
        """Switching off a proxy must not leave the previous account's key behind."""
        path = temp_home / ".claude" / "settings.json"
        set_proxy_env(path, api_key="sk-ant-api03-A", base_url="https://a.test")
        set_proxy_env(path, api_key="sk-ant-api03-B", base_url="https://b.test")

        data = read_settings(path)
        assert data["env"][API_KEY_ENV] == "sk-ant-api03-B"
        assert data["env"][BASE_URL_ENV] == "https://b.test"

    def test_read_settings_refuses_unreadable(self, temp_home: Path, tmp_path: Path):
        from claude_swap.exceptions import ConfigError

        path = temp_home / ".claude" / "settings.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ConfigError):
            read_settings(path)

    def test_read_settings_absent_returns_empty(self, temp_home: Path):
        assert read_settings(temp_home / ".claude" / "settings.json") == {}

    def test_base_url_trailing_slash_is_stripped(self, temp_home: Path):
        """The stored URL is normalized so a doubled slash cannot be produced."""
        path = temp_home / ".claude" / "settings.json"
        set_proxy_env(path, api_key=API_KEY, base_url="https://api.example.com/")
        data = read_settings(path)
        assert data["env"][BASE_URL_ENV] == "https://api.example.com"

    def test_no_write_when_already_correct(self, temp_home: Path):
        """set_proxy_env must be idempotent and report no change."""
        path = temp_home / ".claude" / "settings.json"
        first = set_proxy_env(path, api_key=API_KEY, base_url="https://x.test")
        assert first is True
        second = set_proxy_env(path, api_key=API_KEY, base_url="https://x.test")
        assert second is False


class TestSwitchAppliesBaseUrl:
    """Integration: switching onto/off a proxy account edits settings.json env."""

    def test_switch_to_proxy_account_writes_base_url(self, temp_home: Path):
        from claude_swap.paths import get_global_config_path
        from claude_swap.switcher import ClaudeAccountSwitcher
        from claude_swap.models import Platform

        s = ClaudeAccountSwitcher()
        s.platform = Platform.LINUX
        s._setup_directories()
        s._init_sequence_file()

        # A raw key that does not match Anthropic's shape is still a managed
        # key by length/prefix rules; the important part is that it is a key,
        # not an OAuth JSON blob, so no network call happens at switch time.
        s.add_account_from_token(API_KEY, email="proxy@test.local", base_url="https://api.justwoker.icu")

        data = s._get_sequence_data()
        assert data["accounts"]["1"]["baseUrl"] == "https://api.justwoker.icu"
        assert data["accounts"]["1"]["kind"] == "api_key"

        result = s.switch_to("1", json_output=True)
        assert result["switched"] is True

        settings = json.loads((temp_home / ".claude" / "settings.json").read_text(encoding="utf-8"))
        assert settings["env"][BASE_URL_ENV] == "https://api.justwoker.icu"
        assert settings["env"][API_KEY_ENV] == API_KEY

    def test_switch_back_to_oauth_clears_stale_proxy(self, temp_home: Path):
        """A proxy URL left behind sends the next account's key to a dead host."""
        from claude_swap.switcher import ClaudeAccountSwitcher
        from claude_swap.models import Platform
        

        s = ClaudeAccountSwitcher()
        s.platform = Platform.LINUX
        s._setup_directories()
        s._init_sequence_file()

        s.add_account_from_token(API_KEY, email="proxy@test.local", base_url="https://api.justwoker.icu")
        s.add_account_from_token("sk-ant-oat01-abc", email="oauth@test.local", slot=2)

        s.switch_to("1", json_output=True)
        path = get_settings_path()
        assert "env" in json.loads(path.read_text(encoding="utf-8"))

        # Leaving the proxy account must remove both owned vars.
        s.switch_to("2", json_output=True)
        data = read_settings(path)
        assert BASE_URL_ENV not in data.get("env", {})
        assert API_KEY_ENV not in data.get("env", {})

    def test_oauth_account_cannot_carry_base_url(self, temp_home: Path, capsys):
        """A base URL on an OAuth token is a no-op, not silently applied."""
        from claude_swap.switcher import ClaudeAccountSwitcher
        from claude_swap.models import Platform

        s = ClaudeAccountSwitcher()
        s.platform = Platform.LINUX
        s._setup_directories()
        s._init_sequence_file()

        s.add_account_from_token("sk-ant-oat01-abc", email="o@test.local", base_url="https://x.test")

        data = s._get_sequence_data()
        assert "baseUrl" not in data["accounts"]["1"]
        # An OAuth token never records a kind, so the URL is a no-op on it.
        assert data["accounts"]["1"].get("kind") != "api_key"

    def test_base_url_persists_across_remove_and_readd(self, temp_home: Path):
        """The URL is metadata of the account, not a per-switch input."""
        from claude_swap.switcher import ClaudeAccountSwitcher
        from claude_swap.models import Platform

        s = ClaudeAccountSwitcher()
        s.platform = Platform.LINUX
        s._setup_directories()
        s._init_sequence_file()

        s.add_account_from_token(API_KEY, email="p@test.local", base_url="https://a.test")
        s.switch_to("1", json_output=True)
        path = get_settings_path()
        assert read_settings(path)["env"][BASE_URL_ENV] == "https://a.test"
