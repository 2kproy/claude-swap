"""Read/write the user-facing ``~/.claude/settings.json`` ``env`` block.

claude-swap normally manages only credentials: OAuth tokens in the credential
store, managed API keys in ``~/.claude.json`` ``primaryApiKey``. It never
touches ``settings.json``, because for a normal OAuth login nothing in there
needs to change.

The one case it does need to handle: an account whose API key only works
through a third-party proxy. Such a key is meaningless unless Claude Code is
also pointed at the proxy via ``env.ANTHROPIC_BASE_URL``, and that block lives
in ``settings.json`` -- a file claude-code owns and rewrites on its own (the
``/model`` command, ``/config``, plugin toggles). So the block must be
re-applied on every switch, never baked in once.

This module is deliberately narrow: it only adds or removes the two
``ANTHROPIC_*`` env vars cswap is responsible for, and preserves every other
key in ``settings.json`` untouched, mirroring how ``_update_global_config``
behaves toward ``~/.claude.json``.

Two write invariants:

- **Symlink-safe.** ``settings.json`` is a common symlink target (dotfiles
  deploys). A blind rename detaches the link and silently stops writing to
  the real file, so a link is resolved before the temp file is created.
- **Never destroy.** An unreadable (as opposed to absent) ``settings.json``
  raises rather than being overwritten with ``{}`` -- the same unreadable
  distinction ``_update_global_config`` draws on the file holding the user's
  whole Claude Code state.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from claude_swap.exceptions import ConfigError
from claude_swap.fsutil import replace_with_retry

# The two env vars cswap owns in the user's settings.json. ``ANTHROPIC_API_KEY``
# is written alongside the proxy base URL so a proxy account is complete even
# when the key is not also in ~/.claude.json (some proxies reject the
# oauthAccount flow entirely). Left unset when the account has no proxy, so a
# plain OAuth account is not left pointing at a stale proxy URL.
BASE_URL_ENV = "ANTHROPIC_BASE_URL"
API_KEY_ENV = "ANTHROPIC_API_KEY"

DEFAULT_SETTINGS_FILENAME = "settings.json"


def get_settings_path() -> Path:
    """Path to the user's ``settings.json`` (CLAUDE_CONFIG_DIR or ~/.claude)."""
    return Path(get_claude_config_home_env() or Path.home() / ".claude") / DEFAULT_SETTINGS_FILENAME


def get_claude_config_home_env() -> str | None:
    """The CLAUDE_CONFIG_DIR override, if any."""
    return os.environ.get("CLAUDE_CONFIG_DIR")


def read_settings(path: Path) -> dict:
    """Read ``settings.json``; return ``{}`` for an absent file.

    Raises :class:`ConfigError` when the file exists but cannot be read or
    parsed -- an unreadable settings.json must never be clobbered with ``{}``.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ConfigError(f"{path} could not be read ({e}); refusing to overwrite it") from e
    return data if isinstance(data, dict) else {}


def write_settings(path: Path, data: dict) -> None:
    """Atomically write ``settings.json``, preserving symlinks and 0600 mode."""
    target = Path(os.path.realpath(path)) if path.is_symlink() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform != "win32":
        os.chmod(str(path.parent), 0o700)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    try:
        os.write(fd, json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8"))
        os.close(fd)
        fd = -1
        replace_with_retry(tmp_path, str(path))
        if sys.platform != "win32":
            os.chmod(str(path), 0o600)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def set_proxy_env(
    path: Path,
    *,
    api_key: str | None,
    base_url: str | None,
    keep_extra: bool = True,
) -> bool:
    """Apply (or clear) this account's proxy env vars in ``settings.json``.

    With ``base_url`` set, writes ``env.ANTHROPIC_BASE_URL`` and, when
    ``api_key`` is given, ``env.ANTHROPIC_API_KEY``. With ``base_url`` unset
    the cswap-owned vars are removed so switching away from a proxy account
    does not leave a dead proxy URL behind pointing Claude Code at a host that
    no longer serves this account.

    Every other key in ``settings.json`` -- ``model``, ``enabledPlugins``,
    ``env`` entries that are not cswap-owned, user flags -- is preserved
    exactly. Returns True when the file changed, False when it was already in
    the requested state.
    """
    if not base_url:
        data = read_settings(path)
        env = data.get("env")
        changed = False
        if isinstance(env, dict):
            stripped = {k: v for k, v in env.items() if k not in (BASE_URL_ENV, API_KEY_ENV)}
            if len(stripped) != len(env):
                env.clear()
                env.update(stripped)
                changed = True
            if not env and "env" in data:
                del data["env"]
                changed = True
        if changed:
            write_settings(path, data)
        return changed

    data = read_settings(path)
    env = data.get("env")
    if not isinstance(env, dict):
        env = {}
        data["env"] = env
    before = dict(env)
    # Normalize here as well as at record time: a hand-edited sequence.json
    # with a trailing slash must not produce a doubled slash in the URL that
    # Claude Code sends requests to.
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ConfigError(f"base URL is empty after normalization: {base_url!r}")
    env[BASE_URL_ENV] = normalized
    if api_key:
        env[API_KEY_ENV] = api_key
    elif API_KEY_ENV in env:
        # A proxy that does not need the key re-sent would keep the previous
        # account's key in place -- that is a credential leak across accounts.
        del env[API_KEY_ENV]
    if keep_extra:
        pass  # other env entries are left exactly as found
    write_settings(path, data)
    return before != env
