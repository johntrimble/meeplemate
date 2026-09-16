#!/usr/bin/env python3
"""Apply machine-scope VS Code settings inside the dev container.

Some VS Code settings are declared `"scope": "machine"` by the extension that
owns them. VS Code ignores machine-scope settings when they appear in
workspace settings (`.vscode/settings.json`), so they can only be set in the
container's own Machine settings store:

    ~/.vscode-server/data/Machine/settings.json

That store lives under `.vscode-server`, which is recreated from scratch
whenever the dev container -- or just the VS Code server directory -- is
rebuilt, taking any hand-added settings with it. This script re-applies them
on every container start; `.devcontainer/devcontainer.json` wires it up as
`postStartCommand`.

The settings file is JSONC: it may carry `//` and `/* */` comments and
trailing commas, and it will already hold settings written by VS Code or
seeded from devcontainer.json. Edits here are surgical -- only the keys in
SETTINGS are touched, and all surrounding formatting and comments are left
byte for byte intact.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Machine-scope settings to enforce. See the module docstring for why these
# cannot simply be checked into .vscode/settings.json.
SETTINGS: dict[str, object] = {
    # Claude Code hides its bypass-permissions mode unless this is enabled.
    # Reasonable here because the dev container is itself the sandbox.
    "claudeCode.allowDangerouslySkipPermissions": True,
    # Start new Claude Code conversations in that mode.
    "claudeCode.initialPermissionMode": "bypassPermissions",
}

WHITESPACE = " \t\r\n"


# --- JSONC scanning -------------------------------------------------------
#
# Comments are blanked to spaces rather than deleted, so every offset in the
# masked copy still lines up with the same character in the original text.
# That is what lets us locate a value structurally and then splice the
# original string without disturbing anything else in it.


def mask_comments(text: str) -> str:
    """Return `text` with comments replaced by spaces, offsets preserved."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == '"':
            _, i = json.decoder.scanstring(text, i + 1)
            continue
        if c == "/" and i + 1 < n:
            if text[i + 1] == "/":
                end = text.find("\n", i)
                end = n if end == -1 else end
            elif text[i + 1] == "*":
                end = text.find("*/", i + 2)
                end = n if end == -1 else end + 2
            else:
                i += 1
                continue
            for k in range(i, end):
                # Keep newlines so line numbers survive for error messages.
                if out[k] != "\n":
                    out[k] = " "
            i = end
            continue
        i += 1
    return "".join(out)


def skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in WHITESPACE:
        i += 1
    return i


def value_end(s: str, i: int) -> int:
    """Index just past the JSON value starting at `i` in masked text `s`."""
    depth = 0
    while i < len(s):
        c = s[i]
        if c == '"':
            _, i = json.decoder.scanstring(s, i + 1)
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            if depth == 0:
                return i
            depth -= 1
        elif c == "," and depth == 0:
            return i
        i += 1
    return i


def drop_trailing_commas(masked: str) -> str:
    """Strip commas that sit directly before a closing brace or bracket."""
    out = list(masked)
    i, n = 0, len(masked)
    while i < n:
        c = masked[i]
        if c == '"':
            _, i = json.decoder.scanstring(masked, i + 1)
            continue
        if c == ",":
            j = skip_ws(masked, i + 1)
            if j < n and masked[j] in "}]":
                out[i] = " "
        i += 1
    return "".join(out)


def parse_jsonc(text: str) -> object:
    """Parse JSONC, tolerating comments and trailing commas."""
    src = drop_trailing_commas(mask_comments(text)).strip()
    if not src:
        return {}
    return json.loads(src)


def top_level_scan(masked: str):
    """Yield `(key, value_start, value_end)` for each top-level entry.

    Also returns the index of the object's closing brace. Raises ValueError if
    the text is not a single JSON object.
    """
    entries = []
    i = skip_ws(masked, 0)
    if i >= len(masked) or masked[i] != "{":
        raise ValueError("settings file is not a JSON object")
    i += 1
    depth = 0
    while i < len(masked):
        c = masked[i]
        if c == '"':
            if depth == 0:
                key, j = json.decoder.scanstring(masked, i + 1)
                j = skip_ws(masked, j)
                if j < len(masked) and masked[j] == ":":
                    start = skip_ws(masked, j + 1)
                    end = value_end(masked, start)
                    entries.append((key, start, len(masked[:end].rstrip())))
                    i = end
                    continue
                i = j
                continue
            _, i = json.decoder.scanstring(masked, i + 1)
            continue
        if c in "{[":
            depth += 1
        elif c in "}]":
            if depth == 0:
                return entries, i
            depth -= 1
        i += 1
    raise ValueError("settings file has an unterminated JSON object")


def detect_indent(masked: str) -> str:
    """Infer the file's indent from its first indented line."""
    match = re.search(r"^([ \t]+)\"", masked, re.MULTILINE)
    return match.group(1) if match else "\t"


def set_key(text: str, key: str, value: object) -> str:
    """Return `text` with top-level `key` set to `value`, preserving the rest."""
    masked = mask_comments(text)
    entries, close = top_level_scan(masked)
    encoded = json.dumps(value)

    for name, start, end in entries:
        if name == key:
            return text[:start] + encoded + text[end:]

    indent = detect_indent(masked)
    insert_at = len(masked[:close].rstrip())
    prev = masked[:insert_at][-1:] if insert_at else ""
    # A trailing comma is already present after the last entry, and an empty
    # object needs no separator at all.
    separator = "\n" if prev in ("{", ",") else ",\n"
    entry = f"{separator}{indent}{json.dumps(key)}: {encoded}"
    return text[:insert_at] + entry + text[insert_at:]


# --- applying -------------------------------------------------------------


def machine_settings_paths() -> list[Path]:
    """Machine settings files for every VS Code server flavour present."""
    home = Path.home()
    roots = sorted(p for p in home.glob(".vscode-server*") if p.is_dir())
    if not roots:
        roots = [home / ".vscode-server"]
    return [root / "data" / "Machine" / "settings.json" for root in roots]


def apply_to(path: Path, dry_run: bool) -> str:
    """Bring one settings file in line with SETTINGS.

    Returns "changed", "ok" or "failed".
    """
    original = path.read_text(encoding="utf-8") if path.is_file() else ""

    if original.strip():
        try:
            current = parse_jsonc(original)
        except ValueError as exc:
            # Rewriting a file we cannot parse risks destroying real settings.
            print(f"{path}: cannot parse, leaving untouched ({exc})", file=sys.stderr)
            return "failed"
        if not isinstance(current, dict):
            print(f"{path}: top level is not an object, leaving untouched", file=sys.stderr)
            return "failed"
    else:
        current = {}
        original = "{\n}\n"

    updated = original
    changed = [k for k, v in SETTINGS.items() if current.get(k, object()) != v]
    if not changed:
        print(f"{path}: already up to date")
        return "ok"

    for key in changed:
        updated = set_key(updated, key, SETTINGS[key])

    # Never install a file we cannot read back with the values we intended.
    verify = parse_jsonc(updated)
    if not isinstance(verify, dict) or any(verify.get(k) != SETTINGS[k] for k in SETTINGS):
        print(f"{path}: rewrite failed verification, leaving untouched", file=sys.stderr)
        return "failed"

    for key in changed:
        print(f"{path}: setting {key} = {json.dumps(SETTINGS[key])}")
    if dry_run:
        print(f"{path}: dry run, not written")
        return "changed"

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            path.with_suffix(".json.bak").write_text(original, encoding="utf-8")
        # Write via a temp file in the same directory so a reader never observes
        # a half-written settings file.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(updated, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        # This runs from a container boot hook, where a traceback is just noise.
        print(f"{path}: could not write ({exc})", file=sys.stderr)
        return "failed"
    return "changed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing",
    )
    args = parser.parse_args()

    results = [apply_to(path, args.dry_run) for path in machine_settings_paths()]
    if "changed" in results and not args.dry_run:
        print("Reload the VS Code window for the new settings to take effect.")
    # Surface a bad settings file loudly: silently not applying these is the
    # exact failure this script exists to prevent.
    return 1 if "failed" in results else 0


if __name__ == "__main__":
    sys.exit(main())
