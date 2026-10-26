"""Standalone desktop app (the PyInstaller build in packaging/): double-clicking it starts the local GUI server,
opens peak3d in the browser and shows a small window with Open and Quit. Started with arguments, the same
program is the peak3d command line: that is how the GUI starts runs, and how multiprocessing starts workers,
inside the frozen app.

Compiled numba kernels and matplotlib's font cache go to the user's cache folder: the app's own folder may be
read-only (macOS runs downloaded apps from a temporary, read-only location)."""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP_STATE = Path.home() / ".peak3d" / "app.json"     # link of the running app, so a second start reopens it


def user_cache() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "peak3d"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "peak3d" / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "peak3d"


def prepare_env() -> None:
    """before numba or matplotlib are imported; inherited by runs and their workers"""
    cache = user_cache()
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))


def _running_url() -> str | None:
    """the link of an app already running on this computer (it answers), else None"""
    try:
        info = json.loads(APP_STATE.read_text())
        base = info["url"].split("?")[0].rstrip("/")
        req = urllib.request.Request(base + "/api/info", headers={"X-Peak3d-Token": info["token"]})
        with urllib.request.urlopen(req, timeout=2) as r:
            return info["url"] if r.status == 200 else None
    except Exception:
        return None


def _remember(url: str, token: str) -> None:
    try:
        APP_STATE.parent.mkdir(parents=True, exist_ok=True)
        APP_STATE.write_text(json.dumps({"url": url, "token": token, "pid": os.getpid()}))
        os.chmod(APP_STATE, 0o600)
    except OSError:
        pass


def _forget() -> None:
    try:
        if json.loads(APP_STATE.read_text()).get("pid") == os.getpid():
            APP_STATE.unlink()
    except (OSError, ValueError):
        pass


def _control_window(url: str, httpd) -> bool:
    """the small Open / Quit window; False when there is no display or no Tk (then the console is the UI)"""
    try:
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import messagebox, ttk
        root = tk.Tk()
    except Exception:
        return False
    from . import __version__

    root.title("peak3d")
    root.resizable(False, False)
    bg, ink, ink2 = "#f9f9ff", "#191c20", "#44474e"
    root.configure(bg=bg)
    base = tkfont.nametofont("TkDefaultFont")
    title_font = base.copy()
    title_font.configure(size=max(base.cget("size"), 13) + 5, weight="bold")
    frame = tk.Frame(root, bg=bg, padx=24, pady=20)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, text="peak3d is running", font=title_font, fg=ink, bg=bg, anchor="w").pack(fill="x")
    tk.Label(frame, text="It is open in your web browser. Keep this window open while you work;\n"
                         "quitting stops peak3d and any run in progress.",
             fg=ink2, bg=bg, justify="left", anchor="w").pack(fill="x", pady=(6, 16))
    row = tk.Frame(frame, bg=bg)
    row.pack(fill="x")

    def open_browser():
        webbrowser.open(url)

    def quit_app():
        busy = [r for r in httpd.app.runs.values() if r.state == "running"]
        if busy and not messagebox.askyesno("peak3d", "A run is still in progress. Stop it and quit peak3d?", parent=root):
            return
        root.destroy()

    ttk.Button(row, text="Open peak3d", command=open_browser, default="active").pack(side="left")
    ttk.Button(row, text="Quit", command=quit_app).pack(side="left", padx=(8, 0))
    tk.Label(row, text=f"version {__version__}", fg=ink2, bg=bg).pack(side="right")
    root.protocol("WM_DELETE_WINDOW", quit_app)
    if sys.platform == "darwin":
        root.createcommand("tk::mac::Quit", quit_app)                   # Cmd+Q
        root.createcommand("tk::mac::ReopenApplication", open_browser)  # click on the Dock icon
    root.bind("<Return>", lambda e: open_browser())
    root.update_idletasks()
    w, h = root.winfo_reqwidth(), root.winfo_reqheight()
    root.geometry(f"+{(root.winfo_screenwidth() - w) // 2}+{(root.winfo_screenheight() - h) // 3}")
    root.lift()
    root.attributes("-topmost", True)
    root.after(800, lambda: root.attributes("-topmost", False))
    root.mainloop()
    return True


def _interrupt(*_):
    raise KeyboardInterrupt


def main(argv=None) -> None:
    prepare_env()
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = [a for a in argv if not a.startswith("-psn_")]     # old macOS Finder launches add a process serial number
    if argv:                                                  # the command line (runs started by the GUI use this)
        from .__main__ import main as cli
        cli(argv)
        return
    url = _running_url()
    if url:                                                   # already running: show it again
        webbrowser.open(url)
        return
    from .web.server import make_server
    httpd, url = make_server("127.0.0.1", 8765, None, "auto")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    _remember(url, httpd.app.token)
    signal.signal(signal.SIGTERM, _interrupt)        # logout / kill: clean up as on Ctrl+C
    threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        if not _control_window(url, httpd):
            print(f"peak3d is running; open this link in your browser:\n\n    {url}\n\nPress Ctrl+C to stop.", flush=True)
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        for r in httpd.app.runs.values():
            r.cancel()
        httpd.shutdown()
        httpd.server_close()
        _forget()
