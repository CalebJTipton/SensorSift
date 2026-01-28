import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Dependencies:
from PIL import Image, ExifTags  # pip install pillow

# Optional encryption:
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # pip install cryptography
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet
import base64
import secrets


# -----------------------------
# App identity
# -----------------------------
APP_NAME = "SensorSift"
APP_POINTER_DIR = Path.home() / f".{APP_NAME.lower()}"
APP_POINTER_FILE = APP_POINTER_DIR / "pointer.json"


# -----------------------------
# Defaults
# -----------------------------
DEFAULT_RAW_EXTS = [
    ".cr2", ".cr3", ".arw", ".dng", ".nef", ".raf", ".rw2", ".orf",
    ".x3f", ".srw", ".pef", ".iiq", ".3fr", ".braw",
    ".insv"  # Insta360 video container (treat as RAW/unprocessed)
]
DEFAULT_PROCESSED_EXTS = [
    ".jpg", ".jpeg", ".png", ".heic", ".gif", ".tif", ".tiff",
    ".mp4", ".mov", ".m4v", ".avi", ".mkv"
]

DEFAULT_ROUTING_RULES = [
    # Insta360 (many SD cards have INSTA360 folder; .insv should be RAW-like)
    {
        "name": "Insta360 RAW-like",
        "path_contains_any": ["INSTA360"],
        "ext_in": [".insv", ".dng"],
        "bucket": "raw",
        "dest_tag": "Insta360"
    },
    # GoPro
    {
        "name": "GoPro processed media",
        "path_contains_any": ["GOPRO"],
        "ext_in": [".mp4", ".mov", ".jpg", ".jpeg"],
        "bucket": "processed",
        "dest_tag": "GoPro"
    },
    # Generic camera roll
    {
        "name": "DCIM processed media",
        "path_contains_any": ["DCIM"],
        "ext_in": [".mp4", ".mov", ".jpg", ".jpeg", ".png", ".heic"],
        "bucket": "processed",
        "dest_tag": "Camera"
    }
]

COMMON_MEDIA_FOLDERS = [
    "DCIM",
    "PRIVATE",
    "GOPRO",
    "INSTA360",
    "DJI",
    "MISC",
    "AVCHD",
    "MP_ROOT",
    "XDROOT",
    "CANONMSC",
]


# -----------------------------
# Helpers
# -----------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def detect_suggested_sync_root() -> Path:
    """
    Suggest a config folder that syncs across devices.
    Prioritize OneDrive on Windows, else Dropbox/iCloud/Docs.
    """
    home = Path.home()

    for env_key in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        val = os.getenv(env_key)
        if val and Path(val).exists():
            return Path(val) / APP_NAME

    candidates = [
        home / "OneDrive" / APP_NAME,
        home / "Dropbox" / APP_NAME,
        home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / APP_NAME,  # iCloud (macOS)
        home / "Documents" / APP_NAME
    ]
    for c in candidates:
        if c.parent.exists():
            return c

    return home / "Documents" / APP_NAME


def save_pointer_root(root: Path) -> None:
    ensure_dir(APP_POINTER_DIR)
    APP_POINTER_FILE.write_text(json.dumps({"config_root": str(root)}, indent=2), encoding="utf-8")


def load_pointer_root() -> Optional[Path]:
    try:
        if APP_POINTER_FILE.exists():
            data = json.loads(APP_POINTER_FILE.read_text(encoding="utf-8"))
            p = Path(data["config_root"])
            return p if p.exists() else None
    except Exception:
        return None
    return None


def safe_dest_path(dest_dir: Path, filename: str) -> Path:
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    i = 1
    while True:
        candidate = dest_dir / f"{base}__{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def iter_files(source: Path):
    for p in source.rglob("*"):
        if p.is_file():
            yield p


# -----------------------------
# EXIF date extraction (2)
# -----------------------------
def _parse_exif_datetime(s: str) -> Optional[datetime]:
    """
    Common EXIF format: "YYYY:MM:DD HH:MM:SS"
    """
    if not s:
        return None
    s = str(s).strip()
    m = re.match(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})", s)
    if not m:
        return None
    y, mo, d, hh, mm, ss = map(int, m.groups())
    return datetime(y, mo, d, hh, mm, ss)


def get_capture_datetime_pillow(path: Path) -> Optional[datetime]:
    """
    Fast path for JPG/TIFF using Pillow.
    """
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                return None

            # Map EXIF tag ids -> names
            tagmap = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}

            # Prefer DateTimeOriginal, then CreateDate, then DateTime
            for key in ("DateTimeOriginal", "CreateDate", "DateTime"):
                if key in tagmap:
                    dt = _parse_exif_datetime(tagmap[key])
                    if dt:
                        return dt
    except Exception:
        return None
    return None


def get_capture_datetime_exiftool(exiftool_path: str, path: Path) -> Optional[datetime]:
    """
    More universal (videos, many raws, etc.) if exiftool is installed.
    Calls exiftool once per file (simple starter).
    """
    try:
        cmd = [
            exiftool_path,
            "-j",
            "-DateTimeOriginal",
            "-CreateDate",
            "-MediaCreateDate",
            "-TrackCreateDate",
            "-QuickTime:CreateDate",
            "-ModifyDate",
            str(path)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = json.loads(out.decode("utf-8", errors="ignore"))
        if not data:
            return None
        row = data[0]
        # try keys in priority order
        for k in ("DateTimeOriginal", "CreateDate", "MediaCreateDate", "TrackCreateDate", "CreateDate", "ModifyDate"):
            v = row.get(k)
            dt = _parse_exif_datetime(v) if v else None
            if dt:
                return dt
    except Exception:
        return None
    return None


def get_capture_datetime(cfg: "AppConfig", path: Path) -> datetime:
    """
    date_strategy:
      - exif_then_mtime (default)
      - mtime_only
    """
    if cfg.date_strategy == "mtime_only":
        return datetime.fromtimestamp(path.stat().st_mtime)

    # Pillow first (fast for JPG/TIF)
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg", ".tif", ".tiff"):
        dt = get_capture_datetime_pillow(path)
        if dt:
            return dt

    # exiftool fallback if configured
    if cfg.exiftool_path:
        dt = get_capture_datetime_exiftool(cfg.exiftool_path, path)
        if dt:
            return dt

    # final fallback: mtime
    return datetime.fromtimestamp(path.stat().st_mtime)


def date_folders_from_dt(dt: datetime) -> Tuple[str, str]:
    return dt.strftime("%Y"), dt.strftime("%Y-%m")


# -----------------------------
# Config + secrets (4)
# -----------------------------
@dataclass
class ImmichConfig:
    enabled: bool
    cli_path: str
    server_url: str


@dataclass
class AppConfig:
    config_root: Path

    raw_root: Path
    processed_stage_root: Path
    processed_uploaded_root: Optional[Path]

    copy_mode: str  # copy|move

    raw_extensions: List[str]
    processed_extensions: List[str]

    include_unknown: bool
    unknown_subfolder_name: str

    # New:
    routing_rules: List[Dict[str, Any]]
    date_strategy: str  # exif_then_mtime | mtime_only
    exiftool_path: str  # "" or "exiftool"

    immich: ImmichConfig


def config_file(root: Path) -> Path:
    return root / "config.json"


def secrets_file(root: Path) -> Path:
    return root / "secrets.json"


def load_config(root: Path) -> AppConfig:
    p = config_file(root)
    if not p.exists():
        raise FileNotFoundError("Config not found")

    data = json.loads(p.read_text(encoding="utf-8"))
    imm = data.get("immich", {})

    uploaded = data.get("processed_uploaded_root", "")
    uploaded_path = Path(uploaded) if uploaded else None

    return AppConfig(
        config_root=root,
        raw_root=Path(data["raw_root"]),
        processed_stage_root=Path(data["processed_stage_root"]),
        processed_uploaded_root=uploaded_path,
        copy_mode=str(data.get("copy_mode", "copy")).lower(),
        raw_extensions=[e.lower() for e in data.get("raw_extensions", DEFAULT_RAW_EXTS)],
        processed_extensions=[e.lower() for e in data.get("processed_extensions", DEFAULT_PROCESSED_EXTS)],
        include_unknown=bool(data.get("include_unknown", False)),
        unknown_subfolder_name=str(data.get("unknown_subfolder_name", "_Unknown")),
        routing_rules=list(data.get("routing_rules", DEFAULT_ROUTING_RULES)),
        date_strategy=str(data.get("date_strategy", "exif_then_mtime")),
        exiftool_path=str(data.get("exiftool_path", "")),
        immich=ImmichConfig(
            enabled=bool(imm.get("enabled", True)),
            cli_path=str(imm.get("cli_path", "immich")),
            server_url=str(imm.get("server_url", "")).strip()
        )
    )


def save_config(cfg: AppConfig) -> None:
    ensure_dir(cfg.config_root)
    data = {
        "raw_root": str(cfg.raw_root),
        "processed_stage_root": str(cfg.processed_stage_root),
        "processed_uploaded_root": str(cfg.processed_uploaded_root) if cfg.processed_uploaded_root else "",
        "copy_mode": cfg.copy_mode,
        "raw_extensions": cfg.raw_extensions,
        "processed_extensions": cfg.processed_extensions,
        "include_unknown": cfg.include_unknown,
        "unknown_subfolder_name": cfg.unknown_subfolder_name,
        "routing_rules": cfg.routing_rules,
        "date_strategy": cfg.date_strategy,
        "exiftool_path": cfg.exiftool_path,
        "immich": {
            "enabled": cfg.immich.enabled,
            "cli_path": cfg.immich.cli_path,
            "server_url": cfg.immich.server_url
        }
    }
    config_file(cfg.config_root).write_text(json.dumps(data, indent=2), encoding="utf-8")
    save_pointer_root(cfg.config_root)


def load_secrets(root: Path) -> Dict[str, Any]:
    p = secrets_file(root)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_secrets(root: Path, data: Dict[str, Any]) -> None:
    ensure_dir(root)
    secrets_file(root).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _derive_fernet_key(passphrase: str, salt: bytes) -> bytes:
    """
    PBKDF2 -> 32 bytes -> urlsafe base64 for Fernet
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=200_000,
    )
    key = kdf.derive(passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(key)


def encrypt_text(passphrase: str, plaintext: str) -> Dict[str, str]:
    salt = secrets.token_bytes(16)
    key = _derive_fernet_key(passphrase, salt)
    f = Fernet(key)
    token = f.encrypt(plaintext.encode("utf-8"))
    return {
        "enc": "fernet-pbkdf2-sha256",
        "salt_b64": base64.b64encode(salt).decode("ascii"),
        "token_b64": base64.b64encode(token).decode("ascii"),
    }


def decrypt_text(passphrase: str, payload: Dict[str, str]) -> str:
    salt = base64.b64decode(payload["salt_b64"])
    token = base64.b64decode(payload["token_b64"])
    key = _derive_fernet_key(passphrase, salt)
    f = Fernet(key)
    return f.decrypt(token).decode("utf-8")


def get_immich_api_key(cfg: AppConfig, passphrase: Optional[str]) -> str:
    s = load_secrets(cfg.config_root)
    mode = s.get("mode", "plaintext")
    if mode == "plaintext":
        return (s.get("immich_api_key") or "").strip()
    if mode == "encrypted":
        if not passphrase:
            raise RuntimeError("API key is encrypted. Provide passphrase in Settings.")
        enc_payload = s.get("immich_api_key_encrypted")
        if not isinstance(enc_payload, dict):
            raise RuntimeError("Encrypted API key payload missing or invalid.")
        return decrypt_text(passphrase, enc_payload).strip()
    raise RuntimeError(f"Unknown secrets mode: {mode}")


# -----------------------------
# Routing rules (3)
# -----------------------------
def match_rule(rule: Dict[str, Any], src_path: Path) -> bool:
    s = str(src_path).upper()
    ext = src_path.suffix.lower()

    path_contains_any = [x.upper() for x in rule.get("path_contains_any", [])]
    ext_in = [x.lower() for x in rule.get("ext_in", [])]

    ok_path = True
    if path_contains_any:
        ok_path = any(token in s for token in path_contains_any)

    ok_ext = True
    if ext_in:
        ok_ext = ext in ext_in

    return ok_path and ok_ext


def route_file(cfg: AppConfig, src: Path) -> Tuple[str, str]:
    """
    Returns (bucket, dest_tag)
    bucket: raw|processed|unknown
    dest_tag: optional extra folder label (device bucket tag)
    """
    # Rules first (ordered)
    for rule in cfg.routing_rules:
        if match_rule(rule, src):
            return rule.get("bucket", "unknown"), rule.get("dest_tag", "")

    # Default classification by extension
    ext = src.suffix.lower()
    if ext in cfg.raw_extensions:
        return "raw", ""
    if ext in cfg.processed_extensions:
        return "processed", ""
    return "unknown", ""


# -----------------------------
# SD/source autodetect (1)
# -----------------------------
def find_best_media_subfolder(source_root: Path) -> Optional[Path]:
    """
    Scan for common media folders. If found, return the most promising subfolder.
    Strategy:
      - If <root>/DCIM exists, return it
      - Else if any known folder exists, return first found
      - Else scan one level deep and pick a folder containing many media files
    """
    # Prefer direct common roots
    for name in COMMON_MEDIA_FOLDERS:
        p = source_root / name
        if p.exists() and p.is_dir():
            return p

    # Look one level deep for those folders
    for child in source_root.iterdir():
        if child.is_dir() and child.name.upper() in COMMON_MEDIA_FOLDERS:
            return child

    # Heuristic: pick subfolder with most media-like files (1-level)
    media_exts = set(DEFAULT_RAW_EXTS + DEFAULT_PROCESSED_EXTS)
    best = None
    best_count = 0
    for child in source_root.iterdir():
        if not child.is_dir():
            continue
        count = 0
        try:
            for p in child.rglob("*"):
                if p.is_file() and p.suffix.lower() in media_exts:
                    count += 1
                    if count > best_count:
                        best_count = count
                        best = child
        except Exception:
            continue

    return best if best_count >= 10 else None  # threshold keeps it from choosing random folders


# -----------------------------
# Intake + Upload core
# -----------------------------
def copy_or_move(cfg: AppConfig, src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if cfg.copy_mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def intake_media(cfg: AppConfig, source: Path, log_cb, progress_cb, stop_flag) -> Tuple[Dict[str, int], Path]:
    ensure_dir(cfg.raw_root)
    ensure_dir(cfg.processed_stage_root)
    logs_dir = cfg.config_root / "logs"
    ensure_dir(logs_dir)
    log_path = logs_dir / f"intake_{now_stamp()}.log"

    files = list(iter_files(source))
    total = len(files)
    counts = {"raw": 0, "processed": 0, "unknown": 0, "skipped": 0, "errors": 0}

    def log(msg: str):
        log_cb(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"{APP_NAME} intake run: {datetime.now().isoformat()}")
    log(f"Source: {source}")
    log(f"Mode: {cfg.copy_mode}")
    log(f"Files found: {total}")
    log(f"Date strategy: {cfg.date_strategy} (exiftool: {cfg.exiftool_path or 'none'})")
    log("")

    for i, src in enumerate(files, start=1):
        if stop_flag["stop"]:
            log("STOP requested by user.")
            break

        try:
            bucket, tag = route_file(cfg, src)

            if bucket == "unknown" and not cfg.include_unknown:
                counts["skipped"] += 1
                log(f"SKIP (unknown): {src}")
                progress_cb(i, total)
                continue

            dt = get_capture_datetime(cfg, src)
            year, year_month = date_folders_from_dt(dt)

            if bucket == "raw":
                base_dir = cfg.raw_root
            elif bucket == "processed":
                base_dir = cfg.processed_stage_root
            else:
                base_dir = cfg.processed_stage_root / cfg.unknown_subfolder_name

            # Optional device/tag folder layer
            dest_dir = base_dir
            if tag:
                dest_dir = dest_dir / tag

            dest_dir = dest_dir / year / year_month

            dst = safe_dest_path(dest_dir, src.name)
            copy_or_move(cfg, src, dst)

            counts[bucket] += 1
            log(f"OK  [{bucket}] {src} -> {dst}")

        except Exception as e:
            counts["errors"] += 1
            log(f"ERR {src} :: {repr(e)}")

        progress_cb(i, total)

    log("")
    log("Summary:")
    for k, v in counts.items():
        log(f"  {k}: {v}")
    log(f"Log file: {log_path}")

    return counts, log_path


def immich_login_and_upload(cfg: AppConfig, upload_dir: Path, api_key: str, log_cb):
    if not cfg.immich.server_url:
        raise RuntimeError("Immich server_url not set in config.")
    if not api_key:
        raise RuntimeError("Immich API key is empty.")

    cli = cfg.immich.cli_path

    def run(cmd):
        log_cb("Running: " + " ".join(cmd))
        subprocess.run(cmd, check=True)

    run([cli, "login", cfg.immich.server_url, api_key])
    run([cli, "upload", "--recursive", str(upload_dir)])


# -----------------------------
# GUI: Setup Wizard
# -----------------------------
class SetupWizard(tk.Toplevel):
    def __init__(self, parent, on_done, existing_cfg: Optional[AppConfig] = None):
        super().__init__(parent)
        self.title(f"{APP_NAME} - Setup Wizard")
        self.geometry("860x680")
        self.resizable(False, False)
        self.on_done = on_done
        self.existing_cfg = existing_cfg

        suggested_root = detect_suggested_sync_root() if not existing_cfg else existing_cfg.config_root

        self.var_config_root = tk.StringVar(value=str(suggested_root))
        self.var_raw_root = tk.StringVar(value=str((suggested_root / "RAW_Archive")))
        self.var_stage_root = tk.StringVar(value=str((suggested_root / "Immich_Staging")))
        self.var_uploaded_root = tk.StringVar(value=str((suggested_root / "Immich_Uploaded")))
        self.var_use_uploaded = tk.BooleanVar(value=True)

        self.var_copy_mode = tk.StringVar(value="copy")

        # New: date strategy + exiftool
        self.var_date_strategy = tk.StringVar(value="exif_then_mtime")
        self.var_exiftool_path = tk.StringVar(value="exiftool")  # leave blank to disable

        # Immich
        self.var_immich_enabled = tk.BooleanVar(value=True)
        self.var_immich_url = tk.StringVar(value="http://127.0.0.1:2283/api")
        self.var_cli_path = tk.StringVar(value="immich")

        # Key storage (4)
        self.var_api_key = tk.StringVar(value="")
        self.var_encrypt_key = tk.BooleanVar(value=True)
        self.var_passphrase = tk.StringVar(value="")

        # Rules (3)
        self.var_raw_exts = tk.StringVar(value=" ".join(DEFAULT_RAW_EXTS))
        self.var_proc_exts = tk.StringVar(value=" ".join(DEFAULT_PROCESSED_EXTS))
        self.var_rules_json = tk.StringVar(value=json.dumps(DEFAULT_ROUTING_RULES, indent=2))

        # Unknown handling
        self.var_include_unknown = tk.BooleanVar(value=False)
        self.var_unknown_name = tk.StringVar(value="_Unknown")

        if existing_cfg:
            self._load_from_existing(existing_cfg)

        self._build()

    def _load_from_existing(self, cfg: AppConfig):
        self.var_config_root.set(str(cfg.config_root))
        self.var_raw_root.set(str(cfg.raw_root))
        self.var_stage_root.set(str(cfg.processed_stage_root))
        self.var_uploaded_root.set(str(cfg.processed_uploaded_root) if cfg.processed_uploaded_root else "")
        self.var_use_uploaded.set(bool(cfg.processed_uploaded_root))
        self.var_copy_mode.set(cfg.copy_mode)
        self.var_date_strategy.set(cfg.date_strategy)
        self.var_exiftool_path.set(cfg.exiftool_path)
        self.var_immich_enabled.set(cfg.immich.enabled)
        self.var_immich_url.set(cfg.immich.server_url)
        self.var_cli_path.set(cfg.immich.cli_path)
        self.var_raw_exts.set(" ".join(cfg.raw_extensions))
        self.var_proc_exts.set(" ".join(cfg.processed_extensions))
        self.var_rules_json.set(json.dumps(cfg.routing_rules, indent=2))
        self.var_include_unknown.set(cfg.include_unknown)
        self.var_unknown_name.set(cfg.unknown_subfolder_name)

    def _browse_dir_into(self, var: tk.StringVar):
        p = filedialog.askdirectory(initialdir=var.get() or str(Path.home()))
        if p:
            var.set(p)

    def _build(self):
        frm = ttk.Frame(self, padding=16)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text=f"{APP_NAME} Setup Wizard", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        # Config root
        block = ttk.LabelFrame(frm, text="Where to store settings (sync this folder across machines)", padding=12)
        block.pack(fill="x", pady=10)
        row = ttk.Frame(block); row.pack(fill="x")
        ttk.Label(row, text="Config folder:", width=12).pack(side="left")
        ttk.Entry(row, textvariable=self.var_config_root, width=70).pack(side="left", padx=8)
        ttk.Button(row, text="Browse", command=lambda: self._browse_dir_into(self.var_config_root)).pack(side="left")

        # Dest paths
        block2 = ttk.LabelFrame(frm, text="Import destinations", padding=12)
        block2.pack(fill="x", pady=10)

        def path_row(label, var):
            r = ttk.Frame(block2); r.pack(fill="x", pady=4)
            ttk.Label(r, text=label, width=16).pack(side="left")
            ttk.Entry(r, textvariable=var, width=64).pack(side="left", padx=8)
            ttk.Button(r, text="Browse", command=lambda: self._browse_dir_into(var)).pack(side="left")

        path_row("RAW archive:", self.var_raw_root)
        path_row("Immich staging:", self.var_stage_root)

        r_up = ttk.Frame(block2); r_up.pack(fill="x", pady=4)
        ttk.Checkbutton(r_up, text="Use 'Uploaded' folder", variable=self.var_use_uploaded).pack(side="left")
        ttk.Entry(r_up, textvariable=self.var_uploaded_root, width=56).pack(side="left", padx=8)
        ttk.Button(r_up, text="Browse", command=lambda: self._browse_dir_into(self.var_uploaded_root)).pack(side="left")

        # Behavior
        block3 = ttk.LabelFrame(frm, text="Import behavior", padding=12)
        block3.pack(fill="x", pady=10)
        ttk.Radiobutton(block3, text="Copy (recommended)", value="copy", variable=self.var_copy_mode).pack(anchor="w")
        ttk.Radiobutton(block3, text="Move", value="move", variable=self.var_copy_mode).pack(anchor="w")

        # Date strategy
        block4 = ttk.LabelFrame(frm, text="Folder date strategy (2)", padding=12)
        block4.pack(fill="x", pady=10)
        ttk.Radiobutton(block4, text="EXIF date taken → fallback to file modified time", value="exif_then_mtime",
                        variable=self.var_date_strategy).pack(anchor="w")
        ttk.Radiobutton(block4, text="File modified time only (fastest)", value="mtime_only",
                        variable=self.var_date_strategy).pack(anchor="w")

        r_ex = ttk.Frame(block4); r_ex.pack(fill="x", pady=6)
        ttk.Label(r_ex, text="exiftool path (optional):", width=20).pack(side="left")
        ttk.Entry(r_ex, textvariable=self.var_exiftool_path, width=30).pack(side="left", padx=8)
        ttk.Label(r_ex, text="(blank disables; 'exiftool' if on PATH)").pack(side="left")

        # Extensions
        block5 = ttk.LabelFrame(frm, text="Extensions", padding=12)
        block5.pack(fill="x", pady=10)
        ttk.Label(block5, text="RAW extensions (space-separated):").pack(anchor="w")
        ttk.Entry(block5, textvariable=self.var_raw_exts).pack(fill="x", pady=(0, 6))
        ttk.Label(block5, text="Processed extensions (space-separated):").pack(anchor="w")
        ttk.Entry(block5, textvariable=self.var_proc_exts).pack(fill="x")

        # Routing rules
        block6 = ttk.LabelFrame(frm, text="Routing rules (3) — first match wins (JSON)", padding=12)
        block6.pack(fill="both", pady=10, expand=True)

        rules_box = tk.Text(block6, height=10, wrap="none")
        rules_box.pack(fill="both", expand=True)
        rules_box.insert("1.0", self.var_rules_json.get())

        def sync_rules_out():
            self.var_rules_json.set(rules_box.get("1.0", "end").strip())

        # Immich
        block7 = ttk.LabelFrame(frm, text="Immich upload (optional)", padding=12)
        block7.pack(fill="x", pady=10)

        ttk.Checkbutton(block7, text="Enable Immich upload", variable=self.var_immich_enabled).pack(anchor="w")

        r1 = ttk.Frame(block7); r1.pack(fill="x", pady=4)
        ttk.Label(r1, text="Immich URL:", width=12).pack(side="left")
        ttk.Entry(r1, textvariable=self.var_immich_url, width=60).pack(side="left", padx=8)

        r2 = ttk.Frame(block7); r2.pack(fill="x", pady=4)
        ttk.Label(r2, text="CLI name:", width=12).pack(side="left")
        ttk.Entry(r2, textvariable=self.var_cli_path, width=20).pack(side="left", padx=8)
        ttk.Label(r2, text="(usually 'immich')").pack(side="left")

        # Key storage
        r3 = ttk.Frame(block7); r3.pack(fill="x", pady=4)
        ttk.Label(r3, text="API key:", width=12).pack(side="left")
        ttk.Entry(r3, textvariable=self.var_api_key, width=60, show="•").pack(side="left", padx=8)

        r4 = ttk.Frame(block7); r4.pack(fill="x", pady=4)
        ttk.Checkbutton(r4, text="Encrypt API key with passphrase (recommended)", variable=self.var_encrypt_key).pack(side="left")

        r5 = ttk.Frame(block7); r5.pack(fill="x", pady=4)
        ttk.Label(r5, text="Passphrase:", width=12).pack(side="left")
        ttk.Entry(r5, textvariable=self.var_passphrase, width=30, show="•").pack(side="left", padx=8)
        ttk.Label(r5, text="(you must remember this on each machine)").pack(side="left")

        # Unknowns
        block8 = ttk.LabelFrame(frm, text="Unknown files", padding=12)
        block8.pack(fill="x", pady=10)
        ttk.Checkbutton(block8, text="Include unknown file types (otherwise skip)", variable=self.var_include_unknown).pack(anchor="w")
        r8 = ttk.Frame(block8); r8.pack(fill="x", pady=4)
        ttk.Label(r8, text="Unknown subfolder:", width=16).pack(side="left")
        ttk.Entry(r8, textvariable=self.var_unknown_name, width=30).pack(side="left", padx=8)

        # Buttons
        btns = ttk.Frame(frm)
        btns.pack(fill="x", pady=(8, 0))
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(btns, text="Save & Finish", command=lambda: self._finish(sync_rules_out)).pack(side="right", padx=8)

    def _finish(self, sync_rules_out):
        try:
            sync_rules_out()

            root = Path(self.var_config_root.get()).expanduser()
            ensure_dir(root)

            rules = json.loads(self.var_rules_json.get())
            if not isinstance(rules, list):
                raise ValueError("Routing rules JSON must be a list of rule objects.")

            cfg = AppConfig(
                config_root=root,
                raw_root=Path(self.var_raw_root.get()).expanduser(),
                processed_stage_root=Path(self.var_stage_root.get()).expanduser(),
                processed_uploaded_root=Path(self.var_uploaded_root.get()).expanduser()
                if self.var_use_uploaded.get() and self.var_uploaded_root.get().strip() else None,
                copy_mode=self.var_copy_mode.get().lower(),
                raw_extensions=[x.lower() for x in self.var_raw_exts.get().split() if x.strip()],
                processed_extensions=[x.lower() for x in self.var_proc_exts.get().split() if x.strip()],
                include_unknown=bool(self.var_include_unknown.get()),
                unknown_subfolder_name=self.var_unknown_name.get().strip() or "_Unknown",
                routing_rules=rules,
                date_strategy=self.var_date_strategy.get(),
                exiftool_path=self.var_exiftool_path.get().strip(),  # blank disables
                immich=ImmichConfig(
                    enabled=bool(self.var_immich_enabled.get()),
                    cli_path=self.var_cli_path.get().strip() or "immich",
                    server_url=self.var_immich_url.get().strip()
                )
            )

            save_config(cfg)

            # Save secrets
            api_key = self.var_api_key.get().strip()
            if api_key:
                secrets_data = load_secrets(root)

                if self.var_encrypt_key.get():
                    passphrase = self.var_passphrase.get().strip()
                    if not passphrase:
                        raise ValueError("Passphrase required if encryption is enabled.")
                    secrets_data["mode"] = "encrypted"
                    secrets_data["immich_api_key_encrypted"] = encrypt_text(passphrase, api_key)
                    # remove plaintext if present
                    secrets_data.pop("immich_api_key", None)
                else:
                    secrets_data["mode"] = "plaintext"
                    secrets_data["immich_api_key"] = api_key
                    secrets_data.pop("immich_api_key_encrypted", None)

                save_secrets(root, secrets_data)

            self.on_done(cfg)
            self.destroy()

        except Exception as e:
            messagebox.showerror("Setup failed", str(e))


# -----------------------------
# GUI: Main App
# -----------------------------
class SensorSiftApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME}")
        self.geometry("920x600")

        self.cfg: Optional[AppConfig] = None
        self.queue = Queue()
        self.stop_flag = {"stop": False}
        self.worker_thread: Optional[threading.Thread] = None

        # Encryption passphrase runtime (for encrypted secrets)
        self.runtime_passphrase = tk.StringVar(value="")

        self._build_ui()
        self.after(100, self._poll_queue)
        self._load_or_wizard()

    def _build_ui(self):
        top = ttk.Frame(self, padding=12)
        top.pack(fill="x")

        ttk.Label(top, text=APP_NAME, font=("Segoe UI", 16, "bold")).pack(side="left")

        ttk.Button(top, text="Settings / Wizard", command=self._open_wizard).pack(side="right")

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        # Action buttons
        btn_row = ttk.Frame(body)
        btn_row.pack(fill="x", pady=(0, 10))

        self.btn_intake = ttk.Button(btn_row, text="Intake media (SD / folder)", command=self._on_intake)
        self.btn_intake.pack(side="left")

        self.btn_upload = ttk.Button(btn_row, text="Upload staging to Immich", command=self._on_upload)
        self.btn_upload.pack(side="left", padx=10)

        self.btn_stop = ttk.Button(btn_row, text="Stop current task", command=self._on_stop, state="disabled")
        self.btn_stop.pack(side="left", padx=10)

        # Workflow hints
        flow_frame = ttk.LabelFrame(body, text="SensorSift workflow", padding=10)
        flow_frame.pack(fill="x", pady=(0, 10))
        steps = [
            "1. Run Settings / Wizard to define where SensorSift should store RAW, staged, and optional uploaded media.",
            "2. Click Intake media to read an SD card or folder; SensorSift auto-detects common media roots for you.",
            "3. After intake, the staged processed folder is ready for manual review or automatic Immich upload."
        ]
        for step in steps:
            ttk.Label(flow_frame, text=step).pack(anchor="w", pady=2)

        # Passphrase box (only needed if secrets encrypted)
        pass_row = ttk.Frame(body)
        pass_row.pack(fill="x", pady=(0, 10))
        ttk.Label(pass_row, text="Passphrase (only if API key is encrypted):", width=34).pack(side="left")
        ttk.Entry(pass_row, textvariable=self.runtime_passphrase, width=30, show="•").pack(side="left")

        # Progress
        prog_row = ttk.Frame(body)
        prog_row.pack(fill="x", pady=(0, 10))
        self.prog = ttk.Progressbar(prog_row, orient="horizontal", mode="determinate")
        self.prog.pack(fill="x", expand=True)
        self.prog_lbl = ttk.Label(prog_row, text="Idle")
        self.prog_lbl.pack(anchor="w")

        # Log area
        self.log = tk.Text(body, height=22, wrap="word")
        self.log.pack(fill="both", expand=True)
        self._log("Ready.")

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.insert("end", f"[{ts}] {msg}\n")
        self.log.see("end")

    def _set_busy(self, busy: bool):
        self.btn_intake.config(state="disabled" if busy else "normal")
        self.btn_upload.config(state="disabled" if busy else "normal")
        self.btn_stop.config(state="normal" if busy else "disabled")
        self.stop_flag["stop"] = False

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "progress":
                    cur, total = payload
                    self.prog["maximum"] = max(total, 1)
                    self.prog["value"] = cur
                    self.prog_lbl.config(text=f"Processed {cur}/{total}")
                elif kind == "done":
                    self._set_busy(False)
                    self.prog_lbl.config(text="Done")
                    self._log(payload)
                elif kind == "error":
                    self._set_busy(False)
                    messagebox.showerror("Error", payload)
        except Empty:
            pass
        self.after(100, self._poll_queue)

    def _load_or_wizard(self):
        root = load_pointer_root()
        if root:
            try:
                self.cfg = load_config(root)
                self._log(f"Loaded config from: {root}")
                return
            except Exception:
                pass
        self._open_wizard(first_run=True)

    def _open_wizard(self, first_run: bool = False):
        def done(cfg: AppConfig):
            self.cfg = cfg
            self._log(f"Saved config to: {cfg.config_root}")

        wiz = SetupWizard(self, on_done=done, existing_cfg=self.cfg)
        if first_run:
            wiz.grab_set()

    def _on_stop(self):
        self.stop_flag["stop"] = True
        self._log("Stop requested...")

    def _choose_source_with_autodetect(self) -> Optional[Path]:
        """
        User picks a root, then we auto-detect a better subfolder if present.
        """
        src = filedialog.askdirectory(title="Select SD card / source folder")
        if not src:
            return None

        root = Path(src)
        if not root.exists():
            messagebox.showerror("Invalid source", "Folder does not exist.")
            return None

        best = find_best_media_subfolder(root)
        if best and best != root:
            use_best = messagebox.askyesno(
                "Auto-detected media folder",
                f"I found a likely media folder:\n\n{best}\n\nUse this instead of:\n{root} ?"
            )
            return best if use_best else root

        return root

    def _on_intake(self):
        if not self.cfg:
            messagebox.showwarning("Not configured", "Run the setup wizard first.")
            return

        source = self._choose_source_with_autodetect()
        if not source:
            return

        self._set_busy(True)
        self._log(f"Starting intake from: {source}")

        def worker():
            try:
                def log_cb(m): self.queue.put(("log", m))
                def prog_cb(c, t): self.queue.put(("progress", (c, t)))

                counts, log_path = intake_media(self.cfg, source, log_cb, prog_cb, self.stop_flag)
                self.queue.put(("done", f"Intake complete. Summary: {counts}. Log: {log_path}"))

                def ask_upload():
                    if self.cfg and self.cfg.immich.enabled:
                        if messagebox.askyesno("Upload to Immich?", "Upload staged processed media to Immich now?"):
                            self._on_upload()
                self.after(0, ask_upload)

            except Exception as e:
                self.queue.put(("error", str(e)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _on_upload(self):
        if not self.cfg:
            messagebox.showwarning("Not configured", "Run the setup wizard first.")
            return
        if not self.cfg.immich.enabled:
            messagebox.showinfo("Immich disabled", "Enable Immich in settings to upload.")
            return

        stage = self.cfg.processed_stage_root
        if not stage.exists():
            messagebox.showinfo("Nothing to upload", f"Staging folder does not exist:\n{stage}")
            return

        if not messagebox.askyesno("Confirm upload", f"Upload everything in staging?\n\n{stage}"):
            return

        self._set_busy(True)
        self._log(f"Uploading staging: {stage}")

        def worker():
            try:
                logs_dir = self.cfg.config_root / "logs"
                ensure_dir(logs_dir)
                up_log = logs_dir / f"upload_{now_stamp()}.log"

                def log_cb(m):
                    self.queue.put(("log", m))
                    with up_log.open("a", encoding="utf-8") as f:
                        f.write(m + "\n")

                log_cb(f"{APP_NAME} upload run: {datetime.now().isoformat()}")
                log_cb(f"Staging: {stage}")
                log_cb(f"Immich URL: {self.cfg.immich.server_url}")

                # Load API key from secrets (plaintext or encrypted)
                passphrase = self.runtime_passphrase.get().strip() or None
                api_key = get_immich_api_key(self.cfg, passphrase)

                immich_login_and_upload(self.cfg, stage, api_key, log_cb)
                self.queue.put(("done", f"Upload complete. Log: {up_log}"))

                # Optional post-upload move
                if self.cfg.processed_uploaded_root:
                    def ask_move():
                        if messagebox.askyesno("Move staged files?", f"Move staged files to:\n{self.cfg.processed_uploaded_root}\n\n(now that upload succeeded)"):
                            self._post_upload_move(stage, self.cfg.processed_uploaded_root)
                    self.after(0, ask_move)

            except subprocess.CalledProcessError as e:
                self.queue.put(("error", f"Immich CLI failed: {e}"))
            except Exception as e:
                self.queue.put(("error", str(e)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _post_upload_move(self, stage: Path, uploaded_root: Path):
        self._set_busy(True)
        self._log(f"Moving staging -> uploaded: {uploaded_root}")

        def worker():
            try:
                logs_dir = self.cfg.config_root / "logs"
                ensure_dir(logs_dir)
                mv_log = logs_dir / f"postmove_{now_stamp()}.log"

                moved = 0
                with mv_log.open("a", encoding="utf-8") as f:
                    f.write(f"{APP_NAME} post-move: {datetime.now().isoformat()}\n")
                    f.write(f"From: {stage}\nTo:   {uploaded_root}\n\n")

                    for src in stage.rglob("*"):
                        if not src.is_file():
                            continue
                        rel = src.relative_to(stage)
                        dest_dir = uploaded_root / rel.parent
                        ensure_dir(dest_dir)
                        dst = safe_dest_path(dest_dir, src.name)
                        shutil.move(str(src), str(dst))
                        moved += 1
                        f.write(f"OK {src} -> {dst}\n")

                self.queue.put(("done", f"Moved {moved} files. Log: {mv_log}"))

            except Exception as e:
                self.queue.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    app = SensorSiftApp()
    app.mainloop()
