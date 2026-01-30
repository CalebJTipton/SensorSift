# SensorSift

SensorSift is a lightweight Tkinter-based helper for photographers and creators to help with the automation of intaking camera files into RAW vs processed buckets, stage processed clips for Immich uploads, and optionally encrypt Immich API keys. It watches entire folders or SD cards, auto-detects common media roots, and writes organized archives and staging folders with timestamped subdirectories.

## Prerequisites

- Python 3.9+ (any system that can run Tkinter; Windows, macOS, and Linux are supported)
- [Pillow](https://python-pillow.org/) (`pip install pillow`) for EXIF inspection
- [Cryptography](https://cryptography.io/) (`pip install cryptography`) for optional API key encryption
- Optional tools:
  - `exiftool` on your PATH for more accurate timestamps on exotic RAW/video formats
  - Immich CLI (`immich`) if you plan to push staged media to an Immich server

## Get started

1. Install the dependencies listed above.
2. Run the GUI: `python snapsift_gui.py`.
3. The first launch shows the **Setup Wizard**; provide:
   - A synced config folder (OneDrive/Dropbox/iCloud suggestions) so SensorSift keeps settings consistent.
   - RAW archive, Immich staging, and optional post-upload folders.
   - Copy vs move mode, extension lists, and routing rules if you need device- or folder-specific behavior.
   - Optional Immich credentials (store plaintext or encrypted with a passphrase).
   - A friend/event tag and location that describe this intake—SensorSift prepends them to the RAW archive tree so you can drill down by shoot or trip.
4. Save the wizard. SensorSift writes `config.json` + `secrets.json` under your chosen config root and logs pointer info in `~/.sensorsift/pointer.json`.

## UI workflow

SensorSift’s main window keeps the sequence simple:

1. **Configure** – use the *Settings / Wizard* button to revisit paths, rules, and Immich details whenever your archive layout changes.
2. **Intake media** – select an SD card or folder; SensorSift auto-detects the best media subfolder and copies/moves files into timestamped RAW, processed, or unknown folders defined by your routing rules.
   - Raw files land beneath `<RAW archive>/<photo|video>/<friend tag>/<location>/<year>/<year-month>/…`, so the bucket type comes first and the metadata you entered still follows before the timestamp folders.
3. **Upload staging** – push the processed staging folder to Immich via the configured CLI. After a successful upload you can optionally move the staged files into your uploaded archive.
4. **Monitor progress** – the workflow hints section explains each step, the progress bar keeps you updated, and logs appear in the text area and in `logs/` inside your config root.

The passphrase entry appears near the workflow area only when your secrets are encrypted, and the stop button lets you cancel long intake runs safely.

## Tips & troubleshooting

- Logs are written to `<config_root>/logs/` so you can review exact copy/move/upload details.
- Routing rules are evaluated top to bottom; the first matching rule sets the destination bucket (`raw`, `processed`, or `unknown`).
- RAW intake files are automatically written under `<RAW archive>/<friend tag>/<location>/<photo|video>/<year>/<year-month>/…`, so the metadata you just entered lives in the folder tree and keeps raw clips separated from stills.
- Immich uploads respect the CLI path you set, so point it at a virtual environment or `immich` binary if it isn’t on your `PATH`.
- Use the optional **Uploaded folder** path to stash files once Immich succeeds.
- Check the passphrase field before uploading if you encrypted your API key. SensorSift never stores the passphrase.
