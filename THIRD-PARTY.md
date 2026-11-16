# Third-party files

Files in `web/` that we did not write. They are served from the page's own host, so no script or font is fetched from a CDN. Licence texts are in `web/pyodide/licenses/` unless stated; `web/pyodide/licenses/SOURCES.md` gives each text's upstream source and hash. The wheels are unmodified.

| files | project and version | licence | licence text |
|---|---|---|---|
| `web/pyodide/pyodide.mjs`, `pyodide.asm.mjs`, `pyodide.asm.wasm`, `pyodide-lock.json` | Pyodide (runtime for Python 3.14.2, ABI 2026_0) | MPL-2.0 | `web/pyodide/LICENSE-Pyodide-MPL-2.0.txt` |
| CPython inside the Pyodide runtime, and `web/pyodide/python_stdlib.zip` | CPython 3.14.2 | PSF License Version 2 (and the history notices in the text) | `LICENSE-CPython-3.14.2.txt` |
| `web/pyodide/numpy-2.4.6-*.whl` | NumPy 2.4.6 | BSD-3-Clause, with bundled licences | inside the wheel (`numpy-2.4.6.dist-info/`) |
| `web/pyodide/pyarrow-22.0.0-*.whl` | Apache Arrow 22.0.0 | Apache-2.0, with bundled third-party notices | `LICENSE-Arrow-22.0.0.txt`, `NOTICE-Arrow-22.0.0.txt` (the wheel carries none) |
| `web/pyodide/pyodide_unix_timezones-1.0.0-*.whl` | pyodide-unix-timezones 1.0.0 | MIT | `LICENSE-pyodide-unix-timezones.txt` |
| `web/fonts/DMSans-*`, `DMMono-*` | DM Sans, DM Mono | SIL Open Font License 1.1 | `web/fonts/LICENSE-DM-*.txt` |
| `web/fonts/Syne-*` | Syne | SIL Open Font License 1.1 | `web/fonts/LICENSE-Syne.txt` |

`holdcheck/tipmap.json` is derived from the SO-101 model in MuJoCo Menagerie, `robotstudio_so101/so101.xml` (Apache-2.0). The model file ships in `holdcheck/model/` with its licence and `NOTICE.md`. The corpus figures are derived from `lerobot/community_dataset_v3` (Apache-2.0).
