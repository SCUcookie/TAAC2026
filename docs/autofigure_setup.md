# AutoFigure-Edit Setup

AutoFigure-Edit is installed under:

```text
tools/AutoFigure-Edit
```

It is the editable-SVG successor to AutoFigure for professional scientific illustrations.

## Current Local Status

- Repository cloned from `https://github.com/ResearAI/AutoFigure-Edit`.
- Local virtual environment created at `tools/AutoFigure-Edit/.venv`.
- Python dependencies are mostly installed.
- CLI help works:

```powershell
cd tools\AutoFigure-Edit
.\.venv\Scripts\python.exe autofigure2.py --help
```

- Web server health check passed at:

```text
http://127.0.0.1:8000/healthz
```

Open the UI:

```text
http://127.0.0.1:8000
```

## Start Server

```powershell
cd tools\AutoFigure-Edit
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --no-access-log
```

## Required Keys

Edit `tools/AutoFigure-Edit/.env`.

At minimum, real generation needs one LLM/image route:

- `OPENAI_API_KEY`, or
- OpenRouter/custom provider key/base URL, or
- Gemini key through the UI/CLI.

For full segmentation/background removal:

- `HF_TOKEN` with access to `briaai/RMBG-2.0`
- `ROBOFLOW_API_KEY` or `FAL_KEY` if using API SAM backend

## Notes

- Docker is installed, but Docker Desktop Linux engine was not running. `docker compose up -d --build` failed until Docker Desktop is started.
- On Windows, `cairosvg` import currently fails because the native Cairo DLL is not discoverable. The main AutoFigure CLI/server still imports; local SVG rendering helpers may be limited until GTK/Cairo runtime is installed or Docker is used.
