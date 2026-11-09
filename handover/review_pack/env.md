# Environment Specification & Hardware Profile

## 1. Host Hardware & Operating System
- **Operating System**: macOS (Darwin 27.2.0, Kernel Version 27.2.0, root:xnu-13432.41.4~25/RELEASE_ARM64_T6041)
- **Product Details**: macOS 27.2 (Build 26B123)
- **Architecture**: `arm64` (Apple Silicon)
- **Processor / Chip**: Apple M4 Pro (12 CPU cores)
- **Unified RAM**: 24.00 GB (25,769,803,776 bytes)
- **Disk Storage (Working Directory Volume)**:
  - Total Size: 460 GiB
  - Used Space: 353 GiB
  - Available Free Space: 79 GiB (82% capacity)

## 2. Python & Acceleration Runtime
- **Python Executable**: `/Users/agamghotra/Desktop/ML/.venv/bin/python`
- **Python Version**: 3.13.7
- **PyTorch Version**: 2.14.1
- **PyTorch Devices**:
  - `torch.backends.mps.is_available()`: `True` (Apple Silicon GPU via Metal Performance Shaders)
  - `torch.backends.mps.is_built()`: `True`
  - `torch.cuda.is_available()`: `False`
  - Training Device: `mps` (with `PYTORCH_ENABLE_MPS_FALLBACK=1` and `float32`)
  - Evaluation / Latency Device: Pinned single-thread CPU (`Apple M4 Pro`) & `mps`

## 3. Audio Codec & FFmpeg Tooling
- **FFmpeg Binary**: `/opt/homebrew/bin/ffmpeg`
- **FFmpeg Version**: `ffmpeg version 7.1.1 Copyright (c) 2000-2025 the FFmpeg developers`
- **Available Audio Encoders**:
  - MP3: `libmp3lame` (MP3 - MPEG audio layer 3)
  - AAC: `aac`, `aac_at` (AudioToolbox AAC)
  - OPUS: `libopus`, `opus`
  - FLAC: `flac` (Free Lossless Audio Codec)
  - Telephony G.711: `pcm_alaw`, `pcm_alaw_at`, `pcm_mulaw`, `pcm_mulaw_at`
  - PCM: `pcm_s16le`, `pcm_f32le`, `pcm_s24le`, `pcm_s32le`

## 4. Complete Python Package Manifest (`pip freeze`)
```text
-e git+https://github.com/AgamGhotra1903/VeriVox.git@f02eeda4ce93371da9f141220c699ba4c9c503e0#egg=acforensics
altair==6.3.0
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
attrs==26.1.0
certifi==2026.7.22
cffi==2.1.1
charset-normalizer==3.5.2
click==8.5.0
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
fastapi==0.162.0
filelock==4.2.0
flatbuffers==25.12.19
fonttools==4.67.0
fsspec==2026.9.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.20
iniconfig==2.3.1
itsdangerous==2.2.0
Jinja2==3.1.6
joblib==1.6.0
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
kiwisolver==1.5.1
MarkupSafe==3.0.4
matplotlib==3.11.2
ml_dtypes==0.6.0
mpmath==1.3.0
narwhals==2.27.1
networkx==3.7
numpy==2.2.6
onnx==1.23.2
onnx-ir==1.1.0
onnxruntime==1.31.1
onnxscript==0.7.2
opentelemetry-api==1.45.2
packaging==26.3
pandas==3.0.6
pillow==12.3.0
plotly==7.1.0
pluggy==1.7.0
protobuf==7.36.2
pyarrow==26.0.0
pycparser==3.11
pydantic==2.14.2
pydantic_core==2.50.2
pydeck==0.9.3
Pygments==2.21.0
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==2.9.0.post0
python-multipart==0.0.32
PyYAML==6.0.3
referencing==0.37.0
requests==2.34.2
rpds-py==2026.9.1
scikit-learn==1.9.1
scipy==1.18.1
seaborn==0.13.2
setuptools==84.0.0
six==1.17.0
soundfile==0.14.0
sse-starlette==3.6.0
starlette==1.8.0
streamlit==1.66.0
sympy==1.14.0
tabulate==0.10.0
threadpoolctl==3.7.0
torch==2.14.1
torchaudio==2.11.0
tqdm==4.70.1
typing-inspection==0.4.4
typing_extensions==4.16.0
urllib3==2.8.0
uvicorn==0.54.0
websockets==17.2
```

