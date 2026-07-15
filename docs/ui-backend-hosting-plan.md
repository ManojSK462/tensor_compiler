# UI, Backend & Hosting Plan

Full architecture for the online compiler service. Users paste PyTorch code, the compiler runs server-side, and the browser shows every pipeline stage plus benchmark results.

---

## Pipeline overview

```
PyTorch code → ONNX Graph → Graph IR → Loop IR → C++ source → Benchmark
    (s1)           (s2)        (s3)       (s4)        (s5)
```

---

## UI

### Layout

Split pane: Monaco code editor on the left, pipeline viewer on the right. The pipeline viewer has five tabs (ONNX, Graph IR, Loop IR, C++ source, Benchmark) that unlock in sequence as each stage completes. Earlier tabs stay readable while later ones show a loading state.

### Panel specs

**ONNX Graph** — table of ONNX nodes: op name, inputs, outputs, key attributes. Header line shows total op count, opset version, and export time. No visual DAG at v1.

**Graph IR** — our node list with fusion groups color-coded. Each group gets a background band and a left border in the group's color. Shows: node id, op kind, output shape, group id.

**Loop IR** — one collapsible block per kernel. Header: kernel name + member ops (e.g., `matmul+relu`). Expanded: loop nest as indented pseudocode with parallelism labels and trip counts.

**C++ Source** — full generated `tc_gen.cpp`, syntax highlighted, read-only. Copy button in top-right.

**Benchmark** — CSS bar chart: fused vs. naive execution time. Three stat chips below: speedup (×), memory traffic saved (%), fusion groups count.

### Frontend tech

- Framework: vanilla JS (no build pipeline for v1)
- Code editor: Monaco Editor (CDN, Python + read-only C++ modes)
- Benchmark chart: CSS bars, no library
- Streaming: Server-Sent Events so each stage appears as it completes
- Static hosting: single `index.html` served by Nginx from `/var/www/tc/`

---

## Backend

### Data flow

```
Browser
  → POST /compile  {code, warmup, reps}
  → API Server (FastAPI, Python)
  → spawns Docker sandbox container
  → user code runs, tc.compile() fires torch.onnx.export()
  → ONNX written to shared volume
  → tc_compile binary (C++) reads ONNX, runs full pipeline
  → JSON on stdout: {onnx_ops, graph_ir, loop_ir, generated_cpp, benchmark}
  → API server streams SSE events back to browser
```

### New C++ files

**`src/frontend_onnx.cpp` + `include/tc/frontend_onnx.hpp`**

Parses ONNX protobuf using the `onnx` C++ library. Walks `ModelProto.graph().node()` and calls our Graph builder API for each supported op.

Op mapping:
```
MatMul       → g.matmul(A, B)
Gemm         → g.matmul(A, B, trans_b) then g.add(result, bias)
Add          → g.add(a, b)
Mul          → g.mul(a, b)
Relu         → g.relu(x)
Sigmoid      → g.sigmoid(x)
ReduceSum    → g.sum(x, axis)
ReduceMean   → g.mean(x, axis)
ReduceMax    → g.max(x, axis)
Softmax      → g.softmax(x, axis)
Reshape      → g.reshape(x, new_shape)
Transpose    → g.transpose(x, perm)
```

Function signature:
```cpp
Graph graph_from_onnx(const std::string& onnx_path, std::string& err);
```

**`src/serializers.cpp` + `include/tc/serializers.hpp`**

Converts our IRs to JSON strings for the API response. Hand-built strings (no external JSON library — the IR is small and structured).

```cpp
std::string graph_ir_to_json(const Graph& g);
std::string loop_ir_to_json(const LoopProgram& prog);
```

**`src/main_cli.cpp`** — `tc_compile` binary

Reads one argument (ONNX file path) plus options (`--warmup N --reps N`), runs the full pipeline, writes JSON to stdout:

```json
{
  "onnx_ops":      [{op, inputs, outputs}],
  "graph_ir":      [{id, kind, shape, group}],
  "loop_ir":       [{name, members, loop_pseudocode}],
  "generated_cpp": "static void kernel_0(...) { ... }",
  "benchmark": {
    "naive_ms": 14.7,
    "fused_ms": 6.5,
    "speedup": 2.26,
    "bytes_unfused": 786432,
    "bytes_fused": 344064
  }
}
```

### Python shim — `tc` package

Installed inside the sandbox Docker image. User calls `tc.compile(model, x)` at the end of their script.

```python
# python/tc/__init__.py
import torch, os

def compile(model, example_input, output_dir="/tc_out"):
    path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(
        model, example_input, path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}}
    )
```

### Sandbox

```dockerfile
# docker/sandbox.Dockerfile
FROM python:3.11-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY python/tc /usr/local/lib/python3.11/site-packages/tc
CMD ["python", "/tc_in/user_code.py"]
```

Container launch (from API server):
```bash
docker run --rm \
  --network none \
  --memory 2g --cpus 1 \
  --read-only \
  --tmpfs /tmp:size=64m \
  -v /var/tc/jobs/<id>/in:/tc_in:ro \
  -v /var/tc/jobs/<id>/out:/tc_out \
  tc-sandbox \
  timeout 30 python /tc_in/user_code.py
```

`--network none` blocks all outbound traffic. User code cannot reach the internet or host filesystem.

### API contract

```
POST /compile
Content-Type: application/json
{ "code": "...", "warmup": 2, "reps": 10 }

Response: text/event-stream (SSE)
data: {"stage":"sandbox_done"}
data: {"stage":"onnx_done",    "ops":[...]}
data: {"stage":"graph_ir_done","nodes":[...]}
data: {"stage":"loop_ir_done", "kernels":[...]}
data: {"stage":"codegen_done", "cpp":"..."}
data: {"stage":"bench_done",   "naive_ms":14.7,"fused_ms":6.5,"speedup":2.26,"bytes_saved":442368}
data: {"stage":"done"}
```

### New files

```
src/
  frontend_onnx.cpp    NEW
  serializers.cpp      NEW
  main_cli.cpp         NEW
include/tc/
  frontend_onnx.hpp    NEW
  serializers.hpp      NEW
python/tc/
  __init__.py          NEW
server/
  app.py               NEW  (FastAPI)
docker/
  sandbox.Dockerfile   NEW
web/
  index.html           NEW
  style.css            NEW
```

CMakeLists.txt: add `add_executable(tc_compile src/main_cli.cpp)` linked against `tc` and `onnx`.

---

## Linux Hosting

### Server

Ubuntu 22.04 LTS · 4 GB RAM · 2 vCPU · 40 GB SSD. DigitalOcean, Hetzner, or Fly.io Machines.

Process layout: Nginx (443) → Gunicorn+FastAPI (localhost:8000). Docker on the same host.

### runtime.cpp changes

See `hosting.md` for the full diff. Summary:
- `GEN_DIR`: `K:\\tensor_compiler\\generated\\` → `/var/tc/generated/`
- Compile: `compile.bat` + `cl.exe /O2 /openmp` → `compile.sh` + `g++ -O2 -fopenmp -std=c++17`
- Run: `run.bat` → `run.sh` with `chmod(0755)`
- Add `#include <sys/stat.h>`

### Directory layout

```
/var/tc/
  generated/       ← GEN_DIR: kernel .cpp/.exe/.sh files
  jobs/<uuid>/
    in/user_code.py
    out/model.onnx
  bin/tc_compile   ← compiled CLI binary

/opt/tc/
  server/app.py
  venv/

/var/www/tc/
  index.html
  style.css
```

### One-time setup

```bash
apt-get install -y g++ libomp-dev cmake docker.io nginx certbot python3-certbot-nginx

cmake -S /opt/tc/compiler -B /opt/tc/build -DCMAKE_BUILD_TYPE=Release
cmake --build /opt/tc/build
cp /opt/tc/build/tc_compile /var/tc/bin/

docker build -t tc-sandbox -f docker/sandbox.Dockerfile .

python3 -m venv /opt/tc/venv
/opt/tc/venv/bin/pip install fastapi uvicorn[standard]

mkdir -p /var/tc/{generated,jobs,bin} /var/www/tc
certbot --nginx -d tc-compiler.dev
```

### Nginx (key parts)

```nginx
location / {
    root /var/www/tc;
    try_files $uri /index.html;
}

location /compile {
    proxy_pass         http://127.0.0.1:8000;
    proxy_buffering    off;
    proxy_cache        off;
    proxy_read_timeout 120s;
    limit_req          zone=api burst=5 nodelay;
}
```

### systemd service

```ini
[Service]
User=tc
ExecStart=/opt/tc/venv/bin/gunicorn \
    -w 2 -k uvicorn.workers.UvicornWorker \
    -b 127.0.0.1:8000 server.app:app
Restart=always
```

---

## Implementation sequence

1. `frontend_onnx.cpp` — ONNX → Graph IR (test on command line, no server needed)
2. `serializers.cpp` — Graph IR + Loop IR → JSON strings
3. `main_cli.cpp` — wire full pipeline, test with a local ONNX file
4. `runtime.cpp` Linux update — build on Linux, run all tests
5. Python shim + Dockerfile — verify ONNX appears in output mount
6. `server/app.py` — FastAPI, Docker spawn, SSE streaming
7. `web/index.html` — Monaco + pipeline tabs + benchmark panel
8. Deploy — provision VPS, one-time setup, DNS, systemd start
