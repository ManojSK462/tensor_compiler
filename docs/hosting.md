# Hosting the Compiler on a Linux Server

The local dev build targets Windows + MSVC. To host behind a web UI on a Linux server, one file needs rewriting and a few paths need updating. Everything else (graph IR, fusion, lowering, codegen, all headers) is portable C++17 and requires no changes.

---

## What changes

### `src/runtime.cpp`

This is the only file with platform-specific code. Replace the bat-file + cl.exe approach with shell scripts + g++.

**compile_and_run** — change the compile section from:

```cpp
bat << "call \"...vcvars64.bat\"\r\n";
bat << "cl.exe /O2 /openmp /nologo /Fe:\"" << exe_path << "\" \"" << cpp_path << "\"\r\n";
// system(compile_bat.c_str());
```

to writing a `.sh` and calling g++:

```cpp
std::string compile_sh = gen("compile.sh");
{ std::ofstream sh(compile_sh);
  sh << "#!/bin/sh\n";
  sh << "g++ -O2 -fopenmp -std=c++17 -o \"" << exe_path << "\" \"" << cpp_path << "\"\n"; }
chmod(compile_sh.c_str(), 0755);
int compile_ret = system(compile_sh.c_str());
```

Same pattern for `compile_and_bench` (uses `bench_compile.sh`).

**run.bat → run.sh**

```cpp
std::string run_sh = gen("run.sh");
{ std::ofstream sh(run_sh);
  sh << "#!/bin/sh\n";
  sh << "\"" << exe_path << "\"";
  for (auto& p : in_paths)  sh << " \"" << p << "\"";
  for (auto& p : out_paths) sh << " \"" << p << "\"";
  sh << "\n"; }
chmod(run_sh.c_str(), 0755);
system(run_sh.c_str());
```

For bench, redirect stdout: `sh << " > \"" << stdout_path << "\"\n";`

### `GEN_DIR` path

Change from `K:\\tensor_compiler\\generated\\` to a writable server path, e.g.:

```cpp
static const std::string GEN_DIR = "/var/tc/generated/";
```

Create the directory on the server before starting the service.

### `#include <sys/stat.h>`

Needed for `chmod()`. Add to the top of `runtime.cpp`.

---

## Server setup (one-time)

```bash
apt-get install -y g++ libomp-dev     # Ubuntu/Debian
# or
yum install -y gcc-c++ libgomp        # RHEL/CentOS

mkdir -p /var/tc/generated
chmod 777 /var/tc/generated
```

`g++` includes OpenMP via `libgomp` by default on Linux — the `-fopenmp` flag is equivalent to MSVC's `/openmp`.

---

## CMakeLists.txt

The CMake build (for building the compiler itself, not the generated kernels) already uses standard C++17. On Linux, cmake picks up g++ automatically. No changes needed.

---

## What does NOT change

- All headers (`include/tc/`)
- `src/ops.cpp`, `src/graph.cpp`, `src/passes.cpp`, `src/printer.cpp`
- `src/loop_ir.cpp`, `src/lower.cpp`, `src/codegen.cpp`
- `src/fusion.cpp`, `src/analyze.cpp`, `src/buffer_reuse.cpp`
- All test files
- The generated C++ output itself (standard C++ with OpenMP pragmas, compiles on any platform)

---

## Local dev vs server

Keep `runtime.cpp` as the Windows version in the repo. When deploying, either:

- Maintain a `src/runtime_linux.cpp` alongside it and select via CMake (`if(WIN32) ... else() ...`)
- Or use a simple `#ifdef _WIN32` guard inside a single `runtime.cpp`

The `#ifdef` approach is less noise and easier to maintain for a small project.
