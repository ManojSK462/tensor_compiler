#include "tc/runtime.hpp"
#include <fstream>
#include <cstdlib>
#include <stdexcept>
#include <windows.h>

static const std::string GEN_DIR = "K:\\tensor_compiler\\generated\\";

static std::string gen(const std::string& name) {
    return GEN_DIR + name;
}

static void write_bin(const std::string& path, const std::vector<float>& data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("write_bin: cannot open " + path);
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

static std::vector<float> read_bin(const std::string& path, int64_t n) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("read_bin: cannot open " + path);
    std::vector<float> v(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
    return v;
}

static bool file_exists(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fclose(f);
    return true;
}

RunResult compile_and_run(const std::string& src,
                          const LoopProgram& prog,
                          const std::vector<std::vector<float>>& inputs) {
    RunResult res{false, "", {}};

    std::string cpp_path     = gen("tc_gen.cpp");
    std::string exe_path     = gen("tc_gen.exe");
    std::string compile_bat  = gen("compile.bat");
    std::string run_bat      = gen("run.bat");

    { std::ofstream f(cpp_path); if (!f) { res.error = "cannot write " + cpp_path; return res; } f << src; }

    {
        std::ofstream bat(compile_bat);
        if (!bat) { res.error = "cannot write " + compile_bat; return res; }
        bat << "@echo off\r\n";
        bat << "call \"C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools"
               "\\VC\\Auxiliary\\Build\\vcvars64.bat\"\r\n";
        bat << "cl.exe /O2 /nologo /Fe:\"" << exe_path << "\" \""
            << cpp_path << "\"\r\n";
    }

    int compile_ret = system(compile_bat.c_str());
    if (compile_ret != 0 || !file_exists(exe_path)) {
        res.error = "compilation failed (see " + cpp_path + ")";
        return res;
    }

    std::vector<const Buffer*> ext_bufs, out_bufs;
    for (const Buffer& b : prog.buffers) {
        if (b.role == BufferRole::External) ext_bufs.push_back(&b);
        if (b.role == BufferRole::Output)   out_bufs.push_back(&b);
    }

    if (inputs.size() != ext_bufs.size()) {
        res.error = "input count mismatch";
        return res;
    }

    std::vector<std::string> in_paths, out_paths;
    for (size_t i = 0; i < ext_bufs.size(); ++i) {
        std::string p = gen("in" + std::to_string(i) + ".bin");
        write_bin(p, inputs[i]);
        in_paths.push_back(p);
    }
    for (size_t i = 0; i < out_bufs.size(); ++i)
        out_paths.push_back(gen("out" + std::to_string(i) + ".bin"));

    {
        std::ofstream bat(run_bat);
        if (!bat) { res.error = "cannot write " + run_bat; return res; }
        bat << "@echo off\r\n";
        bat << "\"" << exe_path << "\"";
        for (auto& p : in_paths)  bat << " \"" << p << "\"";
        for (auto& p : out_paths) bat << " \"" << p << "\"";
        bat << "\r\n";
    }

    int run_ret = system(run_bat.c_str());
    if (run_ret != 0) {
        res.error = "execution failed (exit code " + std::to_string(run_ret) + ")";
        return res;
    }

    for (size_t i = 0; i < out_bufs.size(); ++i)
        res.outputs.push_back(read_bin(out_paths[i], out_bufs[i]->shape.numel()));

    res.ok = true;
    return res;
}
