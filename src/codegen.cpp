#include "tc/codegen.hpp"
#include <sstream>
#include <unordered_map>

static void emit_loops(std::ostringstream& out, const std::vector<Loop>& loops,
                        const std::string& body, int depth) {
    std::string ind(depth * 4, ' ');
    if (loops.empty()) {
        for (const auto& line : {body})
            out << ind << line << "\n";
        return;
    }
    const Loop& l = loops[0];
    if (l.tag == LoopTag::Parallel)
        out << ind << "#pragma omp parallel for\n";
    out << ind << "for (int64_t " << l.var << " = 0; "
        << l.var << " < " << l.extent << "; ++" << l.var << ") {\n";

    std::vector<Loop> rest(loops.begin() + 1, loops.end());
    emit_loops(out, rest, body, depth + 1);

    out << ind << "}\n";
}

std::string emit_cpp(const LoopProgram& prog) {
    std::ostringstream out;

    out << "#include <cmath>\n"
           "#include <cstdint>\n"
           "#include <cstdio>\n"
           "#include <cstdlib>\n\n";

    for (const LoopNest& k : prog.kernels) {
        out << "static void " << k.name << "(";
        bool first = true;
        for (auto bid : k.reads) {
            if (!first) out << ", ";
            out << "const float* " << prog.buffers[bid].name;
            first = false;
        }
        for (auto bid : k.writes) {
            if (!first) out << ", ";
            out << "float* " << prog.buffers[bid].name;
            first = false;
        }
        out << ") {\n";
        emit_loops(out, k.loops, k.body, 1);
        out << "}\n\n";
    }

    // Identify external (input) and output buffers in order
    std::vector<const Buffer*> ext_bufs, out_bufs;
    for (const Buffer& b : prog.buffers) {
        if (b.role == BufferRole::External) ext_bufs.push_back(&b);
        if (b.role == BufferRole::Output)   out_bufs.push_back(&b);
    }

    // main(): reads inputs from binary files, writes outputs to binary files.
    // argv: input0.bin input1.bin ... output0.bin output1.bin ...
    out << "int main(int argc, char* argv[]) {\n";
    out << "    if (argc != " << (ext_bufs.size() + out_bufs.size() + 1) << ") return 1;\n";
    out << "    int arg = 1;\n\n";

    for (const Buffer* b : ext_bufs) {
        int64_t n = b->shape.numel();
        out << "    float* " << b->name << " = (float*)malloc(" << n << " * sizeof(float));\n";
        out << "    { FILE* f = fopen(argv[arg++], \"rb\"); fread(" << b->name
            << ", sizeof(float), " << n << ", f); fclose(f); }\n";
    }
    out << "\n";

    for (const Buffer* b : out_bufs) {
        int64_t n = b->shape.numel();
        out << "    float* " << b->name << " = (float*)malloc(" << n << " * sizeof(float));\n";
    }

    std::vector<int64_t> slot_sizes;
    for (const Buffer& b : prog.buffers) {
        if (b.role != BufferRole::Intermediate) continue;
        if (b.slot_id >= 0) {
            if (b.slot_id >= static_cast<int32_t>(slot_sizes.size()))
                slot_sizes.resize(b.slot_id + 1, 0);
            if (b.shape.numel() > slot_sizes[b.slot_id])
                slot_sizes[b.slot_id] = b.shape.numel();
        }
    }
    for (int32_t si = 0; si < static_cast<int32_t>(slot_sizes.size()); ++si)
        out << "    float* _slot_" << si << " = (float*)malloc("
            << slot_sizes[si] << " * sizeof(float));\n";
    for (const Buffer& b : prog.buffers) {
        if (b.role != BufferRole::Intermediate) continue;
        if (b.slot_id >= 0)
            out << "    float* " << b.name << " = _slot_" << b.slot_id << ";\n";
        else
            out << "    float* " << b.name << " = (float*)malloc("
                << b.shape.numel() << " * sizeof(float));\n";
    }
    out << "\n";

    for (const LoopNest& k : prog.kernels) {
        out << "    " << k.name << "(";
        bool first = true;
        for (auto bid : k.reads) {
            if (!first) out << ", ";
            out << prog.buffers[bid].name;
            first = false;
        }
        for (auto bid : k.writes) {
            if (!first) out << ", ";
            out << prog.buffers[bid].name;
            first = false;
        }
        out << ");\n";
    }
    out << "\n";

    for (const Buffer* b : out_bufs) {
        out << "    { FILE* f = fopen(argv[arg++], \"wb\"); fwrite(" << b->name
            << ", sizeof(float), " << b->shape.numel() << ", f); fclose(f); }\n";
    }

    for (int32_t si = 0; si < static_cast<int32_t>(slot_sizes.size()); ++si)
        out << "    free(_slot_" << si << ");\n";
    for (const Buffer& b : prog.buffers)
        if (!(b.role == BufferRole::Intermediate && b.slot_id >= 0))
            out << "    free(" << b.name << ");\n";

    out << "    return 0;\n}\n";

    return out.str();
}

std::string emit_bench_cpp(const LoopProgram& prog, int warmup, int reps) {
    std::ostringstream out;

    out << "#include <cmath>\n"
           "#include <cstdint>\n"
           "#include <cstdio>\n"
           "#include <cstdlib>\n"
           "#include <chrono>\n"
           "#include <algorithm>\n\n";

    for (const LoopNest& k : prog.kernels) {
        out << "static void " << k.name << "(";
        bool first = true;
        for (auto bid : k.reads) {
            if (!first) out << ", ";
            out << "const float* " << prog.buffers[bid].name;
            first = false;
        }
        for (auto bid : k.writes) {
            if (!first) out << ", ";
            out << "float* " << prog.buffers[bid].name;
            first = false;
        }
        out << ") {\n";
        emit_loops(out, k.loops, k.body, 1);
        out << "}\n\n";
    }

    std::vector<const Buffer*> ext_bufs, out_bufs;
    for (const Buffer& b : prog.buffers) {
        if (b.role == BufferRole::External) ext_bufs.push_back(&b);
        if (b.role == BufferRole::Output)   out_bufs.push_back(&b);
    }

    auto emit_kernel_call = [&](std::ostringstream& o, const std::string& indent) {
        for (const LoopNest& k : prog.kernels) {
            o << indent << k.name << "(";
            bool first = true;
            for (auto bid : k.reads) {
                if (!first) o << ", ";
                o << prog.buffers[bid].name;
                first = false;
            }
            for (auto bid : k.writes) {
                if (!first) o << ", ";
                o << prog.buffers[bid].name;
                first = false;
            }
            o << ");\n";
        }
    };

    out << "int main(int argc, char* argv[]) {\n";
    out << "    if (argc != " << (ext_bufs.size() + out_bufs.size() + 1) << ") return 1;\n";
    out << "    int arg = 1;\n\n";

    for (const Buffer* b : ext_bufs) {
        int64_t n = b->shape.numel();
        out << "    float* " << b->name << " = (float*)malloc(" << n << " * sizeof(float));\n";
        out << "    { FILE* f = fopen(argv[arg++], \"rb\"); fread(" << b->name
            << ", sizeof(float), " << n << ", f); fclose(f); }\n";
    }
    out << "\n";

    for (const Buffer* b : out_bufs) {
        int64_t n = b->shape.numel();
        out << "    float* " << b->name << " = (float*)malloc(" << n << " * sizeof(float));\n";
    }
    std::vector<int64_t> bslot_sizes;
    for (const Buffer& b : prog.buffers) {
        if (b.role != BufferRole::Intermediate) continue;
        if (b.slot_id >= 0) {
            if (b.slot_id >= static_cast<int32_t>(bslot_sizes.size()))
                bslot_sizes.resize(b.slot_id + 1, 0);
            if (b.shape.numel() > bslot_sizes[b.slot_id])
                bslot_sizes[b.slot_id] = b.shape.numel();
        }
    }
    for (int32_t si = 0; si < static_cast<int32_t>(bslot_sizes.size()); ++si)
        out << "    float* _slot_" << si << " = (float*)malloc("
            << bslot_sizes[si] << " * sizeof(float));\n";
    for (const Buffer& b : prog.buffers) {
        if (b.role != BufferRole::Intermediate) continue;
        if (b.slot_id >= 0)
            out << "    float* " << b.name << " = _slot_" << b.slot_id << ";\n";
        else
            out << "    float* " << b.name << " = (float*)malloc("
                << b.shape.numel() << " * sizeof(float));\n";
    }
    out << "\n";

    out << "    for (int w = 0; w < " << warmup << "; ++w) {\n";
    emit_kernel_call(out, "        ");
    out << "    }\n\n";

    out << "    double times[" << reps << "];\n";
    out << "    for (int r = 0; r < " << reps << "; ++r) {\n";
    out << "        auto _tc0 = std::chrono::high_resolution_clock::now();\n";
    emit_kernel_call(out, "        ");
    out << "        auto _tc1 = std::chrono::high_resolution_clock::now();\n";
    out << "        times[r] = std::chrono::duration<double, std::milli>(_tc1 - _tc0).count();\n";
    out << "    }\n";
    out << "    std::sort(times, times + " << reps << ");\n";
    out << "    printf(\"TIMING_MS: %.3f\\n\", times[" << reps / 2 << "]);\n\n";

    for (const Buffer* b : out_bufs) {
        out << "    { FILE* f = fopen(argv[arg++], \"wb\"); fwrite(" << b->name
            << ", sizeof(float), " << b->shape.numel() << ", f); fclose(f); }\n";
    }

    for (int32_t si = 0; si < static_cast<int32_t>(bslot_sizes.size()); ++si)
        out << "    free(_slot_" << si << ");\n";
    for (const Buffer& b : prog.buffers)
        if (!(b.role == BufferRole::Intermediate && b.slot_id >= 0))
            out << "    free(" << b.name << ");\n";

    out << "    return 0;\n}\n";
    return out.str();
}
