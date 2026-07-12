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

    for (const Buffer& b : prog.buffers) {
        if (b.role == BufferRole::Intermediate) {
            out << "    float* " << b.name << " = (float*)malloc("
                << b.shape.numel() << " * sizeof(float));\n";
        }
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

    for (const Buffer& b : prog.buffers)
        out << "    free(" << b.name << ");\n";

    out << "    return 0;\n}\n";

    return out.str();
}
