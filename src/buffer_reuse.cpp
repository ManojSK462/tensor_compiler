#include "tc/buffer_reuse.hpp"
#include <vector>
#include <algorithm>
#include <unordered_map>

void assign_buffer_slots(LoopProgram& prog) {
    int32_t nk = static_cast<int32_t>(prog.kernels.size());

    struct Interval {
        BufferId bid;
        int32_t  first_write;
        int32_t  last_read;
        int64_t  elements;
    };

    std::vector<Interval> ivs;
    for (const Buffer& b : prog.buffers) {
        if (b.role != BufferRole::Intermediate) continue;
        int32_t fw = nk, lr = -1;
        for (int32_t ki = 0; ki < nk; ++ki) {
            for (BufferId w : prog.kernels[ki].writes)
                if (w == b.id && ki < fw) fw = ki;
            for (BufferId r : prog.kernels[ki].reads)
                if (r == b.id && ki > lr) lr = ki;
        }
        if (fw >= nk || lr < 0) continue;
        ivs.push_back({b.id, fw, lr, b.shape.numel()});
    }

    std::sort(ivs.begin(), ivs.end(),
              [](const Interval& a, const Interval& b) {
                  return a.first_write < b.first_write;
              });

    struct Slot { int32_t last_read; int64_t max_elems; };
    std::vector<Slot> slots;
    std::unordered_map<BufferId, int32_t> assignment;

    for (const Interval& iv : ivs) {
        int32_t chosen = -1;
        for (int32_t si = 0; si < static_cast<int32_t>(slots.size()); ++si) {
            if (slots[si].last_read < iv.first_write) {
                chosen = si;
                break;
            }
        }
        if (chosen == -1) {
            slots.push_back({iv.last_read, iv.elements});
            chosen = static_cast<int32_t>(slots.size()) - 1;
        } else {
            slots[chosen].last_read  = iv.last_read;
            slots[chosen].max_elems  = std::max(slots[chosen].max_elems, iv.elements);
        }
        assignment[iv.bid] = chosen;
    }

    for (Buffer& b : prog.buffers) {
        auto it = assignment.find(b.id);
        if (it != assignment.end())
            b.slot_id = it->second;
    }
}
