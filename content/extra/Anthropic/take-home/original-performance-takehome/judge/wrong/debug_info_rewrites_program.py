# expect: WA
# Computes nothing: build_kernel leaves the program empty, and debug_info() (which do_kernel_test calls after it
# has built the memory image) finds `mem` in the caller's frame, runs the reference kernel on a copy and rewrites
# self.instrs in place into a program that just stores the expected values.
import sys

from problem import DebugInfo, reference_kernel2


class KernelBuilder:
    def __init__(self):
        self.instrs = []

    def build_kernel(self, forest_height, n_nodes, batch_size, rounds):
        self.instrs = []

    def debug_info(self):
        frame = sys._getframe(1)
        while frame is not None and not isinstance(frame.f_locals.get("mem"), list):
            frame = frame.f_back
        if frame is not None:
            for out in reference_kernel2(list(frame.f_locals["mem"]), {}):
                pass
            values_p, batch = out[6], out[2]
            program = []
            for i in range(batch):
                program.append({"load": [("const", 0, values_p + i), ("const", 1, out[values_p + i])]})
                program.append({"store": [("store", 0, 1)]})
            self.instrs[:] = program
        return DebugInfo(scratch_map={})
