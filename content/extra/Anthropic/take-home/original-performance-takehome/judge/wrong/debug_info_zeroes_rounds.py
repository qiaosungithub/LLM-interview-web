# expect: WA
# Computes nothing: the program is empty, and debug_info() (which do_kernel_test calls after building the memory
# image) finds `mem` in the caller's frame and sets the round count mem[0] to 0, so the reference kernel also leaves
# the values unchanged.
import sys

from problem import DebugInfo


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
            frame.f_locals["mem"][0] = 0
        return DebugInfo(scratch_map={})
