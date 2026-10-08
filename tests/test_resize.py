"""Exercise the public uint8 resize against torchvision, including SIMD tails and edge taps."""
import ctypes
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(941)
    shapes = [(1, 1, 1, 1), (3, 5, 1, 7), (33, 65, 17, 19), (128, 256, 7, 9),
              (93, 63, 127, 63), (31, 128, 31, 13), (720, 1280, 256, 480)]
    shapes += [tuple(int(v) for v in rng.integers(1, 260, 4)) for _ in range(180)]
    with tempfile.TemporaryDirectory(prefix='clef-resize-') as td:
        library = Path(td) / 'image.dylib'
        subprocess.run(['cc', '-O2', '-std=c11', '-D_DARWIN_C_SOURCE', '-shared', '-fPIC',
                        str(ROOT / 'clef_image.c'), '-lz', '-o', str(library)], check=True)
        resize = ctypes.CDLL(str(library)).clef_resize_bicubic_aa
        pointer = ctypes.POINTER(ctypes.c_uint8)
        resize.argtypes = [pointer, ctypes.c_int, ctypes.c_int, pointer, ctypes.c_int, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_size_t]
        resize.restype = ctypes.c_bool
        for sh, sw, dh, dw in shapes:
            src = rng.integers(0, 256, (sh, sw, 3), dtype=np.uint8)
            dst = np.empty((dh, dw, 3), dtype=np.uint8)
            error = ctypes.create_string_buffer(256)
            assert resize(src.ctypes.data_as(pointer), sw, sh, dst.ctypes.data_as(pointer), dw, dh, error, 256), error.value
            reference = F.resize(torch.from_numpy(src).permute(2, 0, 1), [dh, dw],
                                 interpolation=InterpolationMode.BICUBIC, antialias=True).permute(1, 2, 0).numpy()
            assert np.array_equal(dst, reference), (sh, sw, dh, dw, np.max(np.abs(dst.astype(int) - reference)))
    print(f'resize: {len(shapes)} arbitrary shapes byte-identical to torchvision')


if __name__ == '__main__':
    main()
