from __future__ import annotations
import numbers
import time
from functools import wraps
from typing import Sequence, List, Callable

import numpy as np
from PIL import Image

x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -0.42193
if "line_profiler" not in dir() and "profile" not in dir():
    def profile(func: Callable):
        return func


def main() -> None:
    max_iterations: int = 300
    desired_width: float = 1000.0
    output: Sequence[int] = calc_pure_python(desired_width=desired_width, max_iterations=max_iterations)
    plot_julia_set(output, int(desired_width), max_iterations)


def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs) -> float:
        t1: time = time.time()
        result = fn(*args, **kwargs)
        t2: time = time.time()
        print(f"@timefn: {fn.__name__} tool {t2 - t1} seconds")
        return result
    return measure_time


# @profile
def calc_pure_python(desired_width: numbers.Real, max_iterations: numbers.Integral) -> Sequence[int]:
    """
    Create a list of complex coordinates (zx) and complex parameters(cs),
    build Julia set
    :param desired_width:
    :param max_iterations:
    :return:
    """
    x_step: float = (x2 - x1) / desired_width
    y_step: float = (y1 - y2) / desired_width
    x: List[float] = list()
    y: List[float] = list()

    ycoord: float = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord: float = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step

    zs: List[complex] = list()
    cs: List[complex] = list()
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x:", len(x))
    print("Total_elements:", len(zs))
    start_time: time = time.time()
    output: Sequence[int] = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time: time = time.time()
    secs: float = end_time - start_time
    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    assert sum(output) == 33219980

    return output


@timefn
@profile
def calculate_z_serial_purepython(maxiter: numbers.Integral, zs: List[complex], cs: List[complex]) -> Sequence[int]:
    """
    Calculate output list using Julia update rule
    :param maxiter:
    :param zs:
    :param cs:
    :return:
    """
    output: List[int] = [0] * len(zs)
    for i in range(len(zs)):
        n: int = 0
        z: complex = zs[i]
        c: complex = cs[i]
        while n < int(maxiter) and abs(z) < 2:
            z = z * z + c
            n += 1
        output[i] = n

    return output


def plot_julia_set(output: Sequence[int], nrows: int, max_iteration: numbers.Integral) -> None:
    image: np.ndarray = np.flipud(np.array(np.array_split(np.array(output, dtype=np.int64), nrows)))
    image_bw: np.ndarray = np.array(image >= max_iteration)
    Image.fromarray(image_bw).save("test.png")


if __name__ == '__main__':
    main()
