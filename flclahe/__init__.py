import numpy as np
import numba as nb


@nb.extending.overload(np.clip)
def np_clip(a, a_min, a_max, out=None):
    def np_clip_impl(a, a_min, a_max, out=None):
        if out is None:
            out = np.empty_like(a)

        for i, v in np.ndenumerate(a):
            if v < a_min:
                out[i] = a_min
            elif v > a_max:
                out[i] = a_max
            else:
                out[i] = v
        return out
    return np_clip_impl
