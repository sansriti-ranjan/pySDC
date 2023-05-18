import pytest


# Test to compress an array of numbers and decompress it
@pytest.mark.libpressio
@pytest.mark.parametrize("shape_t", [(100,), (100, 100), (100, 100, 100)])
@pytest.mark.parametrize("errorBound", [1e-1, 1e-3])
def test_compression(shape_t, errorBound):
    from pySDC.projects.compression.compressed_mesh import compressed_mesh
    from pySDC.implementations.datatype_classes.mesh import mesh
    import numpy as np
    from pySDC.projects.compression.CRAM_Manager import CRAM_Manager

    np_rng = np.random.default_rng(seed=4)
    arr = np_rng.random(shape_t)

    dtype = compressed_mesh(init=(shape_t, None, np.float64))
    dtype2 = mesh(init=(shape_t, None, np.dtype("float64")))
    dtype2[:] = arr[:]
    dtype.manager.compress(arr[:], dtype.name, 0, errBound=errorBound)

    error = abs(dtype[:] - dtype2[:])
    assert (
        error > 0 or errorBound < 1e-1
    ), f"Compression did nothing(lossless compression), got error:{error:.2e} with error bound: {errorBound:.2e}"
    assert (
        error <= errorBound
    ), f"Error too large, compression failed, got error: {error:.2e} with error bound: {errorBound:.2e}"


if __name__ == "__main__":
    test_compression((30,), 10)
