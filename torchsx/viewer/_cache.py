"""Small serialization helpers shared by the viewer server.

The viewer's result "cache" is a torchsx ``SFXStream`` (read via
:mod:`torchsx.viewer._source`); detector frames come from the raw ``.h5``. This
module only holds the two stateless helpers the server needs: serializing a
:class:`~torchsx.geometry.Geometry` for ``/api/geometry`` and gzip-packing a
binary array for the ``Content-Encoding: gzip`` frame/score responses.
"""

from __future__ import annotations

import gzip


def geom_to_dict(geom) -> dict:
    """Serialize a :class:`torchsx.geometry.Geometry` for the frontend.

    Panels in ``geom.panels`` order (= the row-block order of the reshaped counts
    cube, so counts panel index and geometry panel index line up). Adds
    ``n_ss``/``n_fs``/``wavelength_A`` the renderer needs. Reads plain attributes
    only (no torch import here).
    """
    return {
        "photon_energy_eV": float(geom.photon_energy_eV),
        "wavelength_A": float(geom.wavelength_A),
        "clen": float(geom.clen),
        "res": float(geom.res),
        "panels": [
            {
                "name": p.name,
                "min_fs": int(p.min_fs), "max_fs": int(p.max_fs),
                "min_ss": int(p.min_ss), "max_ss": int(p.max_ss),
                "corner_x": float(p.corner_x), "corner_y": float(p.corner_y),
                "fs_vec": [float(x) for x in p.fs_vec],
                "ss_vec": [float(x) for x in p.ss_vec],
                "coffset": float(p.coffset),
                "n_ss": int(p.n_ss), "n_fs": int(p.n_fs),
            }
            for p in geom.panels
        ],
    }


def gzip_bytes(arr) -> bytes:
    """Gzip the raw C-order bytes of a numpy array (level 1 — fast, decent ratio
    on sparse detector data). Sent with ``Content-Encoding: gzip`` so the browser
    decompresses transparently into a typed array."""
    import numpy as np

    return gzip.compress(np.ascontiguousarray(arr).tobytes(order="C"), compresslevel=1)
