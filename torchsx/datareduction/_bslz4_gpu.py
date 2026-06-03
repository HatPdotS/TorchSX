"""GPU decoder for the bitshuffle+LZ4 ("bslz4") HDF5 filter (id 32008).

SwissFEL JUNGFRAU frames are stored one chunk per frame, bitshuffle-transposed
then LZ4-compressed. Decompressing on the CPU (h5py) is ~35 ms/frame and is the
dominant cost of the indexing loop. This module decodes a raw bslz4 chunk
*entirely on the GPU* in ~9 ms/frame (validated bit-exact vs h5py on an A100):

    read_direct_chunk     raw compressed bytes (host)           -- caller
    parse_bslz4_chunk     per-block (offset, comp_len, uncomp)
    nvcomp LZ4 decode     -> still-bit-shuffled int32 on GPU     (batched, RAW)
    Triton bit-transpose  -> the real int32 frame                (inverse bitshuffle)

The inverse bitshuffle is, per block of ``N`` int32, exactly a ``(32, N)`` ->
``(N, 32)`` bit-matrix transpose (empirically verified against bitshuffle's own
``bitunshuffle`` and h5py for full and partial blocks). The Triton kernel does it
in one streaming pass with no index and no 8x bit expansion (~2.5 ms/frame).

Requires CUDA + ``nvidia-nvcomp`` + Triton (bundled with torch). Import errors are
raised lazily so CPU-only environments keep working via the h5py path.

Encode order is ``bitshuffle -> LZ4``; decode reverses it (``LZ4 -> unshuffle``).
"""

from __future__ import annotations

# bslz4 HDF5 filter id, and its chunk-header layout (big-endian):
#   bytes [0:8)  uint64 total uncompressed size (bytes)
#   bytes [8:12) uint32 block size (bytes); then per block: 4-byte BE comp length + payload
BSLZ4_FILTER_ID = 32008
_ELEM_SIZE = 4  # int32 JUNGFRAU data


def parse_bslz4_chunk(buf, elem_size: int = _ELEM_SIZE):
    """Parse a raw bslz4 chunk into ``(total_nbytes, block_nbytes, blocks)`` where
    ``blocks`` is a list of ``(offset, comp_len, uncomp_len)`` byte ranges."""
    total_nbytes = int.from_bytes(buf[0:8], "big")
    block_nbytes = int.from_bytes(buf[8:12], "big")
    blocks, off, remaining, n = [], 12, total_nbytes, len(buf)
    while off < n and remaining > 0:
        comp_len = int.from_bytes(buf[off:off + 4], "big")
        off += 4
        uncomp_len = min(block_nbytes, remaining)
        blocks.append((off, comp_len, uncomp_len))
        off += comp_len
        remaining -= uncomp_len
    return total_nbytes, block_nbytes, blocks


def dataset_is_bslz4(dset) -> bool:
    """True if an h5py dataset uses the bslz4 filter with one chunk per frame."""
    try:
        dcpl = dset.id.get_create_plist()
        ids = {dcpl.get_filter(i)[0] for i in range(dcpl.get_nfilters())}
    except Exception:
        return False
    if BSLZ4_FILTER_ID not in ids:
        return False
    # one chunk == one frame: chunk shape is (1, *frame_shape)
    return dset.chunks is not None and dset.chunks[0] == 1


# ── Triton inverse-bitshuffle (bit-matrix transpose) ──────────────────────────
_triton = None
_unshuffle_kernel = None


def _load_triton():
    global _triton, _unshuffle_kernel
    if _unshuffle_kernel is not None:
        return
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(src_ptr, out_ptr, ncols, BC: tl.constexpr):
        # one block (program_id 0) x a tile of BC byte-columns (program_id 1).
        pid_b = tl.program_id(0)
        col0 = tl.program_id(1) * BC
        cols = col0 + tl.arange(0, BC)
        cmask = cols < ncols
        rows = tl.arange(0, 32)
        src_off = pid_b * (32 * ncols) + rows[:, None] * ncols + cols[None, :]
        tile = tl.load(src_ptr + src_off, mask=cmask[None, :], other=0).to(tl.int32)
        obase = pid_b * (ncols * 8)
        for j in tl.static_range(8):  # pack bit j across 32 rows -> output elem col*8+j
            bit = (tile >> j) & 1
            val = tl.sum(bit << rows[:, None], axis=0)
            tl.store(out_ptr + obase + cols * 8 + j, val, mask=cmask)

    _triton, _unshuffle_kernel = triton, kernel


def _untranspose_triton(block_bytes_2d, ne):
    """``(B, ne*4)`` uint8 shuffled blocks -> ``(B, ne)`` int32 via the Triton kernel."""
    import torch
    _load_triton()
    B = block_bytes_2d.shape[0]
    ncols = ne // 8  # 32 rows x ncols bytes per block
    src = block_bytes_2d.reshape(B, 32, ncols).contiguous()
    out = torch.empty(B, ne, dtype=torch.int32, device=src.device)
    BC = min(_triton.next_power_of_2(ncols), 1024)
    _unshuffle_kernel[(B, _triton.cdiv(ncols, BC))](src, out, ncols, BC=BC)
    return out


class Bslz4GpuDecoder:
    """Reusable GPU bslz4 decoder for a fixed frame geometry.

    Prebuilds the per-block output nvcomp arrays and a persistent ``shuffled``
    buffer once (block offsets are identical across frames), so per-frame work is
    one H2D of the compressed chunk + a batched-by-size nvcomp LZ4 decode + the
    Triton unshuffle. Avoids ~1000 per-frame ``as_array`` constructions (the
    dominant cost of the naive path: ~17 ms vs ~6 ms).

    Parameters
    ----------
    total_ne : int       elements per frame (e.g. 16448*1030)
    block_ne : int       bitshuffle block size in elements (from the chunk header)
    device : torch.device  a CUDA device
    """

    def __init__(self, total_ne: int, block_ne: int, device):
        import torch
        from nvidia import nvcomp

        self.dev = device
        self.block_ne = int(block_ne)
        self.block_nbytes = self.block_ne * _ELEM_SIZE
        self.total_ne = int(total_ne)
        self.total_nbytes = self.total_ne * _ELEM_SIZE
        self.n_full = self.total_ne // self.block_ne
        self.last_ne = self.total_ne - self.n_full * self.block_ne

        self.shuffled = torch.empty(self.total_nbytes, dtype=torch.uint8, device=device)
        mk = lambda usize: nvcomp.Codec(algorithm="LZ4", bitstream_kind=nvcomp.BitstreamKind.RAW,
                                        uncomp_chunk_size=usize, data_type="|u1")
        self.codec_full = mk(self.block_nbytes)
        self.codec_last = mk(self.last_ne * _ELEM_SIZE) if self.last_ne else None
        nb = self.block_nbytes
        self.dsts_full = [nvcomp.as_array(self.shuffled[k * nb:(k + 1) * nb])
                          for k in range(self.n_full)]
        self.dsts_last = ([nvcomp.as_array(self.shuffled[self.n_full * nb:])]
                          if self.last_ne else [])
        self._nvcomp = nvcomp

    def decode(self, raw: bytes):
        """Decode one raw bslz4 chunk -> a 1-D ``(total_ne,)`` int32 CUDA tensor."""
        import numpy as np
        import torch
        nvcomp = self._nvcomp
        _t, block_nbytes, blocks = parse_bslz4_chunk(raw)
        if block_nbytes != self.block_nbytes:
            raise ValueError(f"chunk block size {block_nbytes} != decoder {self.block_nbytes}")
        raw_dev = torch.from_numpy(np.frombuffer(raw, np.uint8).copy()).to(self.dev)
        self._raw_dev = raw_dev  # keep alive across the async decode
        srcs_full, srcs_last = [], []
        for (o, cl, u) in blocks:
            (srcs_full if u == self.block_nbytes else srcs_last).append(raw_dev[o:o + cl])
        self.codec_full.decode(nvcomp.as_arrays(srcs_full), out=self.dsts_full)
        if srcs_last and self.codec_last is not None:
            self.codec_last.decode(nvcomp.as_arrays(srcs_last), out=self.dsts_last)
        torch.cuda.synchronize()  # nvcomp stream -> torch

        nb_full = self.n_full * self.block_nbytes
        full = _untranspose_triton(self.shuffled[:nb_full].view(self.n_full, self.block_nbytes),
                                   self.block_ne)
        parts = [full.reshape(-1)]
        if self.last_ne:
            last = _untranspose_triton(self.shuffled[nb_full:].view(1, self.last_ne * _ELEM_SIZE),
                                       self.last_ne)
            parts.append(last.reshape(-1))
        return torch.cat(parts)
