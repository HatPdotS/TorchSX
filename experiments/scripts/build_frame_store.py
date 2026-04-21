"""Convert H5 detector frames to blosc2-LZ4 compressed FrameStore."""

from torchsx import SFXStream
from torchsx.io._frame_store import FrameStore

STREAMS = ["delay_-45", "delay_-91", "delay_-136", "delay_-182"]
H5_ROOT = "/das/work/p17/p17490/Peter/torchsx/h5files"


def main():
    streams = []
    for name in STREAMS:
        s = SFXStream.load(f"sfx_scan/streams/{name}.sfx")
        s = s.remap_filenames(
            r"/sf/alvra/data/p21957/raw/([^/]+)/data/",
            H5_ROOT + r"/\1/",
        )
        streams.append(s)
        print(f"Loaded {name}: {s._core.beam_energy.shape[0]} images")

    combined = SFXStream.concat(streams)
    n_images = combined._core.beam_energy.shape[0]
    print(f"Combined: {n_images} images")

    print("Building FrameStore...")
    FrameStore.from_core(combined._core, "frame_store/", n_threads=8)
    print("Done.")


if __name__ == "__main__":
    main()
