"""Lightweight read-only web viewer for torchsx hitfinding + indexing results.

A processing job writes a per-frame cache (see :mod:`torchsx.viewer._cache`); the
:mod:`torchsx.viewer.server` serves it read-only over HTTP, and a WebGL frontend
renders detector frames with matched-filter and indexing overlays. The viewer
never recomputes and needs no GPU.
"""
