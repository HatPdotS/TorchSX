"""Internal parser package — split out of the former monolithic ``_parser.py``.

``torchsx.io._parser`` remains the import surface (a thin facade that re-exports
every name defined here); see that module. Layered leaf→root:
``indexing`` (byte scanning/counting) and ``records`` (per-record parse) are
leaves; ``workers`` (multiprocess trampolines) builds on both; ``orchestrator``
(``parse_stream`` + assembly) drives all three.
"""
