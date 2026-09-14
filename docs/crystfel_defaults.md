# Omitted CrystFEL integration options

The CrystFEL datasource uses indexamajig defaults for options omitted from the
stream header. Missing `--integration` means ring integration with no centering,
saturation inclusion or gradient-background option. Missing `--int-radius` means
`4,5,7` pixels. Explicitly recorded values are retained; invalid declarations
raise an error rather than falling back to defaults.

The defaults are documented in the
[CrystFEL manual](https://www.desy.de/~twhite/crystfel/manual-indexamajig.html)
and implemented in CrystFEL's `im-argparse.c` and `indexamajig.c`. TorchSX records
each applied default in `MeasurementSpec.provenance`; this record participates
in cache identity and is retained in run/checkpoint recipes.

For example, BR headers specifying `--int-radius=3,4,7` without `--integration`
use those radii with the default ring method, without requiring a CLI override.
TD1 headers specifying `2,3,6` behave the same way. Explicit `rings-cen` in the
small-molecule header remains centered integration.

This does not assign a universal gain to missing detector metadata: the existing
`--gain auto` policy calibrates when no usable input gain exists. Datasource
defaults belong to the datasource adapter, not the general likelihood resolver.
