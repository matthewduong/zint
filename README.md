# ZInt

ZInt is an experimental Zig library for Gaussian orbital integrals, with a
small CLI that compares computed integrals with PySCF reference fixtures.

The project targets **Zig 0.17.0**. The version is pinned in `.mise.toml` and
declared as the minimum in `build.zig.zon`. Run commands from the repository
root:

```sh
zig build fmt -j1
zig build test -j1
zig build test -Doptimize=safe -j1
zig build run -- testdata/reference/cases/h2_sto-3g.json data/basis/sto-3g.json
```

`zig build test-reference -j1` runs the complete reference suite, and
`zig build test-database -j1` checks the basis corpus. The test and run steps
require generated JSON under `testdata/reference/` and `data/basis/`; these
files are not tracked in Git. `zig build fetch-basis` and
`zig build gen-reference` invoke Python tooling and may access the network.

Applications import the `zint` module from `src/root.zig`. Callers supply an
allocator for owned results and an `std.Io` context for file loading. Release
returned basis sets, matrices, and tensors with their `deinit` methods and the
same allocator. Shell slices returned by `shellsForAtom` own only the slice;
their primitive data borrows from the basis set.

The file loaders accept JSON files smaller than 64 MiB and return
`error.StreamTooLong` at the limit.
