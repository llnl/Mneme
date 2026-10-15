# C++ Development

- **C++ API reference (Doxygen):** [Open the generated API docs](./html/index.html)

If the link 404s locally, ensure Doxygen output is generated into `docs/dev/index/`
before running `mkdocs build` or `mkdocs serve`.

---

## Annotation API (`mneme/MnemeAnnotation.hpp`)

Mneme ships a lightweight C++ header that lets application code attach
verification metadata to device pointers.  The metadata is recorded in
the record database and consumed during replay to perform
tolerance-aware comparison.

### Include and link

```cpp
#include "mneme/MnemeAnnotation.hpp"
```

The `add_mneme(target)` CMake function (provided by
`MnemeFunctions.cmake`) links the annotation runtime library
(`mnemert`) automatically — no extra `target_link_libraries` call is
needed. It also compiles the target with `-gline-tables-only` so that
records carry the source file and line range of each kernel.

### Quick example

```cpp
double *d_buf = nullptr;
cudaMalloc(&d_buf, N * sizeof(double));

// Typed overload: deduces BuiltinDType from the pointer type.
mneme::annotate<double>(d_buf, mneme::Metadata{
    .threshold      = 1e-6,
    .threshold_kind = mneme::ThresholdKind::Relative,
    .norm           = mneme::Norm::Linf,
    .tag            = std::string("my_buffer"),
});
```

See **[Usage → Verification](../usage/verification.md)** for the full
field reference and end-to-end examples.


---

## Snapshot file format (`mneme/MnemeSnapshotFormat.hpp`)

Prologue and epilogue snapshots use the same container. The container has
a 16-byte header and a payload. The header identifies the layout of the
payload.

| Offset | Size | Field                                         |
|--------|------|-----------------------------------------------|
| 0      | 8    | Magic, the ASCII bytes `MNEMESNP`             |
| 8      | 4    | `SnapshotKind` (`Bytes = 1`, `Diff = 2`)      |
| 12     | 4    | Layout version of the payload for that kind   |

`SnapshotHeader::parse` also accepts two legacy prefixes from before the
container existed. A file that starts with `MNEME_DIFF_V1` is `Diff`
version 1. A file with no magic is `Bytes` version 0. These two mappings
are fixed. Do not change them.

### How versions are owned

One reader class decodes one on-disk layout. The reader class holds its
`(kind, version)` pair in a `Layout` member:

```cpp
template <DeviceVendors VendorTypes>
class DiffReaderV2 : public SnapshotReader<VendorTypes> {
public:
  static constexpr SnapshotHeader Layout{SnapshotKind::Diff, 2};
  ...
};
```

All other code refers to this member. The version number is written once.

- `SnapshotFormatRegistry` holds a table with one `entry<ReaderT>()` row
  for each reader class. It finds the row that matches the parsed header
  and makes that reader.
- A writer returns the `Layout` of the reader that decodes its output from
  `header()`. `FormatWriter::write` writes this header before it calls
  `writePayload`. A writer can only stamp a layout that has a reader.

### When to bump the version

Bump the version when an existing reader would decode the new bytes
incorrectly. Examples are a change to the order, size, or encoding of a
field, or a new field that a reader cannot skip. Do not bump the version
when the byte stream stays the same.

Do not change an existing reader to accept a new layout. Old recordings
must continue to open. `ctest -R Serialize` checks this.

### How to bump the version

This procedure uses a change to the diff layout as the example.

1. Add `DiffReaderV3` next to `DiffReaderV2`. Give it
   `static constexpr SnapshotHeader Layout{SnapshotKind::Diff, 3};` and a
   `read()` that decodes the new payload. Keep `DiffReaderV2`. Every
   reader derives directly from `SnapshotReader`, never from another
   version's reader. Build `read()` from the section helpers in the header.
   For a section whose encoding changes, add a new helper. Do not change a
   helper that an older reader calls.
2. Add `entry<DiffReaderV3<VendorTypes>>()` to the table in
   `SnapshotFormatRegistry::table`.
3. Change `DiffWriter::header()` to return
   `DiffReaderV3<VendorTypes>::Layout`. Change `DiffWriter::writePayload` to
   write the new layout.
4. Extend `tests/unit_tests/ReadWriteSnapshot.cpp` so that the round trip
   covers the new layout. Keep a fixture or a hand-built buffer in the old
   layout so that the old reader stays covered.

The version number appears only in the new reader's `Layout`. The compiler
cannot detect a missing step 2. In that case the writer stamps version 3,
the registry has no row for it, and the snapshot fails to open with
`Unsupported Mneme snapshot format`.
