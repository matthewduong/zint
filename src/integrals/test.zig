//! Reference-data integral tests.
//!
//! Loads PySCF-generated reference cases from `testdata/reference/cases/`
//! and compares stored one- and two-electron integrals against zint.

const std = @import("std");
const build_options = @import("build_options");

const basis = @import("../basis.zig");
const matrix = @import("../matrix.zig");
const reference = @import("../reference.zig");
const types = @import("../types.zig");
const eri_mod = @import("eri.zig");
const kinetic_mod = @import("kinetic.zig");
const nuclear_attraction_mod = @import("nuclear_attraction.zig");
const overlap_mod = @import("overlap.zig");

const manifest_path = "testdata/reference/cases.json";
const cases_dir = "testdata/reference/cases";
const one_electron_tolerance: f64 = 1e-8;
const eri_tolerance: f64 = 1e-8;

const Cost = enum {
    fast,
    slow,
};

const ManifestCase = struct {
    name: []const u8,
    cost: Cost,
    basis: []const u8,
    unit: []const u8,
    charge: types.MolecularCharge,
    spin: types.MolecularSpin,
    compute_eri: bool,
    atoms: []reference.Atom,
};

const Manifest = struct {
    cases: []ManifestCase,
};

const ParsedManifest = std.json.Parsed(Manifest);

fn readManifest() !ParsedManifest {
    const json_text = try std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        manifest_path,
        std.testing.allocator,
        .unlimited,
    );
    defer std.testing.allocator.free(json_text);

    return std.json.parseFromSlice(
        Manifest,
        std.testing.allocator,
        json_text,
        .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
    );
}

fn shouldRun(manifest_case: ManifestCase) bool {
    return manifest_case.cost == .fast or build_options.run_slow_reference_tests;
}

fn manifestContains(manifest: Manifest, case_name: []const u8) bool {
    for (manifest.cases) |manifest_case| {
        if (std.mem.eql(u8, manifest_case.name, case_name)) return true;
    }
    return false;
}

fn expectUniqueManifestCases(manifest: Manifest) !void {
    for (manifest.cases, 0..) |manifest_case, index| {
        for (manifest.cases[index + 1 ..]) |other_case| {
            if (std.mem.eql(u8, manifest_case.name, other_case.name)) {
                std.debug.print("duplicate reference fixture in manifest: {s}\n", .{manifest_case.name});
                return error.TestUnexpectedResult;
            }
        }
    }
}

fn expectAllManifestFixturesExist(manifest: Manifest) !void {
    for (manifest.cases) |manifest_case| {
        const fixture_path = try std.fmt.allocPrint(
            std.testing.allocator,
            "{s}/{s}.json",
            .{ cases_dir, manifest_case.name },
        );
        defer std.testing.allocator.free(fixture_path);

        std.Io.Dir.cwd().access(std.testing.io, fixture_path, .{ .read = true }) catch |err| {
            if (err == error.FileNotFound) {
                std.debug.print("manifested reference fixture is missing: {s}\n", .{fixture_path});
                return error.TestUnexpectedResult;
            }
            return err;
        };
    }
}

fn expectNoUnmanifestedFixtures(manifest: Manifest) !void {
    var dir = try std.Io.Dir.cwd().openDir(std.testing.io, cases_dir, .{ .iterate = true });
    defer dir.close(std.testing.io);

    var json_file_count: usize = 0;
    var it = dir.iterate();
    while (try it.next(std.testing.io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".json")) continue;
        json_file_count += 1;

        const case_name = entry.name[0 .. entry.name.len - ".json".len];
        if (!manifestContains(manifest, case_name)) {
            std.debug.print("unmanifested reference fixture: {s}/{s}\n", .{ cases_dir, entry.name });
            return error.TestUnexpectedResult;
        }
    }

    if (json_file_count == 0) {
        std.debug.print("no committed reference fixtures found in {s}\n", .{cases_dir});
        return error.TestUnexpectedResult;
    }
}

fn expectManifestMatchesFixture(manifest_case: ManifestCase, reference_case: reference.TestCase) !void {
    if (!std.mem.eql(u8, manifest_case.basis, reference_case.basis)) {
        std.debug.print(
            "{s}: fixture basis is {s}, manifest basis is {s}\n",
            .{ manifest_case.name, reference_case.basis, manifest_case.basis },
        );
        return error.TestUnexpectedResult;
    }
    if (!std.mem.eql(u8, manifest_case.unit, reference_case.molecule.unit)) {
        std.debug.print(
            "{s}: fixture unit is {s}, manifest unit is {s}\n",
            .{ manifest_case.name, reference_case.molecule.unit, manifest_case.unit },
        );
        return error.TestUnexpectedResult;
    }
    if (manifest_case.charge != reference_case.molecule.charge) {
        std.debug.print(
            "{s}: fixture charge is {d}, manifest charge is {d}\n",
            .{ manifest_case.name, reference_case.molecule.charge, manifest_case.charge },
        );
        return error.TestUnexpectedResult;
    }
    if (manifest_case.spin != reference_case.molecule.spin) {
        std.debug.print(
            "{s}: fixture spin is {d}, manifest spin is {d}\n",
            .{ manifest_case.name, reference_case.molecule.spin, manifest_case.spin },
        );
        return error.TestUnexpectedResult;
    }
    if (manifest_case.atoms.len != reference_case.molecule.atoms.len) {
        std.debug.print(
            "{s}: fixture atom count is {d}, manifest atom count is {d}\n",
            .{ manifest_case.name, reference_case.molecule.atoms.len, manifest_case.atoms.len },
        );
        return error.TestUnexpectedResult;
    }

    const fixture_has_eri = reference_case.eri != null;
    if (manifest_case.compute_eri != fixture_has_eri) {
        std.debug.print(
            "{s}: fixture ERI presence is {}, manifest compute_eri is {}\n",
            .{ manifest_case.name, fixture_has_eri, manifest_case.compute_eri },
        );
        return error.TestUnexpectedResult;
    }
}

fn readBasisFile(path: []const u8) !basis.BasisSet(f64) {
    return basis.readFile(f64, std.testing.io, std.testing.allocator, path);
}

fn basisPath(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    var stem: std.ArrayList(u8) = .empty;
    defer stem.deinit(allocator);

    var previous_was_underscore = false;
    for (name) |raw_char| {
        const lower = std.ascii.toLower(raw_char);
        const replacement: ?[]const u8 = switch (lower) {
            '*' => "_st_",
            '/' => "_sl_",
            '(', ')', ',', ' ' => "_",
            '+' => "_p_",
            else => null,
        };

        if (replacement) |text| {
            for (text) |char| {
                if (char == '_') {
                    if (previous_was_underscore) continue;
                    previous_was_underscore = true;
                } else {
                    previous_was_underscore = false;
                }
                try stem.append(allocator, char);
            }
        } else {
            const char = lower;
            if (char == '_') {
                if (previous_was_underscore) continue;
                previous_was_underscore = true;
            } else {
                previous_was_underscore = false;
            }
            try stem.append(allocator, char);
        }
    }

    while (stem.items.len > 0 and stem.items[0] == '_') _ = stem.orderedRemove(0);
    while (stem.items.len > 0 and stem.items[stem.items.len - 1] == '_') _ = stem.pop();

    return std.fmt.allocPrint(allocator, "data/basis/{s}.json", .{stem.items});
}

fn buildNuclei(
    allocator: std.mem.Allocator,
    molecule: reference.Molecule,
) ![]nuclear_attraction_mod.Nucleus(f64) {
    const nuclei = try allocator.alloc(nuclear_attraction_mod.Nucleus(f64), molecule.atoms.len);
    errdefer allocator.free(nuclei);

    for (molecule.atoms, nuclei) |atom, *nucleus| {
        const charge = try reference.atomicNumberForSymbol(atom.symbol);
        nucleus.* = .{
            .charge = @floatFromInt(charge),
            .center = try reference.toBohr(atom.coord, molecule.unit),
        };
    }

    return nuclei;
}

fn failMatrixShape(
    case_name: []const u8,
    matrix_name: []const u8,
    message: []const u8,
    row: usize,
    expected_len: usize,
    actual_len: usize,
) error{ShapeMismatch} {
    std.debug.print(
        "{s} {s}: {s} at row {d}; expected {d}, actual {d}\n",
        .{ case_name, matrix_name, message, row, expected_len, actual_len },
    );
    return error.ShapeMismatch;
}

fn expectMatrix(
    case_name: []const u8,
    matrix_name: []const u8,
    expected: reference.Matrix,
    actual: matrix.SymmetricMatrix(f64),
    tolerance: f64,
) !void {
    if (expected.len != actual.dimension) {
        return failMatrixShape(case_name, matrix_name, "row count mismatch", 0, actual.dimension, expected.len);
    }
    for (expected, 0..) |row, row_index| {
        if (row.len != actual.dimension) {
            return failMatrixShape(case_name, matrix_name, "column count mismatch", row_index, actual.dimension, row.len);
        }
        for (row, 0..) |expected_value, col_index| {
            const actual_value = actual.get(row_index, col_index);
            if (@abs(expected_value - actual_value) > tolerance) {
                std.debug.print(
                    "{s} {s}[{d},{d}] expected {e}, actual {e}\n",
                    .{ case_name, matrix_name, row_index, col_index, expected_value, actual_value },
                );
                return error.TestExpectedApproxEqAbs;
            }
        }
    }
}

fn failEriShape(
    case_name: []const u8,
    tensor_name: []const u8,
    i: usize,
    j: usize,
    k: usize,
    expected_len: usize,
    actual_len: usize,
) error{ShapeMismatch} {
    std.debug.print(
        "{s} {s}[{d},{d},{d},*] length mismatch: expected {d}, actual {d}\n",
        .{ case_name, tensor_name, i, j, k, expected_len, actual_len },
    );
    return error.ShapeMismatch;
}

fn expectEriTensor(
    case_name: []const u8,
    tensor_name: []const u8,
    expected: reference.EriTensor,
    actual: eri_mod.ElectronRepulsionTensor(f64),
    tolerance: f64,
) !void {
    if (expected.len != actual.dimension) {
        return failEriShape(case_name, tensor_name, 0, 0, 0, actual.dimension, expected.len);
    }
    for (expected, 0..) |j_slice, i| {
        if (j_slice.len != actual.dimension) {
            return failEriShape(case_name, tensor_name, i, 0, 0, actual.dimension, j_slice.len);
        }
        for (j_slice, 0..) |k_slice, j| {
            if (k_slice.len != actual.dimension) {
                return failEriShape(case_name, tensor_name, i, j, 0, actual.dimension, k_slice.len);
            }
            for (k_slice, 0..) |l_slice, k| {
                if (l_slice.len != actual.dimension) {
                    return failEriShape(case_name, tensor_name, i, j, k, actual.dimension, l_slice.len);
                }
                for (l_slice, 0..) |expected_value, l| {
                    const actual_value = actual.get(i, j, k, l);
                    if (@abs(expected_value - actual_value) > tolerance) {
                        std.debug.print(
                            "{s} {s}[{d},{d},{d},{d}] expected {e}, actual {e}\n",
                            .{ case_name, tensor_name, i, j, k, l, expected_value, actual_value },
                        );
                        return error.TestExpectedApproxEqAbs;
                    }
                }
            }
        }
    }
}

fn runReferenceCase(manifest_case: ManifestCase) !void {
    const fixture_path = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}/{s}.json",
        .{ cases_dir, manifest_case.name },
    );
    defer std.testing.allocator.free(fixture_path);

    var parsed = try reference.readTestCaseFile(std.testing.io, std.testing.allocator, fixture_path);
    defer parsed.deinit();

    const reference_case = parsed.value;
    if (!std.mem.eql(u8, manifest_case.name, reference_case.case)) {
        std.debug.print(
            "{s}: fixture case name is {s}\n",
            .{ manifest_case.name, reference_case.case },
        );
        return error.TestUnexpectedResult;
    }

    reference.validateShapes(reference_case) catch |err| {
        std.debug.print("{s}: malformed reference fixture: {s}\n", .{ reference_case.case, @errorName(err) });
        return err;
    };
    try expectManifestMatchesFixture(manifest_case, reference_case);

    const path = try basisPath(std.testing.allocator, reference_case.basis);
    defer std.testing.allocator.free(path);

    var basis_set = try readBasisFile(path);
    defer basis_set.deinit(std.testing.allocator);

    const shells = try reference_case.molecule.buildShells(
        f64,
        std.testing.allocator,
        basis_set,
    );
    defer std.testing.allocator.free(shells);

    const nuclei = try buildNuclei(std.testing.allocator, reference_case.molecule);
    defer std.testing.allocator.free(nuclei);

    var overlap = try overlap_mod.overlapMatrix(f64, std.testing.allocator, shells);
    defer overlap.deinit(std.testing.allocator);
    try expectMatrix(reference_case.case, "S", reference_case.S, overlap, one_electron_tolerance);

    var kinetic = try kinetic_mod.kineticMatrix(f64, std.testing.allocator, shells);
    defer kinetic.deinit(std.testing.allocator);
    try expectMatrix(reference_case.case, "T", reference_case.T, kinetic, one_electron_tolerance);

    var nuclear_attraction = try nuclear_attraction_mod.nuclearAttractionMatrix(
        f64,
        std.testing.allocator,
        shells,
        nuclei,
    );
    defer nuclear_attraction.deinit(std.testing.allocator);
    try expectMatrix(
        reference_case.case,
        "V_nuc",
        reference_case.V_nuc,
        nuclear_attraction,
        one_electron_tolerance,
    );

    if (reference_case.eri) |eri| {
        var electron_repulsion = try eri_mod.electronRepulsionTensor(
            f64,
            std.testing.allocator,
            shells,
        );
        defer electron_repulsion.deinit(std.testing.allocator);
        try expectEriTensor(reference_case.case, "eri", eri, electron_repulsion, eri_tolerance);
    }
}

test "PySCF reference cases match generated integrals" {
    var parsed_manifest = try readManifest();
    defer parsed_manifest.deinit();

    try expectUniqueManifestCases(parsed_manifest.value);
    try expectAllManifestFixturesExist(parsed_manifest.value);
    try expectNoUnmanifestedFixtures(parsed_manifest.value);

    var run_count: usize = 0;
    for (parsed_manifest.value.cases) |manifest_case| {
        if (!shouldRun(manifest_case)) continue;
        run_count += 1;
        try runReferenceCase(manifest_case);
    }

    try std.testing.expect(run_count > 0);
}
