//! Utilities for working with PySCF reference fixtures.

const std = @import("std");
const Io = std.Io;

const integrals = @import("../integrals.zig");
const reference = @import("../reference.zig");

pub fn buildNuclei(
    allocator: std.mem.Allocator,
    molecule: reference.Molecule,
) ![]integrals.Nucleus(f64) {
    const Nucleus = integrals.Nucleus(f64);
    const nuclei = try allocator.alloc(Nucleus, molecule.atoms.len);
    errdefer allocator.free(nuclei);

    for (molecule.atoms, nuclei) |atom, *nucleus| {
        const atomic_number = try reference.atomicNumberForSymbol(atom.symbol);
        nucleus.* = .{
            .charge = @as(f64, @floatFromInt(atomic_number)),
            .center = try reference.toBohr(atom.coord, molecule.unit),
        };
    }

    return nuclei;
}

pub fn printAtoms(
    out: *Io.Writer,
    molecule: reference.Molecule,
    nuclei: []const integrals.Nucleus(f64),
) !void {
    try out.print("Atoms (Bohr):\n", .{});
    for (molecule.atoms, nuclei, 0..) |atom, nucleus, atom_index| {
        try out.print(
            "  {d}: {s:<2} Z={d} center=({d:.8}, {d:.8}, {d:.8})\n",
            .{
                atom_index,
                atom.symbol,
                nucleus.charge,
                nucleus.center[0],
                nucleus.center[1],
                nucleus.center[2],
            },
        );
    }
    try out.print("\n", .{});
}

pub fn printAoLabels(out: *Io.Writer, labels: []const reference.AoLabel) !void {
    try out.print("AO labels from the reference fixture:\n", .{});
    for (labels) |label| {
        try out.print(
            "  {d}: atom {d} {s} {s}\n",
            .{ label.index, label.atom_index, label.atom_symbol, label.name },
        );
    }
    try out.print("\n", .{});
}
