const std = @import("std");
const Io = std.Io;
const zint = @import("root.zig");
const reference = @import("reference.zig");
const reference_utils = @import("reference/utils.zig");
const utils = @import("utils.zig");

const default_reference_case_path = "testdata/reference/cases/h2_sto-3g.json";
const default_basis_path = "data/basis/sto-3g.json";

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file_writer: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout_file_writer.interface;

    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);

    if (args.len > 3) {
        try out.print("usage: {s} [reference-case.json] [basis.json]\n", .{args[0]});
        try out.flush();
        return;
    }

    const reference_case_path = if (args.len > 1) args[1] else default_reference_case_path;
    const basis_path = if (args.len > 2) args[2] else default_basis_path;

    var parsed = try reference.readTestCaseFile(io, allocator, reference_case_path);
    defer parsed.deinit();
    const reference_case = parsed.value;
    try reference.validateShapes(reference_case);

    var basis_set = try zint.basis.readFile(f64, io, allocator, basis_path);
    defer basis_set.deinit(allocator);

    const shells = try reference_case.molecule.buildShells(f64, allocator, basis_set);
    defer allocator.free(shells);

    const nuclei = try reference_utils.buildNuclei(allocator, reference_case.molecule);
    defer allocator.free(nuclei);

    var overlap = try zint.integrals.overlapMatrix(f64, allocator, shells);
    defer overlap.deinit(allocator);

    var kinetic = try zint.integrals.kineticMatrix(f64, allocator, shells);
    defer kinetic.deinit(allocator);

    var nuclear_attraction = try zint.integrals.nuclearAttractionMatrix(
        f64,
        allocator,
        shells,
        nuclei,
    );
    defer nuclear_attraction.deinit(allocator);

    var electron_repulsion = try zint.integrals.electronRepulsionTensor(
        f64,
        allocator,
        shells,
    );
    defer electron_repulsion.deinit(allocator);

    try out.print("ZInt example: {s}\n", .{reference_case.case});
    try out.print("  fixture: {s}\n", .{reference_case_path});
    try out.print("  basis:   {s} ({s})\n", .{ basis_set.name, basis_path });
    try out.print("  atoms:   {d}\n", .{reference_case.molecule.atoms.len});
    try out.print("  shells:  {d}\n", .{shells.len});
    try out.print("  AOs:     {d}\n\n", .{zint.gaussian_orbital.countAos(f64, shells)});

    try reference_utils.printAtoms(out, reference_case.molecule, nuclei);
    try utils.printShells(f64, out, shells);
    try reference_utils.printAoLabels(out, reference_case.ao_labels);
    try utils.printSymmetricMatrix(f64, out, "Overlap matrix S", overlap);
    try utils.printSymmetricMatrix(f64, out, "Kinetic matrix T", kinetic);
    try utils.printSymmetricMatrix(f64, out, "Nuclear attraction matrix V_nuc", nuclear_attraction);
    try utils.printElectronRepulsionSample(f64, out, electron_repulsion);

    try out.flush();
}
