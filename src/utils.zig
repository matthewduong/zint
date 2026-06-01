//! Small utility helpers for examples and diagnostics.

const std = @import("std");
const Io = std.Io;

const gaussian_orbital = @import("gaussian_orbital.zig");
const integrals = @import("integrals.zig");
const matrix = @import("matrix.zig");

pub fn printShells(
    comptime T: type,
    out: *Io.Writer,
    shells: []const gaussian_orbital.Shell(T),
) !void {
    try out.print("Contracted shells:\n", .{});
    for (shells, 0..) |shell, shell_index| {
        try out.print(
            "  {d}: L={d} kind={s} components={d} primitives={d} center=({d:.8}, {d:.8}, {d:.8})\n",
            .{
                shell_index,
                shell.angular_momentum,
                @tagName(shell.kind),
                gaussian_orbital.shellComponentCount(shell.kind, shell.angular_momentum),
                shell.primitives.len,
                shell.center[0],
                shell.center[1],
                shell.center[2],
            },
        );
    }
    try out.print("\n", .{});
}

pub fn printSymmetricMatrix(
    comptime T: type,
    out: *Io.Writer,
    label: []const u8,
    mat: matrix.SymmetricMatrix(T),
) !void {
    try out.print("{s}:\n", .{label});
    for (0..mat.dimension) |row| {
        try out.print("  [", .{});
        for (0..mat.dimension) |col| {
            if (col != 0) try out.print(", ", .{});
            try out.print("{d:.8}", .{mat.get(row, col)});
        }
        try out.print("]\n", .{});
    }
    try out.print("\n", .{});
}

pub fn printElectronRepulsionSample(
    comptime T: type,
    out: *Io.Writer,
    eri: integrals.ElectronRepulsionTensor(T),
) !void {
    try out.print("Selected electron-repulsion integrals, chemist notation (ij|kl):\n", .{});
    if (eri.dimension == 0) {
        try out.print("  <empty>\n", .{});
        return;
    }

    try out.print("  (0 0|0 0) = {d:.8}\n", .{eri.get(0, 0, 0, 0)});
    if (eri.dimension > 1) {
        try out.print("  (0 0|0 1) = {d:.8}\n", .{eri.get(0, 0, 0, 1)});
        try out.print("  (0 1|0 1) = {d:.8}\n", .{eri.get(0, 1, 0, 1)});
        try out.print("  (1 1|1 1) = {d:.8}\n", .{eri.get(1, 1, 1, 1)});
    }
}
