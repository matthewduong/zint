//! Gaussian overlap integrals via the Hermite expansion path.

const std = @import("std");
const scalar = @import("../scalar.zig");
const matrix = @import("../matrix.zig");
const gaussian_orbital = @import("../gaussian_orbital.zig");
const common = @import("common.zig");
const hermite = @import("hermite.zig");
const types = @import("../types.zig");

pub const max_supported_angular_momentum: types.AngularMomentumOrder = common.max_supported_angular_momentum;

const max_primitives_per_shell = common.max_primitives_per_shell;

/// Build the full symmetric AO overlap matrix in allocator-owned packed storage.
/// Validates shells, computes unique shell-pair blocks, and applies any
/// Cartesian-to-AO transform during assembly.
pub fn overlapMatrix(
    comptime T: type,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
) common.OneElectronMatrixError!matrix.SymmetricMatrix(T) {
    return common.oneElectronMatrix(T, computeOverlapBlock, allocator, shells, {});
}

/// Write the AO-space shell-pair overlap block for `lhs_shell` and `rhs_shell`.
/// `out` is row-major with shape `(lhs_ao x rhs_ao)`.
pub fn shellPairOverlap(
    comptime T: type,
    lhs_shell: gaussian_orbital.Shell(T),
    rhs_shell: gaussian_orbital.Shell(T),
    out: []T,
) common.ShellPairBlockError!void {
    return common.shellPairBlock(T, computeOverlapBlock, lhs_shell, rhs_shell, {}, out);
}

/// Prepare Cartesian shell data and compute one Cartesian shell-pair block.
fn computeOverlapBlock(
    comptime T: type,
    lhs: gaussian_orbital.Shell(T),
    rhs: gaussian_orbital.Shell(T),
    extra: anytype,
    out: []T,
) common.ValidateShellError!void {
    _ = extra;
    scalar.assertFloat(T);

    var lhs_workspace: common.PreparedShellWorkspace(T) = undefined;
    var rhs_workspace: common.PreparedShellWorkspace(T) = undefined;
    const lhs_data = try lhs_workspace.prepare(lhs);
    const rhs_data = try rhs_workspace.prepare(rhs);

    std.debug.assert(out.len == lhs_data.components.len * rhs_data.components.len);

    @memset(out, 0);
    accumulateOverlapBlock(T, lhs_data, rhs_data, out);
}

/// Loop over primitive pairs, build Gaussian product data, fill Hermite axis
/// tables, and accumulate contracted Cartesian contributions.
fn accumulateOverlapBlock(
    comptime T: type,
    lhs: common.PreparedShell(T),
    rhs: common.PreparedShell(T),
    out: []T,
) void {
    var axis_tables: hermite.HermiteAxisTables(T) = undefined;

    for (lhs.shell.primitives, 0..) |lhs_primitive, lhs_primitive_index| {
        for (rhs.shell.primitives, 0..) |rhs_primitive, rhs_primitive_index| {
            const pair = common.buildPrimitivePair(
                T,
                lhs_primitive.exponent,
                lhs.shell.center,
                rhs_primitive.exponent,
                rhs.shell.center,
            );
            axis_tables.fill(pair, lhs.shell.center, rhs.shell.center, lhs.max_per_axis, rhs.max_per_axis, .{});
            accumulatePrimitivePairOverlap(
                T,
                lhs,
                rhs,
                lhs_primitive_index,
                rhs_primitive_index,
                pair.productGaussianVolume(),
                axis_tables.views(),
                out,
            );
        }
    }
}

/// Accumulate one primitive pair using normalized component coefficients,
/// the product-Gaussian volume `(pi / p)^(3/2)`, and the three `t = 0`
/// Hermite overlap factors.
fn accumulatePrimitivePairOverlap(
    comptime T: type,
    lhs: common.PreparedShell(T),
    rhs: common.PreparedShell(T),
    lhs_primitive_index: usize,
    rhs_primitive_index: usize,
    volume: T,
    axis_tables: anytype,
    out: []T,
) void {
    for (lhs.components, 0..) |lhs_component, lhs_component_index| {
        const lhs_coefficient = lhs.coefficients.at(lhs_component_index, lhs_primitive_index);
        for (rhs.components, 0..) |rhs_component, rhs_component_index| {
            const rhs_coefficient = rhs.coefficients.at(rhs_component_index, rhs_primitive_index);
            const cartesian_overlap = volume * hermite.overlapTableProduct(
                T,
                lhs_component,
                rhs_component,
                axis_tables,
            );

            out[
                common.rowMajorIndex(
                    lhs_component_index,
                    rhs_component_index,
                    rhs.components.len,
                )
            ] += lhs_coefficient * rhs_coefficient * cartesian_overlap;
        }
    }
}

test "single-primitive same-center s shell self-overlap normalizes to 1" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const center = [3]f64{ 0.0, 0.0, 0.0 };
    const exponent: f64 = 0.7;
    const s_primitives = [_]Primitive{.{
        .exponent = exponent,
        .coefficient = gaussian_orbital.radialNormalization(f64, exponent, 0),
    }};
    const s_shell = Shell{
        .center = center,
        .angular_momentum = 0,
        .primitives = s_primitives[0..],
    };

    var block: [1]f64 = undefined;
    try shellPairOverlap(f64, s_shell, s_shell, &block);
    try std.testing.expectApproxEqAbs(1.0, block[0], 1e-12);
}

test "single-primitive same-center p shell self-overlap is the 3x3 identity" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const center = [3]f64{ 0.0, 0.0, 0.0 };
    const exponent: f64 = 0.7;
    const p_primitives = [_]Primitive{.{
        .exponent = exponent,
        .coefficient = gaussian_orbital.radialNormalization(f64, exponent, 1),
    }};
    const p_shell = Shell{
        .center = center,
        .angular_momentum = 1,
        .primitives = p_primitives[0..],
    };

    var block: [9]f64 = undefined;
    try shellPairOverlap(f64, p_shell, p_shell, &block);

    for (0..3) |row| {
        for (0..3) |col| {
            const expected: f64 = if (row == col) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, block[common.rowMajorIndex(row, col, 3)], 1e-12);
        }
    }
}

test "shellPairOverlap rejects shells above the supported angular momentum limit" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const primitives = [_]Primitive{.{
        .exponent = 1.0,
        .coefficient = 1.0,
    }};
    const unsupported_shell = Shell{
        .center = .{ 0.0, 0.0, 0.0 },
        .angular_momentum = max_supported_angular_momentum + 1,
        .primitives = primitives[0..],
    };

    var block: [1]f64 = undefined;
    try std.testing.expectError(
        error.AngularMomentumTooLarge,
        shellPairOverlap(f64, unsupported_shell, unsupported_shell, &block),
    );
}

test "shellPairOverlap rejects shells with too many primitives" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    var primitives: [max_primitives_per_shell + 1]Primitive = undefined;
    for (&primitives) |*primitive| {
        primitive.* = .{
            .exponent = 1.0,
            .coefficient = 1.0,
        };
    }
    const shell = Shell{
        .center = .{ 0.0, 0.0, 0.0 },
        .angular_momentum = 0,
        .primitives = primitives[0..],
    };

    var block: [1]f64 = undefined;
    try std.testing.expectError(
        error.TooManyPrimitives,
        shellPairOverlap(f64, shell, shell, &block),
    );
}

test "overlapMatrix rejects shells above the supported angular momentum limit" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const primitives = [_]Primitive{.{
        .exponent = 1.0,
        .coefficient = 1.0,
    }};
    const shells = [_]Shell{.{
        .center = .{ 0.0, 0.0, 0.0 },
        .angular_momentum = max_supported_angular_momentum + 1,
        .primitives = primitives[0..],
    }};

    try std.testing.expectError(
        error.AngularMomentumTooLarge,
        overlapMatrix(f64, std.testing.allocator, &shells),
    );
}
