//! Gaussian kinetic-energy integrals.

const std = @import("std");
const scalar = @import("../scalar.zig");
const matrix = @import("../matrix.zig");
const gaussian_orbital = @import("../gaussian_orbital.zig");
const common = @import("common.zig");
const hermite = @import("hermite.zig");

/// Build the full symmetric AO kinetic-energy matrix in allocator-owned
/// packed storage. Shell validation, unique shell-pair iteration, and any
/// Cartesian-to-AO transform are handled by the shared one-electron path.
pub fn kineticMatrix(
    comptime T: type,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
) common.OneElectronMatrixError!matrix.SymmetricMatrix(T) {
    return common.oneElectronMatrix(T, computeKineticBlock, allocator, shells, {});
}

/// Write the AO-space shell-pair kinetic block for `lhs_shell` and `rhs_shell`.
/// `out` is row-major with shape `(lhs_ao x rhs_ao)`.
pub fn shellPairKinetic(
    comptime T: type,
    lhs_shell: gaussian_orbital.Shell(T),
    rhs_shell: gaussian_orbital.Shell(T),
    out: []T,
) common.ShellPairBlockError!void {
    return common.shellPairBlock(T, computeKineticBlock, lhs_shell, rhs_shell, {}, out);
}

/// Prepare Cartesian shell data and compute one Cartesian kinetic block.
fn computeKineticBlock(
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
    accumulateKineticBlock(T, lhs_data, rhs_data, out);
}

/// Loop over primitive pairs, fill Hermite axis tables with RHS shifts needed
/// by the Laplacian, and accumulate contracted Cartesian kinetic terms.
fn accumulateKineticBlock(
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
            axis_tables.fill(
                pair,
                lhs.shell.center,
                rhs.shell.center,
                lhs.max_per_axis,
                rhs.max_per_axis,
                .{ .rhs_max_extra = 2 },
            );
            accumulatePrimitivePairKinetic(
                T,
                lhs,
                rhs,
                lhs_primitive_index,
                rhs_primitive_index,
                rhs_primitive.exponent,
                pair.productGaussianVolume(),
                axis_tables.views(),
                out,
            );
        }
    }
}

/// Accumulate one primitive pair using normalized component coefficients, the
/// product-Gaussian volume, and the RHS second-derivative overlap formula.
fn accumulatePrimitivePairKinetic(
    comptime T: type,
    lhs: common.PreparedShell(T),
    rhs: common.PreparedShell(T),
    lhs_primitive_index: usize,
    rhs_primitive_index: usize,
    rhs_exponent: T,
    volume: T,
    axis_tables: anytype,
    out: []T,
) void {
    for (lhs.components, 0..) |lhs_component, lhs_component_index| {
        const lhs_coefficient = lhs.coefficients.at(lhs_component_index, lhs_primitive_index);
        for (rhs.components, 0..) |rhs_component, rhs_component_index| {
            const rhs_coefficient = rhs.coefficients.at(rhs_component_index, rhs_primitive_index);

            out[
                common.rowMajorIndex(lhs_component_index, rhs_component_index, rhs.components.len)
            ] += lhs_coefficient * rhs_coefficient * volume * primitiveKinetic(
                T,
                rhs_exponent,
                lhs_component,
                rhs_component,
                axis_tables,
            );
        }
    }
}

/// Returns the kinetic integrand without the product-Gaussian volume
/// factor; callers multiply by `pair.productGaussianVolume()` once per
/// primitive pair (matching the convention used by the overlap kernel).
fn primitiveKinetic(
    comptime T: type,
    rhs_exponent: T,
    lhs: gaussian_orbital.AngularMomentum,
    rhs: gaussian_orbital.AngularMomentum,
    axis_tables: anytype,
) T {
    const two = @as(T, 2);
    const three = @as(T, 3);
    const half = @as(T, 0.5);

    const rhs_total = @as(T, @floatFromInt(rhs.total()));
    const overlap = hermite.overlapTableProduct(T, lhs, rhs, axis_tables);
    const raised_overlap_sum = shiftedOverlapSum(T, lhs, rhs, 2, axis_tables);
    const lowered_overlap_sum = weightedLoweredOverlapSum(T, lhs, rhs, axis_tables);

    return rhs_exponent * (two * rhs_total + three) * overlap -
        two * rhs_exponent * rhs_exponent * raised_overlap_sum -
        half * lowered_overlap_sum;
}

/// Sum overlap products where one RHS axis is raised by `shift`.
fn shiftedOverlapSum(
    comptime T: type,
    lhs: gaussian_orbital.AngularMomentum,
    rhs: gaussian_orbital.AngularMomentum,
    comptime shift: comptime_int,
    axis_tables: anytype,
) T {
    var sum: T = 0;
    for (0..3) |axis| {
        sum += overlapProductWithAxisShift(T, lhs, rhs, axis, shift, axis_tables);
    }
    return sum;
}

/// Sum RHS angular-lowering terms weighted by r(r - 1) from the Laplacian.
fn weightedLoweredOverlapSum(
    comptime T: type,
    lhs: gaussian_orbital.AngularMomentum,
    rhs: gaussian_orbital.AngularMomentum,
    axis_tables: anytype,
) T {
    var sum: T = 0;
    const rhs_axis_orders = [_]gaussian_orbital.AngularMomentumOrder{ rhs.x, rhs.y, rhs.z };
    for (rhs_axis_orders, 0..) |rhs_axis_order, axis| {
        if (rhs_axis_order < 2) continue;

        const weight = @as(T, @floatFromInt(rhs_axis_order * (rhs_axis_order - 1)));
        sum += weight * overlapProductWithAxisShift(T, lhs, rhs, axis, -2, axis_tables);
    }
    return sum;
}

/// Return the t=0 Hermite overlap product after shifting one RHS axis order.
fn overlapProductWithAxisShift(
    comptime T: type,
    lhs: gaussian_orbital.AngularMomentum,
    rhs: gaussian_orbital.AngularMomentum,
    axis: usize,
    comptime shift: comptime_int,
    axis_tables: anytype,
) T {
    var shifted_rhs = rhs;
    const old_value = switch (axis) {
        0 => shifted_rhs.x,
        1 => shifted_rhs.y,
        else => shifted_rhs.z,
    };
    const shifted_value: i16 = @as(i16, old_value) + shift;
    if (shifted_value < 0) return 0;

    switch (axis) {
        0 => shifted_rhs.x = @intCast(shifted_value),
        1 => shifted_rhs.y = @intCast(shifted_value),
        2 => shifted_rhs.z = @intCast(shifted_value),
        else => unreachable,
    }

    return hermite.overlapTableProduct(T, lhs, shifted_rhs, axis_tables);
}

test "single-primitive same-center s shell kinetic energy is 3 alpha / 2" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const exponent: f64 = 0.7;
    const primitives = [_]Primitive{.{
        .exponent = exponent,
        .coefficient = gaussian_orbital.radialNormalization(f64, exponent, 0),
    }};
    const shell = Shell{
        .center = .{ 0, 0, 0 },
        .angular_momentum = 0,
        .primitives = primitives[0..],
    };

    var block: [1]f64 = undefined;
    try shellPairKinetic(f64, shell, shell, &block);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5) * exponent, block[0], 1e-12);
}
