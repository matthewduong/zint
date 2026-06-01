//! Gaussian-basis primitives, shells, and Cartesian-component helpers.
//!
//! A `Shell` is: a center, a total
//! angular momentum L, and a list of contracted primitives shared
//! by every Cartesian component of the shell.

const std = @import("std");
const math = @import("math.zig");
const scalar = @import("scalar.zig");
const types = @import("types.zig");

pub const AngularMomentumOrder = types.AngularMomentumOrder;

/// Highest total angular momentum the library supports per shell.
/// Drives the size of compile-time scratch buffers (recurrence tables,
/// component lists).
pub const max_angular_momentum: AngularMomentumOrder = 8;

pub fn cartesianComponentCount(angular_momentum: AngularMomentumOrder) usize {
    const l = @as(usize, angular_momentum);
    return ((l + 1) * (l + 2)) / 2;
}

pub const ShellKind = enum {
    cartesian,
    spherical,
};

pub fn sphericalComponentCount(angular_momentum: AngularMomentumOrder) usize {
    const l = @as(usize, angular_momentum);
    return 2 * l + 1;
}

pub fn shellComponentCount(kind: ShellKind, angular_momentum: AngularMomentumOrder) usize {
    return switch (kind) {
        .cartesian => cartesianComponentCount(angular_momentum),
        .spherical => sphericalComponentCount(angular_momentum),
    };
}

/// Maximum number of Cartesian components a single shell can have.
pub const max_cartesian_components_per_shell: usize = cartesianComponentCount(max_angular_momentum);
pub const max_spherical_components_per_shell: usize = sphericalComponentCount(max_angular_momentum);
pub const max_components_per_shell: usize =
    @max(max_cartesian_components_per_shell, max_spherical_components_per_shell);

pub fn Primitive(comptime T: type) type {
    return struct {
        exponent: T,
        /// Contraction coefficient with the radial part of the
        /// Cartesian-Gaussian normalization folded in.
        /// The basis loader does this folding at load time via `radialNormalization`.
        coefficient: T,
    };
}

pub const AngularMomentum = struct {
    x: AngularMomentumOrder,
    y: AngularMomentumOrder,
    z: AngularMomentumOrder,

    pub const Error = error{AngularMomentumTooLarge};

    pub fn total(self: AngularMomentum) AngularMomentumOrder {
        // Widen during the sum so we can assert no truncation
        const widened_sum: usize = @as(usize, self.x) + self.y + self.z;
        std.debug.assert(widened_sum <= std.math.maxInt(AngularMomentumOrder));
        return @intCast(widened_sum);
    }

    pub fn validate(self: AngularMomentum, max_per_axis: AngularMomentumOrder) Error!void {
        if (self.x > max_per_axis or self.y > max_per_axis or self.z > max_per_axis) {
            return error.AngularMomentumTooLarge;
        }
    }
};

/// Atom-positioned contracted shell: a total angular momentum L,
/// a center, and shared primitive exponents/coefficients used by every
/// Cartesian component of the shell.
pub fn Shell(comptime T: type) type {
    return struct {
        center: [3]T,
        angular_momentum: AngularMomentumOrder,
        primitives: []const Primitive(T),
        kind: ShellKind = .cartesian,
    };
}

pub fn countAos(comptime T: type, shells: []const Shell(T)) usize {
    scalar.assertFloat(T);
    var count: usize = 0;
    for (shells) |shell| {
        count += shellComponentCount(shell.kind, shell.angular_momentum);
    }
    return count;
}

/// Return cumulative AO offsets for `shells`.
///
/// The returned slice has length `shells.len + 1`; `offsets[i]` is the
/// first AO index for shell `i`, and the final entry is the total AO count.
/// Caller owns the returned slice.
pub fn shellOffsets(
    comptime T: type,
    allocator: std.mem.Allocator,
    shells: []const Shell(T),
) std.mem.Allocator.Error![]usize {
    scalar.assertFloat(T);
    const offsets = try allocator.alloc(usize, shells.len + 1);
    offsets[0] = 0;
    for (shells, 0..) |shell, shell_index| {
        offsets[shell_index + 1] =
            offsets[shell_index] + shellComponentCount(shell.kind, shell.angular_momentum);
    }
    return offsets;
}

/// Radial part of the Cartesian-Gaussian primitive normalization:
///
///     N_radial(α, L) = (2α / π)^(3/4) · (4α)^(L/2)
///
pub fn radialNormalization(
    comptime T: type,
    exponent: T,
    total_angular_momentum: AngularMomentumOrder,
) T {
    scalar.assertFloat(T);
    const scaled_exponent = @as(T, 2) * exponent / @as(T, std.math.pi);
    const s_prefactor = @sqrt(@sqrt(scaled_exponent * scaled_exponent * scaled_exponent));
    const angular_power = math.powUnsigned(T, @as(T, 4) * exponent, total_angular_momentum);
    return s_prefactor * @sqrt(angular_power);
}

/// Component-dependent part of the Cartesian-Gaussian primitive
/// normalization:
///
///     N_angular(lx, ly, lz) = 1 / sqrt((2lx-1)!! · (2ly-1)!! · (2lz-1)!!)
///
/// Applied per Cartesian component inside the integrals kernel. Pairs
/// with `radialNormalization` so that
/// N_total = N_radial(α, L) · N_angular(lx, ly, lz).
pub fn cartesianAngularNorm(
    comptime T: type,
    angular_momentum: AngularMomentum,
) T {
    scalar.assertFloat(T);
    const double_factorial_product =
        math.doubleFactorialOdd(T, angular_momentum.x) *
        math.doubleFactorialOdd(T, angular_momentum.y) *
        math.doubleFactorialOdd(T, angular_momentum.z);
    return 1 / @sqrt(double_factorial_product);
}

/// Write the Cartesian components for a shell of total angular momentum
/// L into `out` in PySCF order: descending lx, then descending ly.
/// Returns the number of components written.
pub fn cartesianComponents(angular_momentum: AngularMomentumOrder, out: []AngularMomentum) usize {
    const total_l = @as(usize, angular_momentum);
    var index: usize = 0;

    for (0..(total_l + 1)) |lx_offset| {
        const lx = total_l - lx_offset;
        const max_ly = total_l - lx;

        for (0..(max_ly + 1)) |ly_offset| {
            const ly = max_ly - ly_offset;
            const lz = total_l - lx - ly;
            out[index] = .{
                .x = @intCast(lx),
                .y = @intCast(ly),
                .z = @intCast(lz),
            };
            index += 1;
        }
    }
    return index;
}

test "cartesianComponentCount matches the standard (L+1)(L+2)/2 series" {
    try std.testing.expectEqual(@as(usize, 1), cartesianComponentCount(0));
    try std.testing.expectEqual(@as(usize, 3), cartesianComponentCount(1));
    try std.testing.expectEqual(@as(usize, 6), cartesianComponentCount(2));
    try std.testing.expectEqual(@as(usize, 10), cartesianComponentCount(3));
    try std.testing.expectEqual(@as(usize, 45), cartesianComponentCount(8));
}

test "shellComponentCount distinguishes Cartesian and spherical shells" {
    try std.testing.expectEqual(@as(usize, 6), shellComponentCount(.cartesian, 2));
    try std.testing.expectEqual(@as(usize, 5), shellComponentCount(.spherical, 2));
}

test "cartesianComponents emits the PySCF order for L = 0, 1, 2" {
    var buf: [max_components_per_shell]AngularMomentum = undefined;

    try std.testing.expectEqual(@as(usize, 1), cartesianComponents(0, &buf));
    try std.testing.expectEqual(AngularMomentum{ .x = 0, .y = 0, .z = 0 }, buf[0]);

    try std.testing.expectEqual(@as(usize, 3), cartesianComponents(1, &buf));
    try std.testing.expectEqual(AngularMomentum{ .x = 1, .y = 0, .z = 0 }, buf[0]);
    try std.testing.expectEqual(AngularMomentum{ .x = 0, .y = 1, .z = 0 }, buf[1]);
    try std.testing.expectEqual(AngularMomentum{ .x = 0, .y = 0, .z = 1 }, buf[2]);

    try std.testing.expectEqual(@as(usize, 6), cartesianComponents(2, &buf));
    try std.testing.expectEqual(AngularMomentum{ .x = 2, .y = 0, .z = 0 }, buf[0]);
    try std.testing.expectEqual(AngularMomentum{ .x = 1, .y = 1, .z = 0 }, buf[1]);
    try std.testing.expectEqual(AngularMomentum{ .x = 1, .y = 0, .z = 1 }, buf[2]);
    try std.testing.expectEqual(AngularMomentum{ .x = 0, .y = 2, .z = 0 }, buf[3]);
    try std.testing.expectEqual(AngularMomentum{ .x = 0, .y = 1, .z = 1 }, buf[4]);
    try std.testing.expectEqual(AngularMomentum{ .x = 0, .y = 0, .z = 2 }, buf[5]);
}

test "Shell stores a runtime primitive slice and a total angular momentum" {
    const GaussianPrimitive = Primitive(f64);
    const ShellF64 = Shell(f64);

    const primitives = [_]GaussianPrimitive{
        .{ .exponent = 3.425250914, .coefficient = 0.1543289673 },
        .{ .exponent = 0.6239137298, .coefficient = 0.5353281423 },
        .{ .exponent = 0.168855404, .coefficient = 0.4446345422 },
    };

    const shell = ShellF64{
        .center = .{ 0.0, 0.0, 0.0 },
        .angular_momentum = 0,
        .primitives = primitives[0..],
    };

    try std.testing.expectEqual(@as(usize, 3), shell.primitives.len);
    try std.testing.expectEqual(@as(usize, 1), cartesianComponentCount(shell.angular_momentum));
}

test "countAos and shellOffsets describe public AO layout" {
    const PrimitiveF64 = Primitive(f64);
    const ShellF64 = Shell(f64);

    const primitives = [_]PrimitiveF64{.{ .exponent = 1.0, .coefficient = 1.0 }};
    const shells = [_]ShellF64{
        .{
            .center = .{ 0.0, 0.0, 0.0 },
            .angular_momentum = 0,
            .primitives = primitives[0..],
        },
        .{
            .center = .{ 0.0, 0.0, 0.0 },
            .angular_momentum = 2,
            .kind = .spherical,
            .primitives = primitives[0..],
        },
    };

    try std.testing.expectEqual(@as(usize, 6), countAos(f64, &shells));

    const offsets = try shellOffsets(f64, std.testing.allocator, &shells);
    defer std.testing.allocator.free(offsets);

    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 6 }, offsets);
}
