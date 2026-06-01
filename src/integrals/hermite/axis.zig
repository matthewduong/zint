//! One-dimensional Hermite expansion tables.

const std = @import("std");
const scalar = @import("../../scalar.zig");
const gaussian_orbital = @import("../../gaussian_orbital.zig");
const common = @import("../common.zig");
const types = @import("../../types.zig");

const max_shifted_angular_momentum: usize =
    @as(usize, common.max_supported_angular_momentum) + 2;
const axis_order_count: usize = max_shifted_angular_momentum + 1;
const axis_t_count: usize = 2 * max_shifted_angular_momentum + 1;
const axis_table_entries: usize = axis_order_count * axis_order_count * axis_t_count;

/// One-axis Hermite expansion table indexed by `(lhs_order, rhs_order, t)`.
pub fn HermiteExpansionTable(comptime T: type) type {
    return struct {
        values: *[axis_table_entries]T,

        const Self = @This();

        /// Read one Hermite coefficient E_t^(lhs_order,rhs_order).
        pub fn get(self: Self, lhs_order: usize, rhs_order: usize, hermite_order: usize) T {
            return self.values[index(lhs_order, rhs_order, hermite_order)];
        }

        /// Write one Hermite coefficient E_t^(lhs_order,rhs_order).
        pub fn set(
            self: Self,
            lhs_order: usize,
            rhs_order: usize,
            hermite_order: usize,
            value: T,
        ) void {
            self.values[index(lhs_order, rhs_order, hermite_order)] = value;
        }

        fn index(lhs_order: usize, rhs_order: usize, hermite_order: usize) usize {
            return ((lhs_order * axis_order_count) + rhs_order) * axis_t_count + hermite_order;
        }
    };
}

/// Extra per-axis extent requested by kernels that need shifted overlaps.
pub const AxisExtentOptions = struct {
    rhs_max_extra: usize = 0,
};

/// Three-axis Hermite expansion tables for a single primitive pair.
/// Owns the per-axis stack storage and bundles the per-pair fill + view
/// operations so kernels don't repeat the triple-buffer / triple-view
/// boilerplate.
pub fn HermiteAxisTables(comptime T: type) type {
    return struct {
        buffers: [3][axis_table_entries]T,

        const Self = @This();

        pub fn fill(
            self: *Self,
            pair: common.PrimitivePair(T),
            lhs_center: [3]T,
            rhs_center: [3]T,
            lhs_max_per_axis: [3]types.AngularMomentumOrder,
            rhs_max_per_axis: [3]types.AngularMomentumOrder,
            options: AxisExtentOptions,
        ) void {
            inline for (0..3) |axis| {
                fillAxisTable(
                    T,
                    pair,
                    lhs_center[axis],
                    rhs_center[axis],
                    axis,
                    lhs_max_per_axis[axis],
                    @as(usize, rhs_max_per_axis[axis]) + options.rhs_max_extra,
                    &self.buffers[axis],
                );
            }
        }

        pub fn views(self: *Self) [3]HermiteExpansionTable(T) {
            return .{
                .{ .values = &self.buffers[0] },
                .{ .values = &self.buffers[1] },
                .{ .values = &self.buffers[2] },
            };
        }
    };
}

/// Fill one axis table from the s-type seed through the requested angular orders.
fn fillAxisTable(
    comptime T: type,
    pair: common.PrimitivePair(T),
    lhs_center_axis: T,
    rhs_center_axis: T,
    axis: usize,
    lhs_max: usize,
    rhs_max: usize,
    table: *[axis_table_entries]T,
) void {
    scalar.assertFloat(T);
    std.debug.assert(lhs_max < axis_order_count);
    std.debug.assert(rhs_max < axis_order_count);

    @memset(table, 0);

    const values = HermiteExpansionTable(T){ .values = table };
    const product_minus_lhs = pair.product_center[axis] - lhs_center_axis;
    const product_minus_rhs = pair.product_center[axis] - rhs_center_axis;
    const lhs_minus_rhs = lhs_center_axis - rhs_center_axis;
    const half_inverse_combined_exponent = @as(T, 0.5) / pair.combined_exponent;
    const reduced_exponent = pair.lhs_exponent * pair.rhs_exponent / pair.combined_exponent;

    values.set(
        0,
        0,
        0,
        @exp(-reduced_exponent * lhs_minus_rhs * lhs_minus_rhs),
    );

    for (0..lhs_max) |lhs_order| {
        for (0..lhs_order + 2) |hermite_order| {
            var cell_value: T = product_minus_lhs *
                values.get(lhs_order, 0, hermite_order);
            if (hermite_order >= 1) {
                cell_value += half_inverse_combined_exponent *
                    values.get(lhs_order, 0, hermite_order - 1);
            }
            if (hermite_order + 1 < axis_t_count) {
                cell_value += @as(T, @floatFromInt(hermite_order + 1)) *
                    values.get(lhs_order, 0, hermite_order + 1);
            }
            values.set(lhs_order + 1, 0, hermite_order, cell_value);
        }
    }

    for (0..rhs_max) |rhs_order| {
        for (0..lhs_max + 1) |lhs_order| {
            for (0..lhs_order + rhs_order + 2) |hermite_order| {
                var cell_value: T = product_minus_rhs *
                    values.get(lhs_order, rhs_order, hermite_order);
                if (hermite_order >= 1) {
                    cell_value += half_inverse_combined_exponent *
                        values.get(lhs_order, rhs_order, hermite_order - 1);
                }
                if (hermite_order + 1 < axis_t_count) {
                    cell_value += @as(T, @floatFromInt(hermite_order + 1)) *
                        values.get(lhs_order, rhs_order, hermite_order + 1);
                }
                values.set(lhs_order, rhs_order + 1, hermite_order, cell_value);
            }
        }
    }
}

/// Triple product of the t=0 hermite-expansion entries on each axis. The
/// product-Gaussian volume factor is NOT included; callers that need the
/// full primitive overlap should multiply by `pair.productGaussianVolume()`
/// once after combining however many of these table-products they need.
pub fn overlapTableProduct(
    comptime T: type,
    lhs: gaussian_orbital.AngularMomentum,
    rhs: gaussian_orbital.AngularMomentum,
    tables: [3]HermiteExpansionTable(T),
) T {
    return tables[0].get(lhs.x, rhs.x, 0) *
        tables[1].get(lhs.y, rhs.y, 0) *
        tables[2].get(lhs.z, rhs.z, 0);
}

test "Hermite overlap seed matches normalized s primitive self-overlap before coefficients" {
    const pair = common.buildPrimitivePair(
        f64,
        0.7,
        .{ 0, 0, 0 },
        0.7,
        .{ 0, 0, 0 },
    );

    var table_x: [axis_table_entries]f64 = undefined;
    var table_y: [axis_table_entries]f64 = undefined;
    var table_z: [axis_table_entries]f64 = undefined;
    fillAxisTable(f64, pair, 0, 0, 0, 0, 0, &table_x);
    fillAxisTable(f64, pair, 0, 0, 1, 0, 0, &table_y);
    fillAxisTable(f64, pair, 0, 0, 2, 0, 0, &table_z);

    const unnormalized = pair.productGaussianVolume() * overlapTableProduct(
        f64,
        .{ .x = 0, .y = 0, .z = 0 },
        .{ .x = 0, .y = 0, .z = 0 },
        .{
            .{ .values = &table_x },
            .{ .values = &table_y },
            .{ .values = &table_z },
        },
    );
    const coefficient = gaussian_orbital.radialNormalization(f64, 0.7, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 1), coefficient * coefficient * unnormalized, 1e-12);
}
