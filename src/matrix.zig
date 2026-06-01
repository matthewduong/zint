//! Dense symmetric matrix

const std = @import("std");
const scalar = @import("scalar.zig");

pub const DimensionError = error{DimensionTooLarge};
pub const SymmetricMatrixInitError = std.mem.Allocator.Error || DimensionError;

pub fn packedSymmetricEntryCount(dimension: usize) DimensionError!usize {
    if (dimension == std.math.maxInt(usize)) return error.DimensionTooLarge;

    const next = dimension + 1;
    const lhs = if ((dimension & 1) == 0) dimension / 2 else dimension;
    const rhs = if ((dimension & 1) == 0) next else next / 2;
    return checkedMul(lhs, rhs);
}

fn checkedMul(lhs: usize, rhs: usize) DimensionError!usize {
    if (lhs != 0 and rhs > std.math.maxInt(usize) / lhs) {
        return error.DimensionTooLarge;
    }
    return lhs * rhs;
}

/// Packed upper-triangular symmetric matrix.
pub fn SymmetricMatrix(comptime T: type) type {
    scalar.assertFloat(T);

    return struct {
        dimension: usize,
        values: []T,

        const Self = @This();

        pub fn init(allocator: std.mem.Allocator, dimension: usize) SymmetricMatrixInitError!Self {
            const entry_count = try packedSymmetricEntryCount(dimension);
            return .{
                .dimension = dimension,
                .values = try allocator.alloc(T, entry_count),
            };
        }

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.values);
            self.* = undefined;
        }

        pub fn get(self: *const Self, row: usize, col: usize) T {
            std.debug.assert(row < self.dimension);
            std.debug.assert(col < self.dimension);
            return self.values[packedIndex(row, col)];
        }

        pub fn set(self: *Self, row: usize, col: usize, value: T) void {
            std.debug.assert(row < self.dimension);
            std.debug.assert(col < self.dimension);
            self.values[packedIndex(row, col)] = value;
        }

        fn packedIndex(row: usize, col: usize) usize {
            const smaller = @min(row, col);
            const larger = @max(row, col);
            return larger * (larger + 1) / 2 + smaller;
        }
    };
}

test "packedSymmetricEntryCount checks dimension overflow" {
    try std.testing.expectEqual(@as(usize, 0), try packedSymmetricEntryCount(0));
    try std.testing.expectEqual(@as(usize, 1), try packedSymmetricEntryCount(1));
    try std.testing.expectEqual(@as(usize, 6), try packedSymmetricEntryCount(3));
    try std.testing.expectError(error.DimensionTooLarge, packedSymmetricEntryCount(std.math.maxInt(usize)));
}

test "SymmetricMatrix stores upper-triangular packed entries and mirrors writes" {
    var symmetric = try SymmetricMatrix(f64).init(std.testing.allocator, 3);
    defer symmetric.deinit(std.testing.allocator);

    symmetric.set(0, 0, 1.0);
    symmetric.set(0, 1, 2.0);
    symmetric.set(1, 1, 3.0);
    symmetric.set(0, 2, 4.0);
    symmetric.set(1, 2, 5.0);
    symmetric.set(2, 2, 6.0);

    try std.testing.expectEqual(@as(usize, 6), symmetric.values.len);

    try std.testing.expectEqual(@as(f64, 1.0), symmetric.get(0, 0));
    try std.testing.expectEqual(@as(f64, 2.0), symmetric.get(0, 1));
    try std.testing.expectEqual(@as(f64, 2.0), symmetric.get(1, 0));
    try std.testing.expectEqual(@as(f64, 3.0), symmetric.get(1, 1));
    try std.testing.expectEqual(@as(f64, 4.0), symmetric.get(0, 2));
    try std.testing.expectEqual(@as(f64, 4.0), symmetric.get(2, 0));
    try std.testing.expectEqual(@as(f64, 5.0), symmetric.get(1, 2));
    try std.testing.expectEqual(@as(f64, 5.0), symmetric.get(2, 1));
    try std.testing.expectEqual(@as(f64, 6.0), symmetric.get(2, 2));

    symmetric.set(1, 0, 7.0);
    try std.testing.expectEqual(@as(f64, 7.0), symmetric.get(0, 1));
}
