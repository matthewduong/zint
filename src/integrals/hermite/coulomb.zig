//! Hermite Coulomb auxiliary recurrence tables.

const std = @import("std");
const boys = @import("boys.zig");

/// Per-pair extent of Coulomb auxiliary entries actually needed for the
/// current shell pair / quartet. Used by `CoulombTable.reset` to bound
/// the bottom-up fill; entries above this extent are left untouched.
pub const CoulombExtent = struct {
    x_max: usize,
    y_max: usize,
    z_max: usize,
};

/// Hermite Coulomb auxiliary table R(n, x, y, z) plus its Boys workspace.
pub fn CoulombTable(comptime T: type) type {
    return struct {
        values: []T,
        boys_values: []T,
        x_count: usize,
        y_count: usize,
        z_count: usize,
        n_count: usize,

        const Self = @This();

        /// Allocate storage for all auxiliary entries up to the supplied maxima.
        pub fn init(
            allocator: std.mem.Allocator,
            x_max: usize,
            y_max: usize,
            z_max: usize,
            n_max: usize,
        ) std.mem.Allocator.Error!Self {
            const x_count = x_max + 1;
            const y_count = y_max + 1;
            const z_count = z_max + 1;
            const n_count = n_max + 1;
            const entry_count = x_count * y_count * z_count * n_count;
            const values = try allocator.alloc(T, entry_count);
            errdefer allocator.free(values);
            return .{
                .values = values,
                .boys_values = try allocator.alloc(T, n_count),
                .x_count = x_count,
                .y_count = y_count,
                .z_count = z_count,
                .n_count = n_count,
            };
        }

        /// Release Coulomb and Boys work buffers.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.values);
            allocator.free(self.boys_values);
            self.* = undefined;
        }

        /// Fill the table for a single primitive pair / quartet. `extent`
        /// gives the actual `(x_max, y_max, z_max)` the kernel will read
        /// (typically `lhs.x + rhs.x`, etc.); the bottom-up fill walks
        /// `s = x + y + z` from 0 up to `x_max + y_max + z_max` so each
        /// entry's dependencies are already written by the previous level.
        pub fn reset(self: *Self, eta: T, diff: [3]T, extent: CoulombExtent) void {
            std.debug.assert(extent.x_max < self.x_count);
            std.debug.assert(extent.y_max < self.y_count);
            std.debug.assert(extent.z_max < self.z_count);

            const n_max = extent.x_max + extent.y_max + extent.z_max;
            std.debug.assert(n_max < self.n_count);

            const argument = eta *
                (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
            boys.fillValues(T, n_max, argument, self.boys_values);

            // (x=0, y=0, z=0) base: R(n, 0, 0, 0) = (-2η)^n * F_n(x).
            var signed_power: T = 1;
            const base = -@as(T, 2) * eta;
            for (0..n_max + 1) |n| {
                self.values[self.index(n, 0, 0, 0)] = signed_power * self.boys_values[n];
                signed_power *= base;
            }

            // Forward fill levels s = 1, 2, ..., n_max. At level s, an
            // entry (n, x, y, z) with x+y+z=s reads from level s-1 (and
            // s-2 along the same axis) at n+1, both already populated.
            var s: usize = 1;
            while (s <= n_max) : (s += 1) {
                const n_upper_exclusive = n_max + 1 - s;
                var x: usize = 0;
                const x_hi = @min(s, extent.x_max);
                while (x <= x_hi) : (x += 1) {
                    const remainder = s - x;
                    if (remainder > extent.y_max + extent.z_max) continue;
                    const y_lo: usize = if (remainder > extent.z_max) remainder - extent.z_max else 0;
                    const y_hi = @min(remainder, extent.y_max);
                    if (y_lo > y_hi) continue;
                    for (y_lo..y_hi + 1) |y| {
                        const z = remainder - y;
                        self.fillAxisColumn(x, y, z, n_upper_exclusive, diff);
                    }
                }
            }
        }

        /// Read one auxiliary entry R(n, x, y, z).
        pub fn get(self: *const Self, n: usize, x: usize, y: usize, z: usize) T {
            std.debug.assert(x < self.x_count);
            std.debug.assert(y < self.y_count);
            std.debug.assert(z < self.z_count);
            std.debug.assert(n < self.n_count);
            return self.values[self.index(n, x, y, z)];
        }

        /// Compute R(n, x, y, z) for n in [0, n_upper_exclusive) using the
        /// MD recurrence on the highest-priority nonzero axis.
        fn fillAxisColumn(
            self: *Self,
            x: usize,
            y: usize,
            z: usize,
            n_upper_exclusive: usize,
            diff: [3]T,
        ) void {
            if (x > 0) {
                const lower_axis_coefficient = @as(T, @floatFromInt(x - 1));
                for (0..n_upper_exclusive) |n| {
                    var value = diff[0] * self.values[self.index(n + 1, x - 1, y, z)];
                    if (x >= 2) {
                        value += lower_axis_coefficient *
                            self.values[self.index(n + 1, x - 2, y, z)];
                    }
                    self.values[self.index(n, x, y, z)] = value;
                }
                return;
            }
            if (y > 0) {
                const lower_axis_coefficient = @as(T, @floatFromInt(y - 1));
                for (0..n_upper_exclusive) |n| {
                    var value = diff[1] * self.values[self.index(n + 1, x, y - 1, z)];
                    if (y >= 2) {
                        value += lower_axis_coefficient *
                            self.values[self.index(n + 1, x, y - 2, z)];
                    }
                    self.values[self.index(n, x, y, z)] = value;
                }
                return;
            }
            // z > 0: the s >= 1, x = y = 0 branch.
            const lower_axis_coefficient = @as(T, @floatFromInt(z - 1));
            for (0..n_upper_exclusive) |n| {
                var value = diff[2] * self.values[self.index(n + 1, x, y, z - 1)];
                if (z >= 2) {
                    value += lower_axis_coefficient *
                        self.values[self.index(n + 1, x, y, z - 2)];
                }
                self.values[self.index(n, x, y, z)] = value;
            }
        }

        fn index(self: *const Self, n: usize, x: usize, y: usize, z: usize) usize {
            return (((x * self.y_count) + y) * self.z_count + z) * self.n_count + n;
        }
    };
}
