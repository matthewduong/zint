//! Pure scalar math helpers shared across the library.

const std = @import("std");
const scalar = @import("scalar.zig");

pub fn powUnsigned(comptime T: type, base: T, exponent: u8) T {
    scalar.assertFloat(T);
    var value: T = 1;
    for (0..exponent) |_| {
        value *= base;
    }
    return value;
}

pub inline fn isOdd(value: u8) bool {
    return (value & 1) != 0;
}

pub fn doubleFactorialOdd(comptime T: type, n: u8) T {
    scalar.assertFloat(T);
    var value: T = 1;
    var k: isize = @as(isize, @intCast(n)) * 2 - 1;
    while (k > 1) : (k -= 2) {
        value *= @as(T, @floatFromInt(k));
    }
    return value;
}

test "powUnsigned multiplies base exponent times" {
    try std.testing.expectEqual(@as(f64, 1.0), powUnsigned(f64, 2.0, 0));
    try std.testing.expectEqual(@as(f64, 8.0), powUnsigned(f64, 2.0, 3));
    try std.testing.expectEqual(@as(f64, 1.0), powUnsigned(f64, 5.0, 0));
}

test "isOdd detects odd u8 values" {
    try std.testing.expect(!isOdd(0));
    try std.testing.expect(isOdd(1));
    try std.testing.expect(!isOdd(2));
    try std.testing.expect(isOdd(255));
}

test "doubleFactorialOdd computes (2n-1)!! and returns 1 for n=0" {
    try std.testing.expectEqual(@as(f64, 1.0), doubleFactorialOdd(f64, 0));
    try std.testing.expectEqual(@as(f64, 1.0), doubleFactorialOdd(f64, 1));
    try std.testing.expectEqual(@as(f64, 3.0), doubleFactorialOdd(f64, 2));
    try std.testing.expectEqual(@as(f64, 15.0), doubleFactorialOdd(f64, 3));
    try std.testing.expectEqual(@as(f64, 105.0), doubleFactorialOdd(f64, 4));
}
