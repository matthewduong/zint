//! Boys function values for Hermite Coulomb auxiliary tables.

const std = @import("std");
const scalar = @import("../../scalar.zig");

/// Fill Boys values F_0(argument) through F_max_n(argument).
pub fn fillValues(comptime T: type, max_n: usize, argument: T, out: []T) void {
    scalar.assertFloat(T);
    std.debug.assert(out.len >= max_n + 1);
    std.debug.assert(argument >= 0);

    if (argument == 0) {
        for (0..max_n + 1) |n| {
            out[n] = @as(T, 1) / @as(T, @floatFromInt(2 * n + 1));
        }
        return;
    }

    const zero_order = zeroOrder(T, argument);
    if (argument <= @as(T, @floatFromInt(max_n + 32))) {
        fillValuesDownward(T, max_n, argument, zero_order, out);
    } else {
        fillValuesUpward(T, max_n, argument, zero_order, out);
    }
}

/// Compute F_0 using a small-argument series or large-argument erfc form.
fn zeroOrder(comptime T: type, argument: T) T {
    if (argument < 16) {
        return series(T, 0, argument);
    }

    return @as(T, 0.5) *
        @sqrt(@as(T, std.math.pi) / argument) *
        (@as(T, 1) - erfcAsymptotic(T, @sqrt(argument)));
}

/// Fill Boys values by a stable downward recurrence and rescale to F_0.
fn fillValuesDownward(
    comptime T: type,
    max_n: usize,
    argument: T,
    zero_order: T,
    out: []T,
) void {
    const seed_order = max_n + @as(usize, @intFromFloat(@ceil(argument))) + 32;
    var carry: T = 0;
    var order = seed_order;
    while (order > 0) {
        carry = (@as(T, 2) * argument * carry + @as(T, 1)) /
            @as(T, @floatFromInt(2 * order - 1));
        const n = order - 1;
        if (n <= max_n) {
            out[n] = carry;
        }
        order -= 1;
    }

    const scale = zero_order / out[0];
    for (0..max_n + 1) |n| {
        out[n] *= scale;
    }
}

/// Fill Boys values by upward recurrence for sufficiently large arguments.
fn fillValuesUpward(
    comptime T: type,
    max_n: usize,
    argument: T,
    zero_order: T,
    out: []T,
) void {
    out[0] = zero_order;

    const exponential = @exp(-argument);
    for (0..max_n) |n| {
        out[n + 1] =
            ((@as(T, @floatFromInt(2 * n + 1)) * out[n]) - exponential) /
            (@as(T, 2) * argument);
    }
}

/// Alternating series for one Boys order at small to moderate argument.
fn series(comptime T: type, order: usize, argument: T) T {
    var term = @as(T, 1) / @as(T, @floatFromInt(2 * order + 1));
    var sum = term;

    for (0..200) |k| {
        const numerator = -argument * @as(T, @floatFromInt(2 * order + 2 * k + 1));
        const denominator = @as(T, @floatFromInt(k + 1)) *
            @as(T, @floatFromInt(2 * order + 2 * k + 3));
        term *= numerator / denominator;
        sum += term;
        if (@abs(term) <= @as(T, 1e-16) * @max(@abs(sum), @as(T, 1))) break;
    }

    return sum;
}

/// Asymptotic complement of erf for large `x`, used in the F_0 expression.
fn erfcAsymptotic(comptime T: type, x: T) T {
    const inverse_two_x_squared = @as(T, 0.5) / (x * x);
    var term: T = 1;
    var sum: T = 1;
    var odd: usize = 1;

    while (odd < 40) : (odd += 2) {
        term *= -@as(T, @floatFromInt(odd)) * inverse_two_x_squared;
        sum += term;
        if (@abs(term) <= @as(T, 1e-16) * @max(@abs(sum), @as(T, 1))) break;
    }

    return @exp(-(x * x)) * sum / (@sqrt(@as(T, std.math.pi)) * x);
}

fn expectApproxEqRel(expected: f64, actual: f64, rel_tolerance: f64) !void {
    try std.testing.expect(std.math.isFinite(actual));
    try std.testing.expectApproxEqAbs(expected, actual, @abs(expected) * rel_tolerance);
}

test "Boys function handles zero and small arguments" {
    var values: [4]f64 = undefined;
    fillValues(f64, 3, 0, &values);
    try std.testing.expectApproxEqAbs(@as(f64, 1), values[0], 1e-14);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 3.0), values[1], 1e-14);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 5.0), values[2], 1e-14);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0 / 7.0), values[3], 1e-14);

    fillValues(f64, 3, 1e-8, &values);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9999999966666667), values[0], 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0.33333333133333334), values[1], 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0.19999999857142858), values[2], 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0.14285714174603176), values[3], 1e-15);
}

test "Boys high-order values through n=96 match references" {
    var values: [97]f64 = undefined;

    fillValues(f64, 96, 16, &values);
    try expectApproxEqRel(6.972837426266314e-10, values[96], 1e-13);
    try expectApproxEqRel(8.690979979309443e-10, values[80], 1e-13);
    try expectApproxEqRel(9.910509903955037e-10, values[72], 1e-13);
    try expectApproxEqRel(1.2546702028665064e-9, values[60], 1e-13);

    fillValues(f64, 96, 30, &values);
    try expectApproxEqRel(6.989679804667571e-16, values[96], 1e-13);
    try expectApproxEqRel(9.161520951834352e-16, values[80], 1e-13);
    try expectApproxEqRel(1.0838174700756975e-15, values[72], 1e-13);
}

test "fuzz Boys function values are finite, positive, and monotonically decreasing" {
    try std.testing.fuzz({}, fuzzBoys, .{});
}

fn fuzzBoys(_: void, smith: *std.testing.Smith) !void {
    @disableInstrumentation();
    const max_n: usize = smith.valueRangeAtMost(u7, 0, 96);
    const argument: f64 = @abs(smith.value(f64));
    if (!std.math.isFinite(argument)) return;

    var values: [97]f64 = undefined;
    fillValues(f64, max_n, argument, values[0 .. max_n + 1]);

    // F_0(x) = 0.5*sqrt(pi/x) for large x, always representable in f64
    try std.testing.expect(values[0] > 0);

    for (0..max_n + 1) |n| {
        try std.testing.expect(std.math.isFinite(values[n]));
        try std.testing.expect(values[n] >= 0);
    }

    // F_n(x) >= F_{n+1}(x): Boys function is monotonically decreasing in order
    for (0..max_n) |n| {
        try std.testing.expect(values[n] >= values[n + 1]);
    }
}

test "Boys high-order values near argument 16 stay positive and finite" {
    var values: [97]f64 = undefined;
    const arguments = [_]f64{ 15.999, 16, 16.001 };

    for (arguments) |argument| {
        fillValues(f64, 96, argument, &values);
        for (values) |value| {
            try std.testing.expect(std.math.isFinite(value));
            try std.testing.expect(value > 0);
        }
    }
}
