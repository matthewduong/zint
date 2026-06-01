//! zint — Gaussian orbital integrals library (early WIP).
//!
//! The library surface is still being built up.

const std = @import("std");

pub const basis = @import("basis.zig");
pub const gaussian_orbital = @import("gaussian_orbital.zig");
pub const integrals = @import("integrals.zig");
pub const math = @import("math.zig");
pub const matrix = @import("matrix.zig");
pub const scalar = @import("scalar.zig");
pub const types = @import("types.zig");

test {
    std.testing.refAllDecls(basis);
    std.testing.refAllDecls(gaussian_orbital);
    std.testing.refAllDecls(integrals);
    std.testing.refAllDecls(math);
    std.testing.refAllDecls(matrix);
    std.testing.refAllDecls(scalar);
    std.testing.refAllDecls(types);
}
