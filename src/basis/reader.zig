//! Raw Basis Set Exchange JSON reader.

const std = @import("std");
const types = @import("../types.zig");

pub const FunctionType = enum {
    gto,
    gto_cartesian,
    gto_spherical,
};

pub const BasisJson = struct {
    name: []const u8,
    elements: std.json.ArrayHashMap(ElementJson),
};

pub const ElementJson = struct {
    electron_shells: []ShellJson = &.{},
    ecp_electrons: ?std.json.Value = null,
    ecp_potentials: ?std.json.Value = null,
};

pub const JsonFloat = struct {
    value: f64,

    pub fn jsonParse(
        allocator: std.mem.Allocator,
        source: anytype,
        options: std.json.ParseOptions,
    ) !JsonFloat {
        const token = try source.nextAllocMax(allocator, .alloc_if_needed, options.max_value_len.?);
        defer freeAllocatedJsonToken(allocator, token);

        const raw = switch (token) {
            inline .number, .allocated_number, .string, .allocated_string => |slice| slice,
            else => return error.UnexpectedToken,
        };
        const trimmed = std.mem.trim(u8, raw, " \t\r\n");
        return .{ .value = try std.fmt.parseFloat(f64, trimmed) };
    }
};

pub const ShellJson = struct {
    angular_momentum: []types.AngularMomentumOrder,
    coefficients: [][]JsonFloat,
    exponents: []JsonFloat,
    function_type: FunctionType,
    region: []const u8 = "",
};

pub const ParsedBasisJson = std.json.Parsed(BasisJson);

pub const ParseJsonError = std.json.ParseError(std.json.Scanner);

pub fn parseJson(allocator: std.mem.Allocator, json_text: []const u8) ParseJsonError!ParsedBasisJson {
    return std.json.parseFromSlice(
        BasisJson,
        allocator,
        json_text,
        .{ .ignore_unknown_fields = true, .allocate = .alloc_always },
    );
}

fn freeAllocatedJsonToken(allocator: std.mem.Allocator, token: std.json.Token) void {
    switch (token) {
        .allocated_number, .allocated_string => |slice| allocator.free(slice),
        else => {},
    }
}

test "parseJson reads BSE string floats" {
    const text =
        \\{
        \\  "name": "STO-3G",
        \\  "ignored": "field",
        \\  "elements": {
        \\    "1": {
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [["0.1543289673E+00 "]],
        \\        "exponents": [" 0.3425250914E+01"],
        \\        "function_type": "gto"
        \\      }]
        \\    }
        \\  }
        \\}
    ;

    const parsed = try parseJson(std.testing.allocator, text);
    defer parsed.deinit();

    try std.testing.expectEqualStrings("STO-3G", parsed.value.name);
    const hydrogen = parsed.value.elements.map.get("1").?;
    try std.testing.expectEqual(@as(usize, 1), hydrogen.electron_shells.len);
    try std.testing.expectApproxEqAbs(@as(f64, 3.425250914), hydrogen.electron_shells[0].exponents[0].value, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.1543289673), hydrogen.electron_shells[0].coefficients[0][0].value, 1e-12);
}
