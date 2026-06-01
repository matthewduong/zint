//! Basis-set loading and conversion helpers.

const std = @import("std");
const gaussian_orbital = @import("gaussian_orbital.zig");
const math = @import("math.zig");
const scalar = @import("scalar.zig");
const types = @import("types.zig");

pub const AtomicNumber = types.AtomicNumber;
const AngularMomentumOrder = types.AngularMomentumOrder;

pub const reader = @import("basis/reader.zig");

pub const FunctionType = reader.FunctionType;
pub const ParsedBasisJson = reader.ParsedBasisJson;
pub const parseJson = reader.parseJson;

pub const ConvertBasisError = std.mem.Allocator.Error || error{
    InvalidBasisShell,
    AngularMomentumTooLarge,
};
pub const ParseError = ConvertBasisError || error{InvalidBasisJson};
pub const ReadFileError = std.Io.Dir.ReadFileAllocError || ParseError;
pub const ShellsForAtomError = std.mem.Allocator.Error || error{ ElementNotFound, UnsupportedEcp };

pub fn ContractedShell(comptime T: type) type {
    scalar.assertFloat(T);

    return struct {
        angular_momentum: AngularMomentumOrder,
        function_type: FunctionType,
        kind: gaussian_orbital.ShellKind,
        primitives: []gaussian_orbital.Primitive(T),

        const Self = @This();

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.primitives);
            self.* = undefined;
        }
    };
}

pub fn ElementBasis(comptime T: type) type {
    scalar.assertFloat(T);

    return struct {
        atomic_number: AtomicNumber,
        shells: []ContractedShell(T),

        const Self = @This();

        fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (self.shells) |*shell| {
                shell.deinit(allocator);
            }
            allocator.free(self.shells);
            self.* = undefined;
        }
    };
}

pub fn BasisSet(comptime T: type) type {
    scalar.assertFloat(T);

    return struct {
        name: []const u8,
        elements: []ElementBasis(T),
        ecp_elements: []AtomicNumber,

        const Self = @This();

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (self.elements) |*element_basis| {
                element_basis.deinit(allocator);
            }
            allocator.free(self.elements);
            allocator.free(self.ecp_elements);
            allocator.free(self.name);
            self.* = undefined;
        }

        pub fn element(self: Self, atomic_number: AtomicNumber) ?*const ElementBasis(T) {
            // `elements` is kept sorted by atomic_number in `fromJson`,
            // so a binary search is correct and faster than a linear
            // scan once a basis set covers many elements.
            var lower: usize = 0;
            var upper: usize = self.elements.len;
            while (lower < upper) {
                const middle = lower + (upper - lower) / 2;
                const candidate = &self.elements[middle];
                if (candidate.atomic_number == atomic_number) return candidate;
                if (candidate.atomic_number < atomic_number) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            return null;
        }

        pub fn elementHasEcp(self: Self, atomic_number: AtomicNumber) bool {
            var lower: usize = 0;
            var upper: usize = self.ecp_elements.len;
            while (lower < upper) {
                const middle = lower + (upper - lower) / 2;
                const candidate = self.ecp_elements[middle];
                if (candidate == atomic_number) return true;
                if (candidate < atomic_number) {
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            return false;
        }

        /// Returns one `Shell` per contracted shell on this element, all
        /// positioned at `center`. Primitives are borrowed from the
        /// basis set; the caller owns only the returned slice.
        pub fn shellsForAtom(
            self: Self,
            allocator: std.mem.Allocator,
            atomic_number: AtomicNumber,
            center: [3]T,
        ) ShellsForAtomError![]gaussian_orbital.Shell(T) {
            const ShellOut = gaussian_orbital.Shell(T);
            if (self.elementHasEcp(atomic_number)) return error.UnsupportedEcp;
            const element_basis = self.element(atomic_number) orelse return error.ElementNotFound;

            const shells = try allocator.alloc(ShellOut, element_basis.shells.len);
            errdefer allocator.free(shells);

            for (element_basis.shells, shells) |source_shell, *destination_shell| {
                destination_shell.* = .{
                    .center = center,
                    .angular_momentum = source_shell.angular_momentum,
                    .kind = source_shell.kind,
                    .primitives = source_shell.primitives,
                };
            }
            return shells;
        }
    };
}

pub fn parse(
    comptime T: type,
    allocator: std.mem.Allocator,
    json_text: []const u8,
) ParseError!BasisSet(T) {
    const parsed = reader.parseJson(allocator, json_text) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidBasisJson,
    };
    defer parsed.deinit();
    return fromJson(T, allocator, parsed.value);
}

pub fn readFile(
    comptime T: type,
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) ReadFileError!BasisSet(T) {
    const json_text = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .unlimited);
    defer allocator.free(json_text);
    return parse(T, allocator, json_text);
}

fn fromJson(comptime T: type, allocator: std.mem.Allocator, json: reader.BasisJson) ParseError!BasisSet(T) {
    const name = try allocator.dupe(u8, json.name);
    errdefer allocator.free(name);

    var elements: std.ArrayList(ElementBasis(T)) = .empty;
    var ecp_elements: std.ArrayList(AtomicNumber) = .empty;
    errdefer {
        for (elements.items) |*element_basis| {
            element_basis.deinit(allocator);
        }
        elements.deinit(allocator);
        ecp_elements.deinit(allocator);
    }

    var it = json.elements.map.iterator();
    while (it.next()) |entry| {
        const atomic_number = std.fmt.parseInt(AtomicNumber, entry.key_ptr.*, 10) catch return error.InvalidBasisJson;
        if (elementJsonHasEcp(entry.value_ptr.*)) {
            try ecp_elements.append(allocator, atomic_number);
            continue;
        }
        {
            var element_basis = try convertElement(T, allocator, atomic_number, entry.value_ptr.*);
            errdefer element_basis.deinit(allocator);
            try elements.append(allocator, element_basis);
        }
    }

    const owned_elements = try elements.toOwnedSlice(allocator);
    errdefer {
        for (owned_elements) |*element_basis| {
            element_basis.deinit(allocator);
        }
        allocator.free(owned_elements);
    }
    std.mem.sort(ElementBasis(T), owned_elements, {}, elementLessThan(T));

    const owned_ecp_elements = try ecp_elements.toOwnedSlice(allocator);
    std.mem.sort(AtomicNumber, owned_ecp_elements, {}, std.sort.asc(AtomicNumber));

    return .{
        .name = name,
        .elements = owned_elements,
        .ecp_elements = owned_ecp_elements,
    };
}

fn convertElement(
    comptime T: type,
    allocator: std.mem.Allocator,
    atomic_number: AtomicNumber,
    json: reader.ElementJson,
) ConvertBasisError!ElementBasis(T) {
    var shells: std.ArrayList(ContractedShell(T)) = .empty;
    errdefer {
        for (shells.items) |*shell| {
            shell.deinit(allocator);
        }
        shells.deinit(allocator);
    }

    var converted_shell_count: usize = 0;
    for (json.electron_shells) |json_shell| {
        try validateShell(json_shell);
        converted_shell_count += json_shell.coefficients.len;
    }
    try shells.ensureTotalCapacity(allocator, converted_shell_count);

    // Match PySCF AO ordering for general/SP shells: all contractions of
    // a lower angular momentum appear before contractions of a higher
    // angular momentum on the same atom.
    for (0..@as(usize, gaussian_orbital.max_angular_momentum) + 1) |angular_momentum_index| {
        const angular_momentum: AngularMomentumOrder = @intCast(angular_momentum_index);
        for (json.electron_shells) |json_shell| {
            for (json_shell.coefficients, 0..) |coefficients, coefficient_index| {
                if (coefficientAngularMomentum(json_shell, coefficient_index) != angular_momentum) {
                    continue;
                }
                const shell = try convertShell(
                    T,
                    allocator,
                    angular_momentum,
                    json_shell.function_type,
                    json_shell.exponents,
                    coefficients,
                );
                shells.appendAssumeCapacity(shell);
            }
        }
    }

    return .{
        .atomic_number = atomic_number,
        .shells = try shells.toOwnedSlice(allocator),
    };
}

fn convertShell(
    comptime T: type,
    allocator: std.mem.Allocator,
    angular_momentum: AngularMomentumOrder,
    function_type: FunctionType,
    exponents: []const reader.JsonFloat,
    coefficients: []const reader.JsonFloat,
) ConvertBasisError!ContractedShell(T) {
    const GaussianPrimitive = gaussian_orbital.Primitive(T);
    const primitives = try allocator.alloc(GaussianPrimitive, exponents.len);
    errdefer allocator.free(primitives);

    // Fold the radial (α, L)-dependent part of the Cartesian-Gaussian
    // normalization into the stored coefficient at load time. The
    // integrals kernel only applies the per-component angular part.
    for (primitives, exponents, coefficients) |*primitive, exponent_f64, coefficient_f64| {
        if (exponent_f64.value <= 0 or !std.math.isFinite(exponent_f64.value)) {
            return error.InvalidBasisShell;
        }
        if (!std.math.isFinite(coefficient_f64.value)) {
            return error.InvalidBasisShell;
        }
        const exponent: T = @floatCast(exponent_f64.value);
        const raw_coefficient: T = @floatCast(coefficient_f64.value);
        const radial_norm = gaussian_orbital.radialNormalization(T, exponent, angular_momentum);
        primitive.* = .{
            .exponent = exponent,
            .coefficient = raw_coefficient * radial_norm,
        };
    }
    try normalizeContraction(T, primitives, angular_momentum);

    return .{
        .angular_momentum = angular_momentum,
        .function_type = function_type,
        .kind = shellKind(function_type),
        .primitives = primitives,
    };
}

fn normalizeContraction(
    comptime T: type,
    primitives: []gaussian_orbital.Primitive(T),
    angular_momentum: AngularMomentumOrder,
) ConvertBasisError!void {
    var norm_squared: T = 0;
    const two_to_l = math.powUnsigned(T, 2, angular_momentum);
    for (primitives) |lhs| {
        for (primitives) |rhs| {
            const combined_exponent = lhs.exponent + rhs.exponent;
            const p_to_l = math.powUnsigned(T, combined_exponent, angular_momentum);
            const radial_overlap =
                (@as(T, std.math.pi) * @sqrt(@as(T, std.math.pi))) /
                (two_to_l * p_to_l * combined_exponent * @sqrt(combined_exponent));
            norm_squared += lhs.coefficient * rhs.coefficient * radial_overlap;
        }
    }

    if (norm_squared <= 0 or !std.math.isFinite(norm_squared)) {
        return error.InvalidBasisShell;
    }

    const scale = 1 / @sqrt(norm_squared);
    for (primitives) |*primitive| {
        primitive.coefficient *= scale;
    }
}

fn shellKind(function_type: FunctionType) gaussian_orbital.ShellKind {
    return switch (function_type) {
        .gto, .gto_cartesian => .cartesian,
        .gto_spherical => .spherical,
    };
}

fn validateShell(shell: reader.ShellJson) ConvertBasisError!void {
    if (shell.exponents.len == 0) return error.InvalidBasisShell;
    if (shell.angular_momentum.len == 0) return error.InvalidBasisShell;
    if (shell.angular_momentum.len != 1 and shell.angular_momentum.len != shell.coefficients.len) {
        return error.InvalidBasisShell;
    }

    for (shell.angular_momentum) |angular_momentum| {
        if (angular_momentum > gaussian_orbital.max_angular_momentum) {
            return error.AngularMomentumTooLarge;
        }
    }

    for (shell.coefficients) |coefficients| {
        if (coefficients.len != shell.exponents.len) return error.InvalidBasisShell;
    }
}

fn coefficientAngularMomentum(shell: reader.ShellJson, coefficient_index: usize) AngularMomentumOrder {
    if (shell.angular_momentum.len == 1) return shell.angular_momentum[0];
    return shell.angular_momentum[coefficient_index];
}

fn elementJsonHasEcp(element: reader.ElementJson) bool {
    return element.ecp_electrons != null or element.ecp_potentials != null;
}

fn elementLessThan(comptime T: type) fn (void, ElementBasis(T), ElementBasis(T)) bool {
    return struct {
        fn lessThan(_: void, lhs: ElementBasis(T), rhs: ElementBasis(T)) bool {
            return lhs.atomic_number < rhs.atomic_number;
        }
    }.lessThan;
}

test "parse reads BSE JSON string floats into numeric shells" {
    const text =
        \\{
        \\  "name": "STO-3G",
        \\  "elements": {
        \\    "1": {
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [["0.1543289673E+00", "0.5353281423E+00", "0.4446345422E+00"]],
        \\        "exponents": ["0.3425250914E+01", "0.6239137298E+00", "0.1688554040E+00"],
        \\        "function_type": "gto",
        \\        "region": ""
        \\      }]
        \\    }
        \\  }
        \\}
    ;

    var basis = try parse(f64, std.testing.allocator, text);
    defer basis.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("STO-3G", basis.name);
    const hydrogen = basis.element(1).?;
    try std.testing.expectEqual(@as(usize, 1), hydrogen.shells.len);
    try std.testing.expectEqual(@as(AngularMomentumOrder, 0), hydrogen.shells[0].angular_momentum);
    try std.testing.expectEqual(gaussian_orbital.ShellKind.cartesian, hydrogen.shells[0].kind);
    try std.testing.expectApproxEqAbs(
        @as(f64, 3.425250914),
        hydrogen.shells[0].primitives[0].exponent,
        1e-12,
    );

    const shells = try basis.shellsForAtom(std.testing.allocator, 1, .{ 0.0, 0.0, 0.0 });
    defer std.testing.allocator.free(shells);

    try std.testing.expectEqual(@as(usize, 1), shells.len);
    try std.testing.expectEqual(@as(AngularMomentumOrder, 0), shells[0].angular_momentum);
    try std.testing.expectEqual(gaussian_orbital.ShellKind.cartesian, shells[0].kind);
    try std.testing.expectEqual(@as(usize, 3), shells[0].primitives.len);
}

test "readFile expands 6-31G carbon SP shells into separate s and p shells" {
    var basis = readFile(
        f64,
        std.testing.io,
        std.testing.allocator,
        "data/basis/6-31g.json",
    ) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer basis.deinit(std.testing.allocator);

    const shells = try basis.shellsForAtom(std.testing.allocator, 6, .{ 0.0, 0.0, 0.0 });
    defer std.testing.allocator.free(shells);

    // Carbon in 6-31G, in PySCF AO order: 1s (6 prims), 2s (3 prims),
    // 2s' (1 prim), 2p (3 prims), 2p' (1 prim).
    try std.testing.expectEqual(@as(usize, 5), shells.len);
    try std.testing.expectEqual(@as(AngularMomentumOrder, 0), shells[0].angular_momentum);
    try std.testing.expectEqual(@as(usize, 6), shells[0].primitives.len);
    try std.testing.expectEqual(@as(AngularMomentumOrder, 0), shells[1].angular_momentum);
    try std.testing.expectEqual(@as(usize, 3), shells[1].primitives.len);
    try std.testing.expectEqual(@as(AngularMomentumOrder, 0), shells[2].angular_momentum);
    try std.testing.expectEqual(@as(usize, 1), shells[2].primitives.len);
    try std.testing.expectEqual(@as(AngularMomentumOrder, 1), shells[3].angular_momentum);
    try std.testing.expectEqual(@as(usize, 3), shells[3].primitives.len);
    try std.testing.expectEqual(@as(AngularMomentumOrder, 1), shells[4].angular_momentum);
    try std.testing.expectEqual(@as(usize, 1), shells[4].primitives.len);

    var total_cartesian_components: usize = 0;
    for (shells) |shell| {
        total_cartesian_components += gaussian_orbital.shellComponentCount(
            shell.kind,
            shell.angular_momentum,
        );
    }
    try std.testing.expectEqual(@as(usize, 9), total_cartesian_components);
}

test "shellsForAtom rejects missing basis elements" {
    const text =
        \\{
        \\  "name": "hydrogen-only",
        \\  "elements": {
        \\    "1": {
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [["1.0"]],
        \\        "exponents": ["1.0"],
        \\        "function_type": "gto"
        \\      }]
        \\    }
        \\  }
        \\}
    ;

    var basis_set = try parse(f64, std.testing.allocator, text);
    defer basis_set.deinit(std.testing.allocator);

    try std.testing.expectError(
        error.ElementNotFound,
        basis_set.shellsForAtom(std.testing.allocator, 8, .{ 0.0, 0.0, 0.0 }),
    );
}

test "parse rejects malformed basis shell shapes" {
    const text =
        \\{
        \\  "name": "bad-shell",
        \\  "elements": {
        \\    "1": {
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [["1.0", "0.5"]],
        \\        "exponents": ["1.0"],
        \\        "function_type": "gto"
        \\      }]
        \\    }
        \\  }
        \\}
    ;

    try std.testing.expectError(
        error.InvalidBasisShell,
        parse(f64, std.testing.allocator, text),
    );
}

test "parse rejects shells with zero primitives" {
    const text =
        \\{
        \\  "name": "empty-shell",
        \\  "elements": {
        \\    "1": {
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [[]],
        \\        "exponents": [],
        \\        "function_type": "gto"
        \\      }]
        \\    }
        \\  }
        \\}
    ;

    try std.testing.expectError(
        error.InvalidBasisShell,
        parse(f64, std.testing.allocator, text),
    );
}

test "parse loads non-ECP elements and rejects requested ECP elements" {
    const text =
        \\{
        \\  "name": "mixed-ecp-basis",
        \\  "elements": {
        \\    "1": {
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [["1.0"]],
        \\        "exponents": ["1.0"],
        \\        "function_type": "gto"
        \\      }]
        \\    },
        \\    "26": {
        \\      "ecp_electrons": 10,
        \\      "ecp_potentials": [],
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [["1.0"]],
        \\        "exponents": ["1.0"],
        \\        "function_type": "gto"
        \\      }]
        \\    }
        \\  }
        \\}
    ;

    var basis_set = try parse(f64, std.testing.allocator, text);
    defer basis_set.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), basis_set.elements.len);
    try std.testing.expectEqual(@as(usize, 1), basis_set.ecp_elements.len);
    try std.testing.expect(basis_set.elementHasEcp(26));

    const hydrogen = try basis_set.shellsForAtom(std.testing.allocator, 1, .{ 0.0, 0.0, 0.0 });
    defer std.testing.allocator.free(hydrogen);
    try std.testing.expectEqual(@as(usize, 1), hydrogen.len);

    try std.testing.expectError(
        error.UnsupportedEcp,
        basis_set.shellsForAtom(std.testing.allocator, 26, .{ 0.0, 0.0, 0.0 }),
    );
}

test "readFile loads def2 light elements while rejecting def2 ECP elements" {
    var basis_set = readFile(
        f64,
        std.testing.io,
        std.testing.allocator,
        "data/basis/def2-svp.json",
    ) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer basis_set.deinit(std.testing.allocator);

    const oxygen = try basis_set.shellsForAtom(std.testing.allocator, 8, .{ 0.0, 0.0, 0.0 });
    defer std.testing.allocator.free(oxygen);
    try std.testing.expect(oxygen.len > 0);

    try std.testing.expectError(
        error.UnsupportedEcp,
        basis_set.shellsForAtom(std.testing.allocator, 37, .{ 0.0, 0.0, 0.0 }),
    );
}

test "parse maps malformed JSON to InvalidBasisJson" {
    try std.testing.expectError(
        error.InvalidBasisJson,
        parse(f64, std.testing.allocator, "{\"name\":\"bad\",\"elements\":"),
    );
}

test "parse maps invalid element keys to InvalidBasisJson" {
    const text =
        \\{
        \\  "name": "bad-element-key",
        \\  "elements": {
        \\    "H": {
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [["1.0"]],
        \\        "exponents": ["1.0"],
        \\        "function_type": "gto"
        \\      }]
        \\    }
        \\  }
        \\}
    ;

    try std.testing.expectError(
        error.InvalidBasisJson,
        parse(f64, std.testing.allocator, text),
    );
}

test "parse rejects shells above the supported angular momentum limit" {
    const unsupported_angular_momentum = gaussian_orbital.max_angular_momentum + 1;
    const text = try std.fmt.allocPrint(
        std.testing.allocator,
        \\{{
        \\  "name": "too-large",
        \\  "elements": {{
        \\    "1": {{
        \\      "electron_shells": [{{
        \\        "angular_momentum": [{d}],
        \\        "coefficients": [["1.0"]],
        \\        "exponents": ["1.0"],
        \\        "function_type": "gto_cartesian",
        \\        "region": ""
        \\      }}]
        \\    }}
        \\  }}
        \\}}
    ,
        .{unsupported_angular_momentum},
    );
    defer std.testing.allocator.free(text);

    try std.testing.expectError(
        error.AngularMomentumTooLarge,
        parse(f64, std.testing.allocator, text),
    );
}

test "parse rejects non-positive and non-finite primitive data" {
    const negative_exponent =
        \\{
        \\  "name": "bad-exponent",
        \\  "elements": {
        \\    "1": {
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [["1.0"]],
        \\        "exponents": ["-1.0"],
        \\        "function_type": "gto"
        \\      }]
        \\    }
        \\  }
        \\}
    ;

    try std.testing.expectError(
        error.InvalidBasisShell,
        parse(f64, std.testing.allocator, negative_exponent),
    );

    const zero_contraction =
        \\{
        \\  "name": "zero-contraction",
        \\  "elements": {
        \\    "1": {
        \\      "electron_shells": [{
        \\        "angular_momentum": [0],
        \\        "coefficients": [["0.0"]],
        \\        "exponents": ["1.0"],
        \\        "function_type": "gto"
        \\      }]
        \\    }
        \\  }
        \\}
    ;

    try std.testing.expectError(
        error.InvalidBasisShell,
        parse(f64, std.testing.allocator, zero_contraction),
    );
}
