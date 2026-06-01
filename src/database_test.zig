//! Opt-in basis database corpus tests.

const std = @import("std");

const basis = @import("basis.zig");
const gaussian_orbital = @import("gaussian_orbital.zig");
const math = @import("math.zig");
const common = @import("integrals/common.zig");

const basis_index_path = "data/basis/_index.json";
const basis_dir = "data/basis";
const normalization_tolerance: f64 = 1e-7;

const BasisIndex = std.json.ArrayHashMap([]const u8);
const ParsedBasisIndex = std.json.Parsed(BasisIndex);

const BasisAnalysis = struct {
    has_ecp: bool = false,
    first_ecp_atomic_number: ?basis.AtomicNumber = null,
    has_electron_shells: bool = false,
    max_angular_momentum: gaussian_orbital.AngularMomentumOrder = 0,
    max_primitives: usize = 0,
};

fn readBasisIndex() !ParsedBasisIndex {
    const json_text = try std.Io.Dir.cwd().readFileAlloc(
        std.testing.io,
        basis_index_path,
        std.testing.allocator,
        .unlimited,
    );
    defer std.testing.allocator.free(json_text);

    return std.json.parseFromSlice(
        BasisIndex,
        std.testing.allocator,
        json_text,
        .{ .allocate = .alloc_always },
    );
}

fn jsonUnsigned(value: std.json.Value) !usize {
    return switch (value) {
        .integer => |integer| blk: {
            if (integer < 0) return error.InvalidBasisJson;
            break :blk @intCast(integer);
        },
        .number_string, .string => |text| std.fmt.parseInt(usize, text, 10),
        else => error.InvalidBasisJson,
    };
}

fn analyzeBasisJson(json_text: []const u8) !BasisAnalysis {
    var parsed = try std.json.parseFromSlice(
        std.json.Value,
        std.testing.allocator,
        json_text,
        .{ .allocate = .alloc_always },
    );
    defer parsed.deinit();

    const root = switch (parsed.value) {
        .object => |object| object,
        else => return error.InvalidBasisJson,
    };
    const elements_value = root.get("elements") orelse return error.InvalidBasisJson;
    const elements = switch (elements_value) {
        .object => |object| object,
        else => return error.InvalidBasisJson,
    };

    var analysis: BasisAnalysis = .{};
    var element_it = elements.iterator();
    while (element_it.next()) |element_entry| {
        const atomic_number = std.fmt.parseInt(basis.AtomicNumber, element_entry.key_ptr.*, 10) catch return error.InvalidBasisJson;
        const element = switch (element_entry.value_ptr.*) {
            .object => |object| object,
            else => return error.InvalidBasisJson,
        };

        if (element.get("ecp_electrons") != null or element.get("ecp_potentials") != null) {
            analysis.has_ecp = true;
            if (analysis.first_ecp_atomic_number == null) {
                analysis.first_ecp_atomic_number = atomic_number;
            }
        }

        const shells_value = element.get("electron_shells") orelse continue;
        const shells = switch (shells_value) {
            .array => |array| array,
            else => return error.InvalidBasisJson,
        };
        if (shells.items.len > 0) analysis.has_electron_shells = true;

        for (shells.items) |shell_value| {
            const shell = switch (shell_value) {
                .object => |object| object,
                else => return error.InvalidBasisJson,
            };

            const angular_momenta_value = shell.get("angular_momentum") orelse return error.InvalidBasisJson;
            const angular_momenta = switch (angular_momenta_value) {
                .array => |array| array,
                else => return error.InvalidBasisJson,
            };
            for (angular_momenta.items) |angular_momentum_value| {
                const angular_momentum = try jsonUnsigned(angular_momentum_value);
                if (angular_momentum > std.math.maxInt(gaussian_orbital.AngularMomentumOrder)) return error.InvalidBasisJson;
                analysis.max_angular_momentum = @max(
                    analysis.max_angular_momentum,
                    @as(gaussian_orbital.AngularMomentumOrder, @intCast(angular_momentum)),
                );
            }

            const exponents_value = shell.get("exponents") orelse return error.InvalidBasisJson;
            const exponents = switch (exponents_value) {
                .array => |array| array,
                else => return error.InvalidBasisJson,
            };
            analysis.max_primitives = @max(analysis.max_primitives, exponents.items.len);
        }
    }

    return analysis;
}

fn contractionNormSquared(shell: basis.ContractedShell(f64)) f64 {
    var norm_squared: f64 = 0;
    const two_to_l = math.powUnsigned(f64, 2, shell.angular_momentum);

    for (shell.primitives) |lhs| {
        for (shell.primitives) |rhs| {
            const combined_exponent = lhs.exponent + rhs.exponent;
            const p_to_l = math.powUnsigned(f64, combined_exponent, shell.angular_momentum);
            const radial_overlap =
                (@as(f64, std.math.pi) * @sqrt(@as(f64, std.math.pi))) /
                (two_to_l * p_to_l * combined_exponent * @sqrt(combined_exponent));
            norm_squared += lhs.coefficient * rhs.coefficient * radial_overlap;
        }
    }

    return norm_squared;
}

fn expectShellNormalized(
    file_name: []const u8,
    atomic_number: basis.AtomicNumber,
    shell_index: usize,
    shell: basis.ContractedShell(f64),
) !void {
    const actual = contractionNormSquared(shell);
    if (@abs(actual - 1) > normalization_tolerance) {
        std.debug.print(
            "{s} Z={d} shell={d} L={d} {s} contraction norm expected 1, actual {e}\n",
            .{
                file_name,
                atomic_number,
                shell_index,
                shell.angular_momentum,
                @tagName(shell.kind),
                actual,
            },
        );
        return error.TestExpectedApproxEqAbs;
    }
}

fn expectBasisShellsNormalized(file_name: []const u8, basis_set: basis.BasisSet(f64)) !void {
    var checked_count: usize = 0;

    for (basis_set.elements) |element_basis| {
        for (element_basis.shells, 0..) |contracted_shell, shell_index| {
            checked_count += 1;

            try expectShellNormalized(
                file_name,
                element_basis.atomic_number,
                shell_index,
                contracted_shell,
            );
        }
    }

    if (checked_count == 0) {
        std.debug.print("supported basis has no shells to normalize: {s}\n", .{file_name});
        return error.TestUnexpectedResult;
    }
}

fn expectEcpElementUnsupported(
    file_name: []const u8,
    basis_set: basis.BasisSet(f64),
    atomic_number: basis.AtomicNumber,
) !void {
    try std.testing.expect(basis_set.elementHasEcp(atomic_number));
    try std.testing.expectError(
        error.UnsupportedEcp,
        basis_set.shellsForAtom(std.testing.allocator, atomic_number, .{ 0.0, 0.0, 0.0 }),
    );
    if (basis_set.element(atomic_number) != null) {
        std.debug.print("ECP element unexpectedly converted: {s} Z={d}\n", .{ file_name, atomic_number });
        return error.TestUnexpectedResult;
    }
}

test "basis database index loads supported non-ECP basis files" {
    var parsed_index = try readBasisIndex();
    defer parsed_index.deinit();

    var supported_count: usize = 0;
    var ecp_count: usize = 0;
    var no_electron_shell_count: usize = 0;
    var high_l_count: usize = 0;
    var too_many_primitives_count: usize = 0;

    var it = parsed_index.value.map.iterator();
    while (it.next()) |entry| {
        const file_name = entry.value_ptr.*;
        const path = try std.fmt.allocPrint(
            std.testing.allocator,
            "{s}/{s}",
            .{ basis_dir, file_name },
        );
        defer std.testing.allocator.free(path);

        const json_text = try std.Io.Dir.cwd().readFileAlloc(
            std.testing.io,
            path,
            std.testing.allocator,
            .unlimited,
        );
        defer std.testing.allocator.free(json_text);

        const analysis = try analyzeBasisJson(json_text);
        if (analysis.has_ecp) {
            ecp_count += 1;
        }
        if (!analysis.has_electron_shells) {
            var basis_set = basis.parse(f64, std.testing.allocator, json_text) catch |err| {
                std.debug.print("ECP-only basis failed to load: {s}: {s}\n", .{ file_name, @errorName(err) });
                return err;
            };
            defer basis_set.deinit(std.testing.allocator);
            if (analysis.first_ecp_atomic_number) |atomic_number| {
                try expectEcpElementUnsupported(file_name, basis_set, atomic_number);
            }
            no_electron_shell_count += 1;
            continue;
        }
        if (analysis.max_angular_momentum > gaussian_orbital.max_angular_momentum) {
            high_l_count += 1;
            if (!std.mem.eql(u8, file_name, "cc-pv9z.json")) {
                std.debug.print(
                    "unexpected L>{d} basis in database: {s} has L={d}\n",
                    .{ gaussian_orbital.max_angular_momentum, file_name, analysis.max_angular_momentum },
                );
                return error.TestUnexpectedResult;
            }
            continue;
        }
        if (analysis.max_primitives > common.max_primitives_per_shell) {
            too_many_primitives_count += 1;
            continue;
        }

        var basis_set = basis.parse(f64, std.testing.allocator, json_text) catch |err| {
            std.debug.print("supported basis failed to load: {s}: {s}\n", .{ file_name, @errorName(err) });
            return err;
        };
        defer basis_set.deinit(std.testing.allocator);

        if (analysis.first_ecp_atomic_number) |atomic_number| {
            try expectEcpElementUnsupported(file_name, basis_set, atomic_number);
        }
        if (basis_set.elements.len == 0) {
            no_electron_shell_count += 1;
            continue;
        }
        try expectBasisShellsNormalized(file_name, basis_set);
        supported_count += 1;
    }

    try std.testing.expect(supported_count > 0);
    try std.testing.expect(ecp_count > 0 or no_electron_shell_count > 0);
    try std.testing.expectEqual(@as(usize, 1), high_l_count);
    try std.testing.expectEqual(@as(usize, 0), too_many_primitives_count);
}
