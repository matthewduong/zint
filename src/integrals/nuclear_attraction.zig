//! Gaussian nuclear-attraction integrals.

const std = @import("std");
const scalar = @import("../scalar.zig");
const matrix = @import("../matrix.zig");
const gaussian_orbital = @import("../gaussian_orbital.zig");
const common = @import("common.zig");
const hermite = @import("hermite.zig");
const types = @import("../types.zig");

pub const ValidateNucleiError = error{InvalidNucleus};
pub const NuclearAttractionMatrixError = common.OneElectronMatrixError || ValidateNucleiError;
pub const ShellPairNuclearAttractionError =
    common.ShellPairBlockError || std.mem.Allocator.Error || ValidateNucleiError;

/// Nuclear charge and center used by the electron-nucleus attraction kernel.
pub fn Nucleus(comptime T: type) type {
    scalar.assertFloat(T);
    return struct {
        charge: T,
        center: [3]T,
    };
}

/// Build the full symmetric AO nuclear-attraction matrix in allocator-owned
/// packed storage. Validates shells and nuclei, reuses Coulomb scratch across
/// shell pairs, and applies any Cartesian-to-AO transform during assembly.
pub fn nuclearAttractionMatrix(
    comptime T: type,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
    nuclei: []const Nucleus(T),
) NuclearAttractionMatrixError!matrix.SymmetricMatrix(T) {
    scalar.assertFloat(T);
    try common.validateShells(T, shells);
    try validateNuclei(T, nuclei);

    // Size the CoulombTable for the basis worst case so the inner kernel
    // can `reset()` it once per (primitive pair × nucleus) instead of
    // re-allocating per shell pair.
    const max_l = common.maxAngularMomentum(T, shells);
    var coulomb_table = try makeCoulombTable(T, allocator, max_l);
    defer coulomb_table.deinit(allocator);
    const weighted_coulomb = try allocator.alloc(T, weightedCoulombScratchEntryCount(max_l));
    defer allocator.free(weighted_coulomb);

    return common.oneElectronMatrixAssumeValid(T, computeNuclearBlock, allocator, shells, .{
        .nuclei = nuclei,
        .coulomb_table = &coulomb_table,
        .weighted_coulomb = weighted_coulomb,
    });
}

/// Write the AO-space shell-pair nuclear-attraction block for `lhs_shell` and
/// `rhs_shell`. `out` is row-major with shape `(lhs_ao x rhs_ao)`.
pub fn shellPairNuclearAttraction(
    comptime T: type,
    allocator: std.mem.Allocator,
    lhs_shell: gaussian_orbital.Shell(T),
    rhs_shell: gaussian_orbital.Shell(T),
    nuclei: []const Nucleus(T),
    out: []T,
) ShellPairNuclearAttractionError!void {
    scalar.assertFloat(T);
    try common.validateShell(T, lhs_shell);
    try common.validateShell(T, rhs_shell);
    try validateNuclei(T, nuclei);

    const max_l = @max(lhs_shell.angular_momentum, rhs_shell.angular_momentum);
    var coulomb_table = try makeCoulombTable(T, allocator, max_l);
    defer coulomb_table.deinit(allocator);
    const weighted_coulomb = try allocator.alloc(T, weightedCoulombScratchEntryCount(max_l));
    defer allocator.free(weighted_coulomb);

    return common.shellPairBlockAssumeValid(T, computeNuclearBlock, lhs_shell, rhs_shell, .{
        .nuclei = nuclei,
        .coulomb_table = &coulomb_table,
        .weighted_coulomb = weighted_coulomb,
    }, out);
}

/// Allocate a Coulomb auxiliary table sized for any shell pair up to `max_l`.
fn makeCoulombTable(
    comptime T: type,
    allocator: std.mem.Allocator,
    max_l: types.AngularMomentumOrder,
) std.mem.Allocator.Error!hermite.CoulombTable(T) {
    // For a single shell pair: x = lhs.x + rhs.x ≤ 2·max_l per axis,
    // n = max_x + max_y + max_z ≤ 6·max_l.
    const max_axis: usize = 2 * @as(usize, max_l);
    return hermite.CoulombTable(T).init(allocator, max_axis, max_axis, max_axis, 3 * max_axis);
}

/// Count scratch entries for nucleus-weighted R(0, x, y, z) values.
fn weightedCoulombScratchEntryCount(max_l: types.AngularMomentumOrder) usize {
    const axis_count = 2 * @as(usize, max_l) + 1;
    return axis_count * axis_count * axis_count;
}

/// Reject nuclei with non-finite charges or centers before kernel entry.
fn validateNuclei(comptime T: type, nuclei: []const Nucleus(T)) ValidateNucleiError!void {
    for (nuclei) |nucleus| {
        if (!std.math.isFinite(nucleus.charge)) return error.InvalidNucleus;
        for (nucleus.center) |coordinate| {
            if (!std.math.isFinite(coordinate)) return error.InvalidNucleus;
        }
    }
}

/// Prepare Cartesian shell data and compute one Cartesian nuclear block.
fn computeNuclearBlock(
    comptime T: type,
    lhs: gaussian_orbital.Shell(T),
    rhs: gaussian_orbital.Shell(T),
    extra: anytype,
    out: []T,
) common.ValidateShellError!void {
    const nuclei = extra.nuclei;
    const coulomb_table = extra.coulomb_table;
    const weighted_coulomb_scratch = extra.weighted_coulomb;
    scalar.assertFloat(T);

    var lhs_workspace: common.PreparedShellWorkspace(T) = undefined;
    var rhs_workspace: common.PreparedShellWorkspace(T) = undefined;
    const lhs_data = try lhs_workspace.prepare(lhs);
    const rhs_data = try rhs_workspace.prepare(rhs);

    std.debug.assert(out.len == lhs_data.components.len * rhs_data.components.len);

    @memset(out, 0);
    if (nuclei.len == 0) return;

    const coulomb_extent = shellPairCoulombExtent(T, lhs_data, rhs_data);
    const weighted_coulomb_len = weightedCoulombEntryCount(coulomb_extent);
    const weighted_coulomb = weighted_coulomb_scratch[0..weighted_coulomb_len];

    accumulateNuclearBlock(
        T,
        lhs_data,
        rhs_data,
        nuclei,
        coulomb_table,
        coulomb_extent,
        weighted_coulomb,
        out,
    );
}

/// Maximum Hermite Coulomb orders needed by one prepared shell pair.
fn shellPairCoulombExtent(
    comptime T: type,
    lhs: common.PreparedShell(T),
    rhs: common.PreparedShell(T),
) hermite.CoulombExtent {
    return .{
        .x_max = @as(usize, lhs.max_per_axis[0]) + rhs.max_per_axis[0],
        .y_max = @as(usize, lhs.max_per_axis[1]) + rhs.max_per_axis[1],
        .z_max = @as(usize, lhs.max_per_axis[2]) + rhs.max_per_axis[2],
    };
}

/// Loop over primitive pairs, fill Hermite expansion tables, build the
/// nucleus-weighted Coulomb entries, and accumulate contracted contributions.
fn accumulateNuclearBlock(
    comptime T: type,
    lhs: common.PreparedShell(T),
    rhs: common.PreparedShell(T),
    nuclei: []const Nucleus(T),
    coulomb_table: *hermite.CoulombTable(T),
    coulomb_extent: hermite.CoulombExtent,
    weighted_coulomb: []T,
    out: []T,
) void {
    var axis_tables: hermite.HermiteAxisTables(T) = undefined;

    for (lhs.shell.primitives, 0..) |lhs_primitive, lhs_primitive_index| {
        for (rhs.shell.primitives, 0..) |rhs_primitive, rhs_primitive_index| {
            const pair = common.buildPrimitivePair(
                T,
                lhs_primitive.exponent,
                lhs.shell.center,
                rhs_primitive.exponent,
                rhs.shell.center,
            );
            axis_tables.fill(pair, lhs.shell.center, rhs.shell.center, lhs.max_per_axis, rhs.max_per_axis, .{});
            fillPrimitivePairWeightedCoulomb(
                T,
                pair,
                nuclei,
                coulomb_table,
                coulomb_extent,
                weighted_coulomb,
            );
            accumulatePrimitivePairNuclear(
                T,
                lhs,
                rhs,
                lhs_primitive_index,
                rhs_primitive_index,
                axis_tables.views(),
                coulomb_extent,
                weighted_coulomb,
                out,
            );
        }
    }
}

/// Sum all nuclei into a primitive-pair Coulomb scratch table with the
/// electron-nucleus prefactor and charge sign already applied.
fn fillPrimitivePairWeightedCoulomb(
    comptime T: type,
    pair: common.PrimitivePair(T),
    nuclei: []const Nucleus(T),
    coulomb_table: *hermite.CoulombTable(T),
    coulomb_extent: hermite.CoulombExtent,
    weighted_coulomb: []T,
) void {
    @memset(weighted_coulomb, 0);

    const pair_prefactor = @as(T, 2) * @as(T, std.math.pi) / pair.combined_exponent;
    for (nuclei) |nucleus| {
        coulomb_table.reset(pair.combined_exponent, .{
            pair.product_center[0] - nucleus.center[0],
            pair.product_center[1] - nucleus.center[1],
            pair.product_center[2] - nucleus.center[2],
        }, coulomb_extent);

        accumulateWeightedCoulomb(
            T,
            coulomb_table,
            coulomb_extent,
            -nucleus.charge * pair_prefactor,
            weighted_coulomb,
        );
    }
}

/// Accumulate one primitive pair using normalized component coefficients and
/// the preweighted Coulomb auxiliary table.
fn accumulatePrimitivePairNuclear(
    comptime T: type,
    lhs: common.PreparedShell(T),
    rhs: common.PreparedShell(T),
    lhs_primitive_index: usize,
    rhs_primitive_index: usize,
    axis_tables: anytype,
    coulomb_extent: hermite.CoulombExtent,
    weighted_coulomb: []const T,
    out: []T,
) void {
    for (lhs.components, 0..) |lhs_component, lhs_component_index| {
        const lhs_coefficient = lhs.coefficients.at(lhs_component_index, lhs_primitive_index);
        for (rhs.components, 0..) |rhs_component, rhs_component_index| {
            const rhs_coefficient = rhs.coefficients.at(rhs_component_index, rhs_primitive_index);

            out[
                common.rowMajorIndex(lhs_component_index, rhs_component_index, rhs.components.len)
            ] += lhs_coefficient * rhs_coefficient * primitiveNuclear(
                T,
                lhs_component,
                rhs_component,
                axis_tables,
                coulomb_extent,
                weighted_coulomb,
            );
        }
    }
}

/// Count the packed-by-extents R(0, x, y, z) scratch entries.
fn weightedCoulombEntryCount(extent: hermite.CoulombExtent) usize {
    return (extent.x_max + 1) * (extent.y_max + 1) * (extent.z_max + 1);
}

/// Row-major index into the `(x, y, z)` weighted Coulomb scratch cube.
fn weightedCoulombIndex(extent: hermite.CoulombExtent, x: usize, y: usize, z: usize) usize {
    return ((x * (extent.y_max + 1)) + y) * (extent.z_max + 1) + z;
}

/// Add one weighted Coulomb table's R(0, x, y, z) entries into scratch.
fn accumulateWeightedCoulomb(
    comptime T: type,
    coulomb_table: *hermite.CoulombTable(T),
    extent: hermite.CoulombExtent,
    weight: T,
    out: []T,
) void {
    std.debug.assert(out.len >= weightedCoulombEntryCount(extent));

    for (0..extent.x_max + 1) |x| {
        for (0..extent.y_max + 1) |y| {
            for (0..extent.z_max + 1) |z| {
                out[weightedCoulombIndex(extent, x, y, z)] += weight *
                    coulomb_table.get(0, x, y, z);
            }
        }
    }
}

/// Contract Hermite expansion coefficients with weighted Coulomb entries for
/// one Cartesian primitive component pair.
fn primitiveNuclear(
    comptime T: type,
    lhs: gaussian_orbital.AngularMomentum,
    rhs: gaussian_orbital.AngularMomentum,
    axis_tables: anytype,
    coulomb_extent: hermite.CoulombExtent,
    weighted_coulomb: []const T,
) T {
    var hermite_coulomb_sum: T = 0;
    const x_hermite_expansion_table = axis_tables[0];
    const y_hermite_expansion_table = axis_tables[1];
    const z_hermite_expansion_table = axis_tables[2];

    for (0..@as(usize, lhs.x) + rhs.x + 1) |x_hermite_order| {
        const x_axis_entry = x_hermite_expansion_table.get(lhs.x, rhs.x, x_hermite_order);
        if (x_axis_entry == 0) continue;
        for (0..@as(usize, lhs.y) + rhs.y + 1) |y_hermite_order| {
            const y_axis_entry = y_hermite_expansion_table.get(lhs.y, rhs.y, y_hermite_order);
            if (y_axis_entry == 0) continue;
            for (0..@as(usize, lhs.z) + rhs.z + 1) |z_hermite_order| {
                const z_axis_entry = z_hermite_expansion_table.get(lhs.z, rhs.z, z_hermite_order);
                if (z_axis_entry == 0) continue;
                hermite_coulomb_sum += x_axis_entry * y_axis_entry * z_axis_entry *
                    weighted_coulomb[
                        weightedCoulombIndex(
                            coulomb_extent,
                            x_hermite_order,
                            y_hermite_order,
                            z_hermite_order,
                        )
                    ];
            }
        }
    }
    return hermite_coulomb_sum;
}

test "single normalized s primitive attraction to same-center unit nucleus" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const exponent: f64 = 0.7;
    const primitives = [_]Primitive{.{
        .exponent = exponent,
        .coefficient = gaussian_orbital.radialNormalization(f64, exponent, 0),
    }};
    const shell = Shell{
        .center = .{ 0, 0, 0 },
        .angular_momentum = 0,
        .primitives = primitives[0..],
    };
    const nuclei = [_]Nucleus(f64){.{ .charge = 1, .center = .{ 0, 0, 0 } }};

    var block: [1]f64 = undefined;
    try shellPairNuclearAttraction(f64, std.testing.allocator, shell, shell, &nuclei, &block);

    const expected = -@as(f64, 2) * @sqrt(@as(f64, 2) * exponent / @as(f64, std.math.pi));
    try std.testing.expectApproxEqAbs(expected, block[0], 1e-12);
}

test "nuclear attraction rejects non-finite nuclei" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const primitives = [_]Primitive{.{
        .exponent = 1.0,
        .coefficient = 1.0,
    }};
    const shell = Shell{
        .center = .{ 0, 0, 0 },
        .angular_momentum = 0,
        .primitives = primitives[0..],
    };
    const bad_charge = [_]Nucleus(f64){.{ .charge = std.math.inf(f64), .center = .{ 0, 0, 0 } }};
    const bad_center = [_]Nucleus(f64){.{ .charge = 1, .center = .{ 0, std.math.nan(f64), 0 } }};

    var block: [1]f64 = undefined;
    try std.testing.expectError(
        error.InvalidNucleus,
        shellPairNuclearAttraction(f64, std.testing.allocator, shell, shell, &bad_charge, &block),
    );
    try std.testing.expectError(
        error.InvalidNucleus,
        nuclearAttractionMatrix(f64, std.testing.allocator, &[_]Shell{shell}, &bad_center),
    );
}

test "shellPairNuclearAttraction validates shells before scratch allocation" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const primitives = [_]Primitive{.{
        .exponent = 1.0,
        .coefficient = 1.0,
    }};
    const shell = Shell{
        .center = .{ 0, 0, 0 },
        .angular_momentum = std.math.maxInt(types.AngularMomentumOrder),
        .primitives = primitives[0..],
    };
    const nuclei = [_]Nucleus(f64){.{ .charge = 1, .center = .{ 0, 0, 0 } }};

    var block: [1]f64 = undefined;
    try std.testing.expectError(
        error.AngularMomentumTooLarge,
        shellPairNuclearAttraction(f64, std.testing.failing_allocator, shell, shell, &nuclei, &block),
    );
}
