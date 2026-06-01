//! Gaussian electron-repulsion integrals.

const std = @import("std");
const scalar = @import("../scalar.zig");
const matrix = @import("../matrix.zig");
const gaussian_orbital = @import("../gaussian_orbital.zig");
const common = @import("common.zig");
const hermite = @import("hermite.zig");
const types = @import("../types.zig");

pub const ElectronRepulsionTensorInitError = std.mem.Allocator.Error || error{DimensionTooLarge};
pub const ElectronRepulsionTensorError = common.ValidateShellError || ElectronRepulsionTensorInitError;

/// Error set for streaming ERI blocks through a caller-provided callback.
pub fn ElectronRepulsionBlocksError(comptime CallbackError: type) type {
    return ElectronRepulsionTensorError || CallbackError;
}

/// Extract the callback error set used by `electronRepulsionBlocks`.
fn CallbackReturnError(comptime callback: anytype) type {
    const callback_fn = @typeInfo(@TypeOf(callback)).@"fn";
    const return_type = callback_fn.return_type orelse @compileError("callback must return an error union");
    return @typeInfo(return_type).error_union.error_set;
}

/// Packed AO ERI tensor using pair-pair symmetry for `(ij|kl)`.
pub fn ElectronRepulsionTensor(comptime T: type) type {
    scalar.assertFloat(T);
    return struct {
        dimension: usize,
        pair_count: usize,
        values: []T,

        const Self = @This();

        /// Allocate packed storage for an AO dimension.
        pub fn init(allocator: std.mem.Allocator, dimension: usize) ElectronRepulsionTensorInitError!Self {
            const counts = try electronRepulsionTensorEntryCounts(dimension);
            return .{
                .dimension = dimension,
                .pair_count = counts.pair_count,
                .values = try allocator.alloc(T, counts.value_count),
            };
        }

        /// Release packed ERI storage.
        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            allocator.free(self.values);
            self.* = undefined;
        }

        /// Read one AO ERI entry, applying the tensor's packed symmetries.
        pub fn get(
            self: *const Self,
            first_ao_index: usize,
            second_ao_index: usize,
            third_ao_index: usize,
            fourth_ao_index: usize,
        ) T {
            std.debug.assert(first_ao_index < self.dimension);
            std.debug.assert(second_ao_index < self.dimension);
            std.debug.assert(third_ao_index < self.dimension);
            std.debug.assert(fourth_ao_index < self.dimension);
            return self.values[
                self.valueIndex(
                    first_ao_index,
                    second_ao_index,
                    third_ao_index,
                    fourth_ao_index,
                )
            ];
        }

        /// Write one AO ERI entry, applying the tensor's packed symmetries.
        pub fn set(
            self: *Self,
            first_ao_index: usize,
            second_ao_index: usize,
            third_ao_index: usize,
            fourth_ao_index: usize,
            value: T,
        ) void {
            std.debug.assert(first_ao_index < self.dimension);
            std.debug.assert(second_ao_index < self.dimension);
            std.debug.assert(third_ao_index < self.dimension);
            std.debug.assert(fourth_ao_index < self.dimension);
            self.values[
                self.valueIndex(
                    first_ao_index,
                    second_ao_index,
                    third_ao_index,
                    fourth_ao_index,
                )
            ] = value;
        }

        /// Packed pair-pair offset for one four-index AO entry.
        fn valueIndex(
            self: Self,
            first_ao_index: usize,
            second_ao_index: usize,
            third_ao_index: usize,
            fourth_ao_index: usize,
        ) usize {
            const first_pair_index = triangleIndex(first_ao_index, second_ao_index);
            const second_pair_index = triangleIndex(third_ao_index, fourth_ao_index);
            std.debug.assert(first_pair_index < self.pair_count);
            std.debug.assert(second_pair_index < self.pair_count);
            return triangleIndex(first_pair_index, second_pair_index);
        }
    };
}

const TensorEntryCounts = struct {
    pair_count: usize,
    value_count: usize,
};

/// Count AO-pair and pair-pair packed entries for a materialized ERI tensor.
fn electronRepulsionTensorEntryCounts(dimension: usize) error{DimensionTooLarge}!TensorEntryCounts {
    const pair_count = try matrix.packedSymmetricEntryCount(dimension);
    const value_count = try matrix.packedSymmetricEntryCount(pair_count);
    return .{
        .pair_count = pair_count,
        .value_count = value_count,
    };
}

test "electron repulsion tensor sizing checks packed overflow" {
    const counts = try electronRepulsionTensorEntryCounts(2);
    try std.testing.expectEqual(@as(usize, 3), counts.pair_count);
    try std.testing.expectEqual(@as(usize, 6), counts.value_count);

    try std.testing.expectError(
        error.DimensionTooLarge,
        electronRepulsionTensorEntryCounts(std.math.maxInt(usize)),
    );

    const too_large_for_value_count = @as(usize, 1) << (@bitSizeOf(usize) / 2);
    try std.testing.expectError(
        error.DimensionTooLarge,
        electronRepulsionTensorEntryCounts(too_large_for_value_count),
    );
}

/// Runtime options for materialized or streamed ERI generation.
pub fn ElectronRepulsionOptions(comptime T: type) type {
    scalar.assertFloat(T);
    return struct {
        /// Skip shell quartets whose Schwarz bound is less than or equal to
        /// this threshold. The default preserves the current unscreened path.
        schwarz_threshold: T = 0,
    };
}

/// One AO-space shell-quartet block emitted by the streaming ERI API.
pub fn ElectronRepulsionBlock(comptime T: type) type {
    scalar.assertFloat(T);
    return struct {
        /// Shell quartet indexes emitted by `electronRepulsionBlocks`.
        /// The stream contains only unique quartets: `(a,b)` and `(c,d)`
        /// are shell pairs with `a <= b`, `c <= d`, and pair `(a,b)` is
        /// ordered before or equal to `(c,d)`.
        shell_indices: [4]usize,
        ao_offsets: [4]usize,
        ao_counts: [4]usize,
        /// AO-space shell-quartet block, row-major over
        /// (a_local, b_local, c_local, d_local). Borrowed until the callback
        /// passed to `electronRepulsionBlocks` returns.
        values: []const T,
    };
}

/// Packed upper-triangular index. Used both for (ao_i, ao_j) → pair and
/// (pair_ij, pair_kl) → quartet; same triangular pack on different keys.
fn triangleIndex(a: usize, b: usize) usize {
    const smaller = @min(a, b);
    const larger = @max(a, b);
    return larger * (larger + 1) / 2 + smaller;
}

/// Build the full packed AO ERI tensor with default options.
pub fn electronRepulsionTensor(
    comptime T: type,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
) ElectronRepulsionTensorError!ElectronRepulsionTensor(T) {
    return electronRepulsionTensorWithOptions(T, allocator, shells, .{});
}

/// Build the full packed AO ERI tensor, optionally screening shell quartets.
pub fn electronRepulsionTensorWithOptions(
    comptime T: type,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
    options: ElectronRepulsionOptions(T),
) ElectronRepulsionTensorError!ElectronRepulsionTensor(T) {
    scalar.assertFloat(T);
    try common.validateShells(T, shells);

    const shell_offsets = try gaussian_orbital.shellOffsets(T, allocator, shells);
    defer allocator.free(shell_offsets);

    var result = try ElectronRepulsionTensor(T).init(allocator, shell_offsets[shells.len]);
    errdefer result.deinit(allocator);
    @memset(result.values, 0);

    var context = TensorFillContext(T){ .tensor = &result };
    try electronRepulsionBlocksAssumeValid(
        T,
        allocator,
        shells,
        shell_offsets,
        options,
        &context,
        tensorFillCallback(T),
    );

    return result;
}

/// Stream unique AO-space shell-quartet ERI blocks.
///
/// Blocks are emitted in chemist notation `(ab|cd)` with only the unique
/// shell-pair ordering described on `ElectronRepulsionBlock.shell_indices`.
/// Callers that need a full AO tensor or a direct contraction must account
/// for the usual ERI permutation symmetry:
/// `(ab|cd) = (ba|cd) = (ab|dc) = (ba|dc) = (cd|ab) = ...`.
pub fn electronRepulsionBlocks(
    comptime T: type,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
    options: ElectronRepulsionOptions(T),
    context: anytype,
    comptime callback: anytype,
) ElectronRepulsionBlocksError(CallbackReturnError(callback))!void {
    scalar.assertFloat(T);
    try common.validateShells(T, shells);

    const shell_offsets = try gaussian_orbital.shellOffsets(T, allocator, shells);
    defer allocator.free(shell_offsets);

    return electronRepulsionBlocksAssumeValid(
        T,
        allocator,
        shells,
        shell_offsets,
        options,
        context,
        callback,
    );
}

/// Stream unique shell quartets after validation and AO offset computation.
fn electronRepulsionBlocksAssumeValid(
    comptime T: type,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
    shell_offsets: []const usize,
    options: ElectronRepulsionOptions(T),
    context: anytype,
    comptime callback: anytype,
) ElectronRepulsionBlocksError(CallbackReturnError(callback))!void {
    scalar.assertFloat(T);

    // Worst-case scratch sized to the basis once, reused across every
    // shell quartet. For a typical basis (max_l ≤ 3) the CoulombTable is
    // well under 1 MiB; per-quartet alloc/free was the dominant
    // allocation cost in the inner loop.
    const max_l = common.maxAngularMomentum(T, shells);
    var coulomb_table = try makeCoulombTable(T, allocator, max_l);
    defer coulomb_table.deinit(allocator);

    const max_cart = gaussian_orbital.cartesianComponentCount(max_l);
    const max_ao = gaussian_orbital.shellComponentCount(.cartesian, max_l);
    const cartesian_block = try allocator.alloc(T, max_cart * max_cart * max_cart * max_cart);
    defer allocator.free(cartesian_block);
    const ao_block = try allocator.alloc(T, max_ao * max_ao * max_ao * max_ao);
    defer allocator.free(ao_block);
    const transform_block = try allocator.alloc(T, max_cart * max_cart * max_cart * max_cart);
    defer allocator.free(transform_block);
    const primitive_pair_cache = try allocator.alloc(PrimitivePairEntry(T), maxPrimitivePairCount(T, shells));
    defer allocator.free(primitive_pair_cache);

    var scratch = QuartetScratch(T){
        .coulomb_table = &coulomb_table,
        .cartesian_block = cartesian_block,
        .ao_block = ao_block,
        .transform_block = transform_block,
        .primitive_pair_cache = primitive_pair_cache,
    };

    const schwarz_bounds = if (options.schwarz_threshold > 0)
        try computeSchwarzBounds(T, allocator, shells, &scratch)
    else
        null;
    defer if (schwarz_bounds) |bounds| allocator.free(bounds);

    // Iterate unique shell quartets with ab_pair_index ≤ cd_pair_index,
    // where each pair index runs over (i ≤ j) in lexicographic order.
    // That is: c > a, or (c == a and d ≥ b).
    for (0..shells.len) |a_index| {
        for (a_index..shells.len) |b_index| {
            for (a_index..shells.len) |c_index| {
                const d_start = if (c_index == a_index) b_index else c_index;
                for (d_start..shells.len) |d_index| {
                    const a_shell = shells[a_index];
                    const b_shell = shells[b_index];
                    const c_shell = shells[c_index];
                    const d_shell = shells[d_index];

                    if (schwarz_bounds) |bounds| {
                        const ab_pair_index = triangleIndex(a_index, b_index);
                        const cd_pair_index = triangleIndex(c_index, d_index);
                        const bound = @sqrt(bounds[ab_pair_index] * bounds[cd_pair_index]);
                        if (bound <= options.schwarz_threshold) continue;
                    }

                    const a_count = gaussian_orbital.shellComponentCount(a_shell.kind, a_shell.angular_momentum);
                    const b_count = gaussian_orbital.shellComponentCount(b_shell.kind, b_shell.angular_momentum);
                    const c_count = gaussian_orbital.shellComponentCount(c_shell.kind, c_shell.angular_momentum);
                    const d_count = gaussian_orbital.shellComponentCount(d_shell.kind, d_shell.angular_momentum);

                    const block = scratch.ao_block[0 .. a_count * b_count * c_count * d_count];
                    try shellQuartetElectronRepulsion(
                        T,
                        a_shell,
                        b_shell,
                        c_shell,
                        d_shell,
                        block,
                        &scratch,
                    );

                    try callback(context, ElectronRepulsionBlock(T){
                        .shell_indices = .{ a_index, b_index, c_index, d_index },
                        .ao_offsets = .{
                            shell_offsets[a_index],
                            shell_offsets[b_index],
                            shell_offsets[c_index],
                            shell_offsets[d_index],
                        },
                        .ao_counts = .{ a_count, b_count, c_count, d_count },
                        .values = block,
                    });
                }
            }
        }
    }
}

/// Callback context for filling a materialized ERI tensor from streamed blocks.
fn TensorFillContext(comptime T: type) type {
    return struct {
        tensor: *ElectronRepulsionTensor(T),
    };
}

/// Build the callback that stamps streamed shell-quartet blocks into a tensor.
fn tensorFillCallback(comptime T: type) fn (*TensorFillContext(T), ElectronRepulsionBlock(T)) error{}!void {
    return struct {
        fn callback(context: *TensorFillContext(T), block: ElectronRepulsionBlock(T)) error{}!void {
            const a_count = block.ao_counts[0];
            const b_count = block.ao_counts[1];
            const c_count = block.ao_counts[2];
            const d_count = block.ao_counts[3];

            for (0..a_count) |a_local| {
                for (0..b_count) |b_local| {
                    for (0..c_count) |c_local| {
                        for (0..d_count) |d_local| {
                            context.tensor.set(
                                block.ao_offsets[0] + a_local,
                                block.ao_offsets[1] + b_local,
                                block.ao_offsets[2] + c_local,
                                block.ao_offsets[3] + d_local,
                                block.values[
                                    common.rowMajorIndex4(
                                        a_local,
                                        b_local,
                                        c_local,
                                        d_local,
                                        b_count,
                                        c_count,
                                        d_count,
                                    )
                                ],
                            );
                        }
                    }
                }
            }
        }
    }.callback;
}

/// Maximum primitive-pair cache entries needed for any shell pair in `shells`.
fn maxPrimitivePairCount(
    comptime T: type,
    shells: []const gaussian_orbital.Shell(T),
) usize {
    var max_primitive_count: usize = 0;
    for (shells) |shell| {
        max_primitive_count = @max(max_primitive_count, shell.primitives.len);
    }
    return max_primitive_count * max_primitive_count;
}

/// Compute Schwarz bounds `(ab|ab)` for shell-pair screening.
fn computeSchwarzBounds(
    comptime T: type,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
    scratch: *QuartetScratch(T),
) ElectronRepulsionTensorError![]T {
    const pair_count = try matrix.packedSymmetricEntryCount(shells.len);
    const bounds = try allocator.alloc(T, pair_count);
    errdefer allocator.free(bounds);
    @memset(bounds, 0);

    for (0..shells.len) |a_index| {
        for (a_index..shells.len) |b_index| {
            const a_shell = shells[a_index];
            const b_shell = shells[b_index];
            const a_count = gaussian_orbital.shellComponentCount(a_shell.kind, a_shell.angular_momentum);
            const b_count = gaussian_orbital.shellComponentCount(b_shell.kind, b_shell.angular_momentum);
            const block = scratch.ao_block[0 .. a_count * b_count * a_count * b_count];

            try shellQuartetElectronRepulsion(
                T,
                a_shell,
                b_shell,
                a_shell,
                b_shell,
                block,
                scratch,
            );

            var max_abs: T = 0;
            for (block) |value| {
                max_abs = @max(max_abs, @abs(value));
            }
            bounds[triangleIndex(a_index, b_index)] = max_abs;
        }
    }

    return bounds;
}

/// Top-level scratch for an ERI computation. The CoulombTable is sized
/// for the basis worst case (`max_l` across all shells); the cartesian
/// and AO blocks are sized for the worst-case quartet. All three live
/// for the duration of one `electronRepulsionTensor` call.
fn QuartetScratch(comptime T: type) type {
    return struct {
        coulomb_table: *hermite.CoulombTable(T),
        cartesian_block: []T,
        ao_block: []T,
        transform_block: []T,
        primitive_pair_cache: []PrimitivePairEntry(T),
    };
}

/// Cached primitive-pair product data and Hermite tables for one shell pair.
fn PrimitivePairEntry(comptime T: type) type {
    return struct {
        pair: common.PrimitivePair(T),
        lhs_primitive_index: usize,
        rhs_primitive_index: usize,
        axis_tables: hermite.HermiteAxisTables(T),
    };
}

/// Allocate a Coulomb auxiliary table sized for any quartet up to `max_l`.
fn makeCoulombTable(
    comptime T: type,
    allocator: std.mem.Allocator,
    max_l: types.AngularMomentumOrder,
) std.mem.Allocator.Error!hermite.CoulombTable(T) {
    // Quartet axis range: ab_x + cd_x ≤ 4·max_l per axis; n ≤ 12·max_l.
    const max_axis: usize = 4 * @as(usize, max_l);
    return hermite.CoulombTable(T).init(allocator, max_axis, max_axis, max_axis, 3 * max_axis);
}

/// Prepare four shells, compute a Cartesian quartet, and transform it to AO space.
fn shellQuartetElectronRepulsion(
    comptime T: type,
    a_shell: gaussian_orbital.Shell(T),
    b_shell: gaussian_orbital.Shell(T),
    c_shell: gaussian_orbital.Shell(T),
    d_shell: gaussian_orbital.Shell(T),
    out: []T,
    scratch: *QuartetScratch(T),
) common.ValidateShellError!void {
    var a_workspace: common.PreparedShellWorkspace(T) = undefined;
    var b_workspace: common.PreparedShellWorkspace(T) = undefined;
    var c_workspace: common.PreparedShellWorkspace(T) = undefined;
    var d_workspace: common.PreparedShellWorkspace(T) = undefined;

    const a = try a_workspace.prepare(a_shell);
    const b = try b_workspace.prepare(b_shell);
    const c = try c_workspace.prepare(c_shell);
    const d = try d_workspace.prepare(d_shell);

    const cartesian_block = scratch.cartesian_block[0 .. a.components.len * b.components.len * c.components.len * d.components.len];

    computeCartesianQuartet(T, a, b, c, d, cartesian_block, scratch);

    transformQuartetBlock(T, a_shell, b_shell, c_shell, d_shell, cartesian_block, scratch.transform_block, out);
}

/// Loop over primitive-pair quartets, reset Coulomb auxiliaries, and accumulate
/// one Cartesian shell-quartet ERI block.
fn computeCartesianQuartet(
    comptime T: type,
    a: common.PreparedShell(T),
    b: common.PreparedShell(T),
    c: common.PreparedShell(T),
    d: common.PreparedShell(T),
    out: []T,
    scratch: *QuartetScratch(T),
) void {
    std.debug.assert(out.len == a.components.len * b.components.len * c.components.len * d.components.len);

    const coulomb_extent = quartetCoulombExtent(T, a, b, c, d);
    @memset(out, 0);

    var ab_axis_tables: hermite.HermiteAxisTables(T) = undefined;
    const cd_entries = fillPrimitivePairCache(T, c, d, scratch.primitive_pair_cache);

    for (a.shell.primitives, 0..) |a_primitive, a_primitive_index| {
        for (b.shell.primitives, 0..) |b_primitive, b_primitive_index| {
            const ab_pair = common.buildPrimitivePair(
                T,
                a_primitive.exponent,
                a.shell.center,
                b_primitive.exponent,
                b.shell.center,
            );
            ab_axis_tables.fill(ab_pair, a.shell.center, b.shell.center, a.max_per_axis, b.max_per_axis, .{});
            const ab_views = ab_axis_tables.views();

            for (cd_entries) |*cd_entry| {
                const cd_views = cd_entry.axis_tables.views();
                const cd_pair = cd_entry.pair;

                resetQuartetCoulombTable(
                    T,
                    scratch.coulomb_table,
                    ab_pair,
                    cd_pair,
                    coulomb_extent,
                );

                accumulatePrimitiveQuartet(
                    T,
                    a,
                    b,
                    c,
                    d,
                    out,
                    .{
                        .a = a_primitive_index,
                        .b = b_primitive_index,
                        .c = cd_entry.lhs_primitive_index,
                        .d = cd_entry.rhs_primitive_index,
                    },
                    ab_views,
                    cd_views,
                    scratch.coulomb_table,
                    eriPrefactor(T, ab_pair, cd_pair),
                );
            }
        }
    }
}

/// Maximum Hermite Coulomb orders needed by one prepared shell quartet.
fn quartetCoulombExtent(
    comptime T: type,
    a: common.PreparedShell(T),
    b: common.PreparedShell(T),
    c: common.PreparedShell(T),
    d: common.PreparedShell(T),
) hermite.CoulombExtent {
    return .{
        .x_max = @as(usize, a.max_per_axis[0]) + b.max_per_axis[0] + c.max_per_axis[0] + d.max_per_axis[0],
        .y_max = @as(usize, a.max_per_axis[1]) + b.max_per_axis[1] + c.max_per_axis[1] + d.max_per_axis[1],
        .z_max = @as(usize, a.max_per_axis[2]) + b.max_per_axis[2] + c.max_per_axis[2] + d.max_per_axis[2],
    };
}

/// Reset the Coulomb table for the interaction between two product Gaussians.
fn resetQuartetCoulombTable(
    comptime T: type,
    coulomb_table: *hermite.CoulombTable(T),
    ab_pair: common.PrimitivePair(T),
    cd_pair: common.PrimitivePair(T),
    coulomb_extent: hermite.CoulombExtent,
) void {
    const coulomb_exponent = ab_pair.combined_exponent * cd_pair.combined_exponent /
        (ab_pair.combined_exponent + cd_pair.combined_exponent);
    coulomb_table.reset(coulomb_exponent, .{
        ab_pair.product_center[0] - cd_pair.product_center[0],
        ab_pair.product_center[1] - cd_pair.product_center[1],
        ab_pair.product_center[2] - cd_pair.product_center[2],
    }, coulomb_extent);
}

/// McMurchie-Davidson primitive ERI prefactor outside the Hermite Coulomb sum.
fn eriPrefactor(
    comptime T: type,
    ab_pair: common.PrimitivePair(T),
    cd_pair: common.PrimitivePair(T),
) T {
    return @as(T, 2) *
        @as(T, std.math.pi) *
        @as(T, std.math.pi) *
        @sqrt(@as(T, std.math.pi)) /
        (ab_pair.combined_exponent *
            cd_pair.combined_exponent *
            @sqrt(ab_pair.combined_exponent + cd_pair.combined_exponent));
}

/// Cache product data and Hermite tables for all primitive pairs in one shell pair.
fn fillPrimitivePairCache(
    comptime T: type,
    lhs: common.PreparedShell(T),
    rhs: common.PreparedShell(T),
    cache: []PrimitivePairEntry(T),
) []PrimitivePairEntry(T) {
    std.debug.assert(cache.len >= lhs.shell.primitives.len * rhs.shell.primitives.len);

    var count: usize = 0;
    for (lhs.shell.primitives, 0..) |lhs_primitive, lhs_primitive_index| {
        for (rhs.shell.primitives, 0..) |rhs_primitive, rhs_primitive_index| {
            const pair = common.buildPrimitivePair(
                T,
                lhs_primitive.exponent,
                lhs.shell.center,
                rhs_primitive.exponent,
                rhs.shell.center,
            );
            const entry = &cache[count];
            entry.pair = pair;
            entry.lhs_primitive_index = lhs_primitive_index;
            entry.rhs_primitive_index = rhs_primitive_index;
            entry.axis_tables.fill(
                pair,
                lhs.shell.center,
                rhs.shell.center,
                lhs.max_per_axis,
                rhs.max_per_axis,
                .{},
            );
            count += 1;
        }
    }

    return cache[0..count];
}

/// Primitive indexes for the four shells in one primitive quartet.
const PrimitiveIndexes = struct {
    a: usize,
    b: usize,
    c: usize,
    d: usize,
};

/// Accumulate one primitive quartet into the Cartesian shell-quartet block.
fn accumulatePrimitiveQuartet(
    comptime T: type,
    a: common.PreparedShell(T),
    b: common.PreparedShell(T),
    c: common.PreparedShell(T),
    d: common.PreparedShell(T),
    out: []T,
    primitive_indexes: PrimitiveIndexes,
    ab_axis_tables: anytype,
    cd_axis_tables: anytype,
    coulomb_table: *hermite.CoulombTable(T),
    eri_prefactor: T,
) void {
    const b_count = b.components.len;
    const c_count = c.components.len;
    const d_count = d.components.len;

    for (a.components, 0..) |a_component, a_component_index| {
        const a_coefficient = a.coefficients.at(a_component_index, primitive_indexes.a);
        for (b.components, 0..) |b_component, b_component_index| {
            const ab_coefficient = a_coefficient *
                b.coefficients.at(b_component_index, primitive_indexes.b);

            for (c.components, 0..) |c_component, c_component_index| {
                const abc_coefficient = ab_coefficient *
                    c.coefficients.at(c_component_index, primitive_indexes.c);
                for (d.components, 0..) |d_component, d_component_index| {
                    const quartet_coefficient = abc_coefficient *
                        d.coefficients.at(d_component_index, primitive_indexes.d);

                    out[
                        common.rowMajorIndex4(
                            a_component_index,
                            b_component_index,
                            c_component_index,
                            d_component_index,
                            b_count,
                            c_count,
                            d_count,
                        )
                    ] += quartet_coefficient * eri_prefactor * primitiveEri(
                        T,
                        a_component,
                        b_component,
                        c_component,
                        d_component,
                        ab_axis_tables,
                        cd_axis_tables,
                        coulomb_table,
                    );
                }
            }
        }
    }
}

/// Contract AB and CD Hermite expansions with the Coulomb auxiliary table.
fn primitiveEri(
    comptime T: type,
    a: gaussian_orbital.AngularMomentum,
    b: gaussian_orbital.AngularMomentum,
    c: gaussian_orbital.AngularMomentum,
    d: gaussian_orbital.AngularMomentum,
    ab_axis_tables: anytype,
    cd_axis_tables: anytype,
    coulomb_table: *hermite.CoulombTable(T),
) T {
    var eri_sum: T = 0;
    const ab_x_hermite_expansion_table = ab_axis_tables[0];
    const ab_y_hermite_expansion_table = ab_axis_tables[1];
    const ab_z_hermite_expansion_table = ab_axis_tables[2];
    const cd_x_hermite_expansion_table = cd_axis_tables[0];
    const cd_y_hermite_expansion_table = cd_axis_tables[1];
    const cd_z_hermite_expansion_table = cd_axis_tables[2];

    for (0..@as(usize, a.x) + b.x + 1) |ab_x_hermite_order| {
        const ab_x_axis_entry = ab_x_hermite_expansion_table.get(a.x, b.x, ab_x_hermite_order);
        if (ab_x_axis_entry == 0) continue;
        for (0..@as(usize, a.y) + b.y + 1) |ab_y_hermite_order| {
            const ab_y_axis_entry = ab_y_hermite_expansion_table.get(a.y, b.y, ab_y_hermite_order);
            if (ab_y_axis_entry == 0) continue;
            for (0..@as(usize, a.z) + b.z + 1) |ab_z_hermite_order| {
                const ab_z_axis_entry = ab_z_hermite_expansion_table.get(a.z, b.z, ab_z_hermite_order);
                if (ab_z_axis_entry == 0) continue;
                const ab_axis_table_product =
                    ab_x_axis_entry * ab_y_axis_entry * ab_z_axis_entry;

                for (0..@as(usize, c.x) + d.x + 1) |cd_x_hermite_order| {
                    const cd_x_axis_entry = cd_x_hermite_expansion_table.get(c.x, d.x, cd_x_hermite_order);
                    if (cd_x_axis_entry == 0) continue;
                    for (0..@as(usize, c.y) + d.y + 1) |cd_y_hermite_order| {
                        const cd_y_axis_entry = cd_y_hermite_expansion_table.get(c.y, d.y, cd_y_hermite_order);
                        if (cd_y_axis_entry == 0) continue;
                        for (0..@as(usize, c.z) + d.z + 1) |cd_z_hermite_order| {
                            const cd_z_axis_entry = cd_z_hermite_expansion_table.get(c.z, d.z, cd_z_hermite_order);
                            if (cd_z_axis_entry == 0) continue;

                            const sign: T = if (((cd_x_hermite_order + cd_y_hermite_order + cd_z_hermite_order) & 1) == 0) 1 else -1;
                            eri_sum += sign * ab_axis_table_product *
                                cd_x_axis_entry * cd_y_axis_entry * cd_z_axis_entry *
                                coulomb_table.get(
                                    0,
                                    ab_x_hermite_order + cd_x_hermite_order,
                                    ab_y_hermite_order + cd_y_hermite_order,
                                    ab_z_hermite_order + cd_z_hermite_order,
                                );
                        }
                    }
                }
            }
        }
    }

    return eri_sum;
}

/// Transform a Cartesian four-index shell-quartet block into AO space.
fn transformQuartetBlock(
    comptime T: type,
    a_shell: gaussian_orbital.Shell(T),
    b_shell: gaussian_orbital.Shell(T),
    c_shell: gaussian_orbital.Shell(T),
    d_shell: gaussian_orbital.Shell(T),
    cartesian_block: []T,
    transform_block: []T,
    out: []T,
) void {
    const a_cart_count = gaussian_orbital.cartesianComponentCount(a_shell.angular_momentum);
    const b_cart_count = gaussian_orbital.cartesianComponentCount(b_shell.angular_momentum);
    const c_cart_count = gaussian_orbital.cartesianComponentCount(c_shell.angular_momentum);
    const d_cart_count = gaussian_orbital.cartesianComponentCount(d_shell.angular_momentum);
    const a_count = gaussian_orbital.shellComponentCount(a_shell.kind, a_shell.angular_momentum);
    const b_count = gaussian_orbital.shellComponentCount(b_shell.kind, b_shell.angular_momentum);
    const c_count = gaussian_orbital.shellComponentCount(c_shell.kind, c_shell.angular_momentum);
    const d_count = gaussian_orbital.shellComponentCount(d_shell.kind, d_shell.angular_momentum);

    std.debug.assert(cartesian_block.len == a_cart_count * b_cart_count * c_cart_count * d_cart_count);
    std.debug.assert(out.len == a_count * b_count * c_count * d_count);

    if (a_shell.kind == .cartesian and
        b_shell.kind == .cartesian and
        c_shell.kind == .cartesian and
        d_shell.kind == .cartesian)
    {
        @memcpy(out, cartesian_block[0..out.len]);
        return;
    }

    const a_transform = SparseShellTransform(T).init(a_shell);
    const b_transform = SparseShellTransform(T).init(b_shell);
    const c_transform = SparseShellTransform(T).init(c_shell);
    const d_transform = SparseShellTransform(T).init(d_shell);

    var dims = [4]usize{ a_cart_count, b_cart_count, c_cart_count, d_cart_count };

    dims = transformAxis4(T, 0, cartesian_block, dims, &a_transform, transform_block);
    var current = transform_block[0..productDimensions(dims)];

    dims = transformAxis4(T, 1, current, dims, &b_transform, cartesian_block);
    current = cartesian_block[0..productDimensions(dims)];

    dims = transformAxis4(T, 2, current, dims, &c_transform, transform_block);
    current = transform_block[0..productDimensions(dims)];

    dims = transformAxis4(T, 3, current, dims, &d_transform, out);
    std.debug.assert(dims[0] == a_count);
    std.debug.assert(dims[1] == b_count);
    std.debug.assert(dims[2] == c_count);
    std.debug.assert(dims[3] == d_count);
}

/// One nonzero Cartesian-to-AO transform coefficient.
fn TransformEntry(comptime T: type) type {
    return struct {
        cart_index: usize,
        ao_index: usize,
        coefficient: T,
    };
}

/// Sparse transform entries for one shell, used by four-index transforms.
fn SparseShellTransform(comptime T: type) type {
    return struct {
        buffer: [common.max_cartesian_components_per_shell * common.max_components_per_shell]TransformEntry(T),
        count: usize,
        cart_count: usize,
        ao_count: usize,

        const Self = @This();

        fn init(shell: gaussian_orbital.Shell(T)) Self {
            var self: Self = undefined;
            self.count = 0;
            self.cart_count = gaussian_orbital.cartesianComponentCount(shell.angular_momentum);
            self.ao_count = gaussian_orbital.shellComponentCount(shell.kind, shell.angular_momentum);

            for (0..self.cart_count) |cart_index| {
                for (0..self.ao_count) |ao_index| {
                    const coefficient = common.cartesianToAoCoefficient(
                        T,
                        shell.kind,
                        shell.angular_momentum,
                        cart_index,
                        ao_index,
                    );
                    if (coefficient == 0) continue;
                    self.buffer[self.count] = .{
                        .cart_index = cart_index,
                        .ao_index = ao_index,
                        .coefficient = coefficient,
                    };
                    self.count += 1;
                }
            }

            return self;
        }

        fn entries(self: *const Self) []const TransformEntry(T) {
            return self.buffer[0..self.count];
        }
    };
}

/// Product of the four block dimensions.
fn productDimensions(dims: [4]usize) usize {
    return dims[0] * dims[1] * dims[2] * dims[3];
}

/// Apply one Cartesian-to-AO transform axis of a four-index block.
fn transformAxis4(
    comptime T: type,
    comptime axis: usize,
    source: []const T,
    source_dims: [4]usize,
    transform: *const SparseShellTransform(T),
    destination: []T,
) [4]usize {
    std.debug.assert(axis < 4);
    std.debug.assert(source_dims[axis] == transform.cart_count);
    std.debug.assert(source.len == productDimensions(source_dims));

    var destination_dims = source_dims;
    destination_dims[axis] = transform.ao_count;
    const destination_len = productDimensions(destination_dims);
    std.debug.assert(destination.len >= destination_len);
    @memset(destination[0..destination_len], 0);

    switch (axis) {
        0 => {
            for (transform.entries()) |entry| {
                for (0..source_dims[1]) |b| {
                    for (0..source_dims[2]) |c| {
                        for (0..source_dims[3]) |d| {
                            destination[
                                common.rowMajorIndex4(
                                    entry.ao_index,
                                    b,
                                    c,
                                    d,
                                    destination_dims[1],
                                    destination_dims[2],
                                    destination_dims[3],
                                )
                            ] += entry.coefficient * source[
                                common.rowMajorIndex4(
                                    entry.cart_index,
                                    b,
                                    c,
                                    d,
                                    source_dims[1],
                                    source_dims[2],
                                    source_dims[3],
                                )
                            ];
                        }
                    }
                }
            }
        },
        1 => {
            for (transform.entries()) |entry| {
                for (0..source_dims[0]) |a| {
                    for (0..source_dims[2]) |c| {
                        for (0..source_dims[3]) |d| {
                            destination[
                                common.rowMajorIndex4(
                                    a,
                                    entry.ao_index,
                                    c,
                                    d,
                                    destination_dims[1],
                                    destination_dims[2],
                                    destination_dims[3],
                                )
                            ] += entry.coefficient * source[
                                common.rowMajorIndex4(
                                    a,
                                    entry.cart_index,
                                    c,
                                    d,
                                    source_dims[1],
                                    source_dims[2],
                                    source_dims[3],
                                )
                            ];
                        }
                    }
                }
            }
        },
        2 => {
            for (transform.entries()) |entry| {
                for (0..source_dims[0]) |a| {
                    for (0..source_dims[1]) |b| {
                        for (0..source_dims[3]) |d| {
                            destination[
                                common.rowMajorIndex4(
                                    a,
                                    b,
                                    entry.ao_index,
                                    d,
                                    destination_dims[1],
                                    destination_dims[2],
                                    destination_dims[3],
                                )
                            ] += entry.coefficient * source[
                                common.rowMajorIndex4(
                                    a,
                                    b,
                                    entry.cart_index,
                                    d,
                                    source_dims[1],
                                    source_dims[2],
                                    source_dims[3],
                                )
                            ];
                        }
                    }
                }
            }
        },
        3 => {
            for (transform.entries()) |entry| {
                for (0..source_dims[0]) |a| {
                    for (0..source_dims[1]) |b| {
                        for (0..source_dims[2]) |c| {
                            destination[
                                common.rowMajorIndex4(
                                    a,
                                    b,
                                    c,
                                    entry.ao_index,
                                    destination_dims[1],
                                    destination_dims[2],
                                    destination_dims[3],
                                )
                            ] += entry.coefficient * source[
                                common.rowMajorIndex4(
                                    a,
                                    b,
                                    c,
                                    entry.cart_index,
                                    source_dims[1],
                                    source_dims[2],
                                    source_dims[3],
                                )
                            ];
                        }
                    }
                }
            }
        },
        else => unreachable,
    }

    return destination_dims;
}

fn StreamingCompareContext(comptime T: type) type {
    return struct {
        tensor: *const ElectronRepulsionTensor(T),
        block_count: usize = 0,
    };
}

const StreamingCompareError = error{TestExpectedApproxEqAbs};

fn streamingCompareCallback(comptime T: type) fn (*StreamingCompareContext(T), ElectronRepulsionBlock(T)) StreamingCompareError!void {
    return struct {
        fn callback(context: *StreamingCompareContext(T), block: ElectronRepulsionBlock(T)) StreamingCompareError!void {
            context.block_count += 1;

            const a_count = block.ao_counts[0];
            const b_count = block.ao_counts[1];
            const c_count = block.ao_counts[2];
            const d_count = block.ao_counts[3];
            for (0..a_count) |a_local| {
                for (0..b_count) |b_local| {
                    for (0..c_count) |c_local| {
                        for (0..d_count) |d_local| {
                            const actual = block.values[
                                common.rowMajorIndex4(
                                    a_local,
                                    b_local,
                                    c_local,
                                    d_local,
                                    b_count,
                                    c_count,
                                    d_count,
                                )
                            ];
                            const expected = context.tensor.get(
                                block.ao_offsets[0] + a_local,
                                block.ao_offsets[1] + b_local,
                                block.ao_offsets[2] + c_local,
                                block.ao_offsets[3] + d_local,
                            );
                            try std.testing.expectApproxEqAbs(expected, actual, 1e-12);
                        }
                    }
                }
            }
        }
    }.callback;
}

const BlockCountContext = struct {
    block_count: usize = 0,
};

fn countBlocksCallbackF64(context: *BlockCountContext, _: ElectronRepulsionBlock(f64)) !void {
    context.block_count += 1;
}

fn failBlocksCallbackF64(_: *BlockCountContext, _: ElectronRepulsionBlock(f64)) error{StopStreaming}!void {
    return error.StopStreaming;
}

const ExpandedSymmetryContext = struct {
    dimension: usize,
    values: []f64,
    seen: []bool,
};

fn denseFourIndex(dimension: usize, i: usize, j: usize, k: usize, l: usize) usize {
    return (((i * dimension) + j) * dimension + k) * dimension + l;
}

fn setExpandedSymmetryValue(
    context: *ExpandedSymmetryContext,
    i: usize,
    j: usize,
    k: usize,
    l: usize,
    value: f64,
) !void {
    const index = denseFourIndex(context.dimension, i, j, k, l);
    if (context.seen[index]) {
        try std.testing.expectApproxEqAbs(context.values[index], value, 1e-12);
    }
    context.values[index] = value;
    context.seen[index] = true;
}

fn expandSymmetryCallback(context: *ExpandedSymmetryContext, block: ElectronRepulsionBlock(f64)) !void {
    const a_count = block.ao_counts[0];
    const b_count = block.ao_counts[1];
    const c_count = block.ao_counts[2];
    const d_count = block.ao_counts[3];

    for (0..a_count) |a_local| {
        const a = block.ao_offsets[0] + a_local;
        for (0..b_count) |b_local| {
            const b = block.ao_offsets[1] + b_local;
            for (0..c_count) |c_local| {
                const c = block.ao_offsets[2] + c_local;
                for (0..d_count) |d_local| {
                    const d = block.ao_offsets[3] + d_local;
                    const value = block.values[
                        common.rowMajorIndex4(a_local, b_local, c_local, d_local, b_count, c_count, d_count)
                    ];

                    try setExpandedSymmetryValue(context, a, b, c, d, value);
                    try setExpandedSymmetryValue(context, b, a, c, d, value);
                    try setExpandedSymmetryValue(context, a, b, d, c, value);
                    try setExpandedSymmetryValue(context, b, a, d, c, value);
                    try setExpandedSymmetryValue(context, c, d, a, b, value);
                    try setExpandedSymmetryValue(context, d, c, a, b, value);
                    try setExpandedSymmetryValue(context, c, d, b, a, value);
                    try setExpandedSymmetryValue(context, d, c, b, a, value);
                }
            }
        }
    }
}

test "single normalized s primitive self electron repulsion" {
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

    var eri = try electronRepulsionTensor(f64, std.testing.allocator, &[_]Shell{shell});
    defer eri.deinit(std.testing.allocator);

    const expected = @as(f64, 2) * @sqrt(exponent / @as(f64, std.math.pi));
    try std.testing.expectApproxEqAbs(expected, eri.get(0, 0, 0, 0), 1e-12);
}

test "electronRepulsionBlocks streams shell quartet blocks matching materialized tensor" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const primitives = [_]Primitive{.{
        .exponent = 0.7,
        .coefficient = gaussian_orbital.radialNormalization(f64, 0.7, 0),
    }};
    const shells = [_]Shell{
        .{
            .center = .{ 0, 0, 0 },
            .angular_momentum = 0,
            .primitives = primitives[0..],
        },
        .{
            .center = .{ 0, 0, 1.4 },
            .angular_momentum = 0,
            .primitives = primitives[0..],
        },
    };

    var tensor = try electronRepulsionTensor(f64, std.testing.allocator, &shells);
    defer tensor.deinit(std.testing.allocator);

    var context = StreamingCompareContext(f64){ .tensor = &tensor };
    try electronRepulsionBlocks(
        f64,
        std.testing.allocator,
        &shells,
        .{},
        &context,
        streamingCompareCallback(f64),
    );

    try std.testing.expectEqual(@as(usize, 6), context.block_count);
}

test "electronRepulsionBlocks propagates callback errors" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const primitives = [_]Primitive{.{
        .exponent = 0.7,
        .coefficient = gaussian_orbital.radialNormalization(f64, 0.7, 0),
    }};
    const shell = Shell{
        .center = .{ 0, 0, 0 },
        .angular_momentum = 0,
        .primitives = primitives[0..],
    };

    var context: BlockCountContext = .{};
    try std.testing.expectError(
        error.StopStreaming,
        electronRepulsionBlocks(
            f64,
            std.testing.allocator,
            &[_]Shell{shell},
            .{},
            &context,
            failBlocksCallbackF64,
        ),
    );
}

test "electronRepulsionBlocks can be expanded through ERI permutation symmetry" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const primitives = [_]Primitive{.{
        .exponent = 0.7,
        .coefficient = gaussian_orbital.radialNormalization(f64, 0.7, 0),
    }};
    const shells = [_]Shell{
        .{
            .center = .{ 0, 0, 0 },
            .angular_momentum = 0,
            .primitives = primitives[0..],
        },
        .{
            .center = .{ 0, 0, 1.4 },
            .angular_momentum = 0,
            .primitives = primitives[0..],
        },
    };

    var tensor = try electronRepulsionTensor(f64, std.testing.allocator, &shells);
    defer tensor.deinit(std.testing.allocator);

    const dense_len = tensor.dimension * tensor.dimension * tensor.dimension * tensor.dimension;
    const dense_values = try std.testing.allocator.alloc(f64, dense_len);
    defer std.testing.allocator.free(dense_values);
    const seen = try std.testing.allocator.alloc(bool, dense_len);
    defer std.testing.allocator.free(seen);
    @memset(seen, false);

    var context = ExpandedSymmetryContext{
        .dimension = tensor.dimension,
        .values = dense_values,
        .seen = seen,
    };

    try electronRepulsionBlocks(
        f64,
        std.testing.allocator,
        &shells,
        .{},
        &context,
        expandSymmetryCallback,
    );

    for (0..tensor.dimension) |i| {
        for (0..tensor.dimension) |j| {
            for (0..tensor.dimension) |k| {
                for (0..tensor.dimension) |l| {
                    const dense_index = denseFourIndex(tensor.dimension, i, j, k, l);
                    try std.testing.expect(context.seen[dense_index]);
                    try std.testing.expectApproxEqAbs(
                        tensor.get(i, j, k, l),
                        context.values[dense_index],
                        1e-12,
                    );
                }
            }
        }
    }
}

test "electronRepulsionTensorWithOptions leaves screened quartets zero" {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);

    const primitives = [_]Primitive{.{
        .exponent = 0.7,
        .coefficient = gaussian_orbital.radialNormalization(f64, 0.7, 0),
    }};
    const shells = [_]Shell{
        .{
            .center = .{ 0, 0, 0 },
            .angular_momentum = 0,
            .primitives = primitives[0..],
        },
        .{
            .center = .{ 0, 0, 1.4 },
            .angular_momentum = 0,
            .primitives = primitives[0..],
        },
    };

    var tensor = try electronRepulsionTensorWithOptions(
        f64,
        std.testing.allocator,
        &shells,
        .{ .schwarz_threshold = 1e9 },
    );
    defer tensor.deinit(std.testing.allocator);

    for (tensor.values) |value| {
        try std.testing.expectEqual(@as(f64, 0), value);
    }

    var context = BlockCountContext{};
    try electronRepulsionBlocks(
        f64,
        std.testing.allocator,
        &shells,
        .{ .schwarz_threshold = 1e9 },
        &context,
        countBlocksCallbackF64,
    );
    try std.testing.expectEqual(@as(usize, 0), context.block_count);
}
