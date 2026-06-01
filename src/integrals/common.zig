//! Shared helpers for Gaussian integral kernels.

const std = @import("std");
const scalar = @import("../scalar.zig");
const math = @import("../math.zig");
const matrix = @import("../matrix.zig");
const gaussian_orbital = @import("../gaussian_orbital.zig");
const types = @import("../types.zig");

const AngularMomentumOrder = types.AngularMomentumOrder;

pub const max_supported_angular_momentum: AngularMomentumOrder = gaussian_orbital.max_angular_momentum;
pub const max_cartesian_components_per_shell: usize =
    gaussian_orbital.max_cartesian_components_per_shell;
pub const max_components_per_shell: usize = gaussian_orbital.max_components_per_shell;

/// Cap on primitives per shell for per-shell stack buffers.
pub const max_primitives_per_shell: usize = 64;

pub const ValidateShellError = error{
    AngularMomentumTooLarge,
    TooManyPrimitives,
    InvalidPrimitive,
};
pub const ShellPairBlockError = ValidateShellError;
pub const OneElectronMatrixError = ValidateShellError || matrix.SymmetricMatrixInitError;

/// Row-major offset for a two-index dense block.
pub inline fn rowMajorIndex(row_index: usize, column_index: usize, column_count: usize) usize {
    return row_index * column_count + column_index;
}

/// Row-major offset for a four-index dense block.
pub inline fn rowMajorIndex4(
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    b_count: usize,
    c_count: usize,
    d_count: usize,
) usize {
    return (((a * b_count) + b) * c_count + c) * d_count + d;
}

/// Gaussian product data for one primitive pair.
pub fn PrimitivePair(comptime T: type) type {
    return struct {
        /// Combined exponent p = α_a + α_b.
        combined_exponent: T,
        /// Raw lhs exponent α_a.
        lhs_exponent: T,
        /// Raw rhs exponent α_b.
        rhs_exponent: T,
        /// Gaussian-product center P = (α_a · A + α_b · B) / p.
        product_center: [3]T,

        /// ∫ exp(-p r²) d³r = (π/p)^(3/2), the volume integral of the product Gaussian.
        pub fn productGaussianVolume(self: @This()) T {
            const pi_over_p = @as(T, std.math.pi) / self.combined_exponent;
            return pi_over_p * @sqrt(pi_over_p);
        }
    };
}

/// Build the Gaussian product center and combined exponent for one primitive pair.
pub fn buildPrimitivePair(
    comptime T: type,
    lhs_exponent: T,
    lhs_center: [3]T,
    rhs_exponent: T,
    rhs_center: [3]T,
) PrimitivePair(T) {
    const combined_exponent = lhs_exponent + rhs_exponent;
    const inv_combined_exponent = @as(T, 1) / combined_exponent;

    var product_center: [3]T = undefined;
    for (&product_center, lhs_center, rhs_center) |*product, lhs, rhs| {
        product.* = (lhs_exponent * lhs + rhs_exponent * rhs) * inv_combined_exponent;
    }

    return .{
        .combined_exponent = combined_exponent,
        .lhs_exponent = lhs_exponent,
        .rhs_exponent = rhs_exponent,
        .product_center = product_center,
    };
}

/// Validate shell shape, angular momentum, center coordinates, and primitives.
pub fn validateShell(comptime T: type, shell: gaussian_orbital.Shell(T)) ValidateShellError!void {
    if (shell.angular_momentum > max_supported_angular_momentum) {
        return error.AngularMomentumTooLarge;
    }
    if (shell.primitives.len > max_primitives_per_shell) {
        return error.TooManyPrimitives;
    }
    if (shell.primitives.len == 0) {
        return error.InvalidPrimitive;
    }
    for (shell.center) |coordinate| {
        if (!std.math.isFinite(coordinate)) return error.InvalidPrimitive;
    }
    for (shell.primitives) |primitive| {
        if (primitive.exponent <= 0) return error.InvalidPrimitive;
        if (!std.math.isFinite(primitive.exponent)) return error.InvalidPrimitive;
        if (!std.math.isFinite(primitive.coefficient)) return error.InvalidPrimitive;
    }
}

/// Validate every shell before entering a matrix or tensor kernel.
pub fn validateShells(comptime T: type, shells: []const gaussian_orbital.Shell(T)) ValidateShellError!void {
    for (shells) |shell| try validateShell(T, shell);
}

/// Enumerate the Cartesian components used by the raw integral kernel.
pub fn shellCartesianComponents(
    comptime T: type,
    shell: gaussian_orbital.Shell(T),
    out: []gaussian_orbital.AngularMomentum,
) []const gaussian_orbital.AngularMomentum {
    const count = gaussian_orbital.cartesianComponents(shell.angular_momentum, out);
    return out[0..count];
}

/// Maximum Cartesian order needed on each axis for a prepared shell.
pub fn maxAngularMomentumPerAxis(
    components: []const gaussian_orbital.AngularMomentum,
) ValidateShellError![3]AngularMomentumOrder {
    var max_per_axis: [3]AngularMomentumOrder = .{ 0, 0, 0 };

    for (components) |component| {
        try component.validate(max_supported_angular_momentum);
        max_per_axis[0] = @max(max_per_axis[0], component.x);
        max_per_axis[1] = @max(max_per_axis[1], component.y);
        max_per_axis[2] = @max(max_per_axis[2], component.z);
    }

    return max_per_axis;
}

/// Premultiply each (component, primitive) coefficient by the
/// component's angular normalization. Assumes the shell has already
/// been validated via `validateShell` (positive exponents, supported L).
pub fn fillNormalizedCoefficients(
    comptime T: type,
    components: []const gaussian_orbital.AngularMomentum,
    primitives: []const gaussian_orbital.Primitive(T),
    out: []T,
) void {
    scalar.assertFloat(T);
    for (components, 0..) |component, component_index| {
        const angular_norm = gaussian_orbital.cartesianAngularNorm(T, component);
        for (primitives, 0..) |primitive, primitive_index| {
            out[rowMajorIndex(component_index, primitive_index, primitives.len)] =
                primitive.coefficient * angular_norm;
        }
    }
}

/// Normalized contraction coefficients for one shell, indexed by
/// (component, primitive). Bundles the per-shell stack buffer with its
/// dimensions so kernels can fill once and index by name instead of
/// recomputing row-major offsets.
pub fn ShellCoefficients(comptime T: type) type {
    return struct {
        buffer: [max_cartesian_components_per_shell * max_primitives_per_shell]T,
        component_count: usize,
        primitive_count: usize,

        const Self = @This();

        pub fn fill(
            self: *Self,
            components: []const gaussian_orbital.AngularMomentum,
            primitives: []const gaussian_orbital.Primitive(T),
        ) void {
            self.component_count = components.len;
            self.primitive_count = primitives.len;
            fillNormalizedCoefficients(T, components, primitives, self.slice());
        }

        pub fn slice(self: *Self) []T {
            return self.buffer[0 .. self.component_count * self.primitive_count];
        }

        pub fn at(self: *const Self, component_index: usize, primitive_index: usize) T {
            return self.buffer[rowMajorIndex(component_index, primitive_index, self.primitive_count)];
        }
    };
}

/// Shell data prepared once per block before entering primitive loops.
pub fn PreparedShell(comptime T: type) type {
    return struct {
        shell: gaussian_orbital.Shell(T),
        components: []const gaussian_orbital.AngularMomentum,
        max_per_axis: [3]AngularMomentumOrder,
        coefficients: *const ShellCoefficients(T),
    };
}

/// Stack workspace backing `PreparedShell.components` and coefficients.
/// Keep the workspace alive while using the returned `PreparedShell`.
pub fn PreparedShellWorkspace(comptime T: type) type {
    return struct {
        components_buffer: [max_cartesian_components_per_shell]gaussian_orbital.AngularMomentum,
        coefficients: ShellCoefficients(T),

        const Self = @This();

        pub fn prepare(self: *Self, shell: gaussian_orbital.Shell(T)) ValidateShellError!PreparedShell(T) {
            const components = shellCartesianComponents(T, shell, &self.components_buffer);
            const max_per_axis = try maxAngularMomentumPerAxis(components);
            self.coefficients.fill(components, shell.primitives);

            return .{
                .shell = shell,
                .components = components,
                .max_per_axis = max_per_axis,
                .coefficients = &self.coefficients,
            };
        }
    };
}

/// Maximum total angular momentum across a list of shells. Bounded by
/// `max_supported_angular_momentum` per validation; useful for sizing
/// worst-case scratch buffers (CoulombTable, ERI block) once per top-level
/// call instead of per shell pair / quartet.
pub fn maxAngularMomentum(
    comptime T: type,
    shells: []const gaussian_orbital.Shell(T),
) AngularMomentumOrder {
    var max_l: AngularMomentumOrder = 0;
    for (shells) |shell| max_l = @max(max_l, shell.angular_momentum);
    return max_l;
}

/// Dense (cartesian × AO) transform for a single shell. Built once from
/// the shell's `(kind, angular_momentum)` so the inner loop of
/// `transformTwoIndexBlock` / `transformQuartetBlock` reads a table entry
/// instead of going through the comptime switch in
/// `cartesianToAoCoefficient` for every (cart, ao) pair.
pub fn ShellTransform(comptime T: type) type {
    return struct {
        buffer: [max_cartesian_components_per_shell * max_cartesian_components_per_shell]T,
        cart_count: usize,
        ao_count: usize,

        const Self = @This();

        pub fn init(shell: gaussian_orbital.Shell(T)) Self {
            scalar.assertFloat(T);
            var self: Self = undefined;
            self.cart_count = gaussian_orbital.cartesianComponentCount(shell.angular_momentum);
            self.ao_count = gaussian_orbital.shellComponentCount(shell.kind, shell.angular_momentum);
            for (0..self.cart_count) |cart_index| {
                for (0..self.ao_count) |ao_index| {
                    self.buffer[cart_index * self.ao_count + ao_index] = cartesianToAoCoefficient(
                        T,
                        shell.kind,
                        shell.angular_momentum,
                        cart_index,
                        ao_index,
                    );
                }
            }
            return self;
        }

        pub fn at(self: *const Self, cart_index: usize, ao_index: usize) T {
            return self.buffer[cart_index * self.ao_count + ao_index];
        }
    };
}

/// Cartesian-to-AO coefficient for identity Cartesian shells or spherical transforms.
pub fn cartesianToAoCoefficient(
    comptime T: type,
    kind: gaussian_orbital.ShellKind,
    angular_momentum: AngularMomentumOrder,
    cartesian_index: usize,
    ao_index: usize,
) T {
    scalar.assertFloat(T);
    switch (kind) {
        .cartesian => return if (cartesian_index == ao_index) 1 else 0,
        .spherical => switch (angular_momentum) {
            0, 1 => return if (cartesian_index == ao_index) 1 else 0,
            inline 2...max_supported_angular_momentum => |comptime_l| {
                return sphericalTransformCoefficient(T, comptime_l, cartesian_index, ao_index);
            },
            else => return 0,
        },
    }
}

fn HarmonicPart() type {
    return enum {
        cos,
        sin,
    };
}

fn sphericalTransformCoefficient(
    comptime T: type,
    comptime angular_momentum: AngularMomentumOrder,
    cartesian_index: usize,
    ao_index: usize,
) T {
    const spherical_count = gaussian_orbital.sphericalComponentCount(angular_momentum);
    const transform = comptime sphericalTransformMatrix(T, angular_momentum);
    return transform[rowMajorIndex(cartesian_index, ao_index, spherical_count)];
}

fn sphericalTransformMatrix(
    comptime T: type,
    comptime angular_momentum: AngularMomentumOrder,
) [
    gaussian_orbital.cartesianComponentCount(angular_momentum) *
        gaussian_orbital.sphericalComponentCount(angular_momentum)
]T {
    scalar.assertFloat(T);
    @setEvalBranchQuota(5_000_000);

    const cartesian_count = gaussian_orbital.cartesianComponentCount(angular_momentum);
    const spherical_count = gaussian_orbital.sphericalComponentCount(angular_momentum);
    var transform = [_]T{0} ** (cartesian_count * spherical_count);

    inline for (0..spherical_count) |column_index| {
        const is_negative_m = column_index < angular_momentum;
        const part: HarmonicPart() = if (is_negative_m) .sin else .cos;

        const magnetic_m: AngularMomentumOrder = @intCast(if (is_negative_m)
            angular_momentum - column_index
        else
            column_index - angular_momentum);

        fillSphericalTransformColumn(
            T,
            angular_momentum,
            magnetic_m,
            part,
            column_index,
            &transform,
        );
    }

    return transform;
}

fn fillSphericalTransformColumn(
    comptime T: type,
    comptime angular_momentum: AngularMomentumOrder,
    comptime magnetic_m: AngularMomentumOrder,
    comptime part: HarmonicPart(),
    comptime column_index: usize,
    transform: *[
        gaussian_orbital.cartesianComponentCount(angular_momentum) *
            gaussian_orbital.sphericalComponentCount(angular_momentum)
    ]T,
) void {
    const cartesian_count = gaussian_orbital.cartesianComponentCount(angular_momentum);
    const spherical_count = gaussian_orbital.sphericalComponentCount(angular_momentum);

    var column = [_]T{0} ** cartesian_count;
    fillRegularSolidHarmonic(T, angular_momentum, magnetic_m, part, &column);
    normalizeSphericalColumn(T, angular_momentum, &column);

    for (0..cartesian_count) |cartesian_index| {
        transform[rowMajorIndex(cartesian_index, column_index, spherical_count)] =
            column[cartesian_index];
    }
}

fn fillRegularSolidHarmonic(
    comptime T: type,
    comptime angular_momentum: AngularMomentumOrder,
    comptime magnetic_m: AngularMomentumOrder,
    comptime part: HarmonicPart(),
    column: *[gaussian_orbital.cartesianComponentCount(angular_momentum)]T,
) void {
    @memset(column, 0);

    var radial_power_index: AngularMomentumOrder = 0;
    while (radial_power_index <= (angular_momentum - magnetic_m) / 2) : (radial_power_index += 1) {
        const base_coefficient = signedUnit(T, radial_power_index) *
            binomial(T, angular_momentum, radial_power_index) *
            binomial(T, 2 * angular_momentum - 2 * radial_power_index, angular_momentum) *
            factorial(T, angular_momentum - 2 * radial_power_index) /
            factorial(T, angular_momentum - 2 * radial_power_index - magnetic_m);
        const z_power = angular_momentum - magnetic_m - 2 * radial_power_index;

        var x_radial_power: AngularMomentumOrder = 0;
        while (x_radial_power <= radial_power_index) : (x_radial_power += 1) {
            var y_radial_power: AngularMomentumOrder = 0;
            while (y_radial_power <= radial_power_index - x_radial_power) : (y_radial_power += 1) {
                const z_radial_power = radial_power_index - x_radial_power - y_radial_power;
                const radial_coefficient = multinomial3(
                    T,
                    radial_power_index,
                    x_radial_power,
                    y_radial_power,
                    z_radial_power,
                );

                addComplexPowerTerms(
                    T,
                    angular_momentum,
                    magnetic_m,
                    part,
                    .{
                        .coefficient = base_coefficient * radial_coefficient,
                        .x_offset = 2 * x_radial_power,
                        .y_offset = 2 * y_radial_power,
                        .z_offset = 2 * z_radial_power + z_power,
                    },
                    column,
                );
            }
        }
    }
}

fn addComplexPowerTerms(
    comptime T: type,
    comptime angular_momentum: AngularMomentumOrder,
    comptime magnetic_m: AngularMomentumOrder,
    comptime part: HarmonicPart(),
    offset: anytype,
    column: *[gaussian_orbital.cartesianComponentCount(angular_momentum)]T,
) void {
    switch (part) {
        .cos => {
            var j: AngularMomentumOrder = 0;
            while (j <= magnetic_m / 2) : (j += 1) {
                const y_power = 2 * j;
                const x_power = magnetic_m - y_power;
                const term_coefficient = offset.coefficient *
                    signedUnit(T, j) *
                    binomial(T, magnetic_m, y_power);
                addNormalizedCartesianCoefficient(
                    T,
                    angular_momentum,
                    offset.x_offset + x_power,
                    offset.y_offset + y_power,
                    offset.z_offset,
                    term_coefficient,
                    column,
                );
            }
        },
        .sin => {
            var j: AngularMomentumOrder = 0;
            while (j <= (magnetic_m - 1) / 2) : (j += 1) {
                const y_power = 2 * j + 1;
                const x_power = magnetic_m - y_power;
                const term_coefficient = offset.coefficient *
                    signedUnit(T, j) *
                    binomial(T, magnetic_m, y_power);
                addNormalizedCartesianCoefficient(
                    T,
                    angular_momentum,
                    offset.x_offset + x_power,
                    offset.y_offset + y_power,
                    offset.z_offset,
                    term_coefficient,
                    column,
                );
            }
        },
    }
}

fn addNormalizedCartesianCoefficient(
    comptime T: type,
    comptime angular_momentum: AngularMomentumOrder,
    x_power: AngularMomentumOrder,
    y_power: AngularMomentumOrder,
    z_power: AngularMomentumOrder,
    raw_coefficient: T,
    column: *[gaussian_orbital.cartesianComponentCount(angular_momentum)]T,
) void {
    const cartesian_index = cartesianComponentIndex(
        angular_momentum,
        x_power,
        y_power,
        z_power,
    );
    const inverse_angular_norm =
        @sqrt(
            math.doubleFactorialOdd(T, x_power) *
                math.doubleFactorialOdd(T, y_power) *
                math.doubleFactorialOdd(T, z_power),
        );
    column[cartesian_index] += raw_coefficient * inverse_angular_norm;
}

fn normalizeSphericalColumn(
    comptime T: type,
    comptime angular_momentum: AngularMomentumOrder,
    column: *[gaussian_orbital.cartesianComponentCount(angular_momentum)]T,
) void {
    const cartesian_count = gaussian_orbital.cartesianComponentCount(angular_momentum);
    const components = cartesianComponentsFor(angular_momentum);
    var norm_squared: T = 0;

    for (0..cartesian_count) |lhs_index| {
        for (0..cartesian_count) |rhs_index| {
            norm_squared += column[lhs_index] *
                normalizedCartesianMetric(T, components[lhs_index], components[rhs_index]) *
                column[rhs_index];
        }
    }

    const inverse_norm = 1 / @sqrt(norm_squared);
    for (0..cartesian_count) |index| {
        column[index] *= inverse_norm;
    }
}

fn normalizedCartesianMetric(
    comptime T: type,
    lhs: gaussian_orbital.AngularMomentum,
    rhs: gaussian_orbital.AngularMomentum,
) T {
    return normalizedAxisMetric(T, lhs.x, rhs.x) *
        normalizedAxisMetric(T, lhs.y, rhs.y) *
        normalizedAxisMetric(T, lhs.z, rhs.z);
}

fn normalizedAxisMetric(comptime T: type, lhs_order: AngularMomentumOrder, rhs_order: AngularMomentumOrder) T {
    const total_order = lhs_order + rhs_order;
    if (math.isOdd(total_order)) return 0;

    return math.doubleFactorialOdd(T, total_order / 2) /
        @sqrt(
            math.doubleFactorialOdd(T, lhs_order) *
                math.doubleFactorialOdd(T, rhs_order),
        );
}

fn cartesianComponentIndex(
    comptime angular_momentum: AngularMomentumOrder,
    x_power: AngularMomentumOrder,
    y_power: AngularMomentumOrder,
    z_power: AngularMomentumOrder,
) usize {
    std.debug.assert(x_power + y_power + z_power == angular_momentum);

    var index: usize = 0;
    for (0..@as(usize, angular_momentum) + 1) |x_offset_usize| {
        const x_offset: AngularMomentumOrder = @intCast(x_offset_usize);
        const x_candidate = angular_momentum - x_offset;
        const max_y = angular_momentum - x_candidate;
        for (0..@as(usize, max_y) + 1) |y_offset_usize| {
            const y_offset: AngularMomentumOrder = @intCast(y_offset_usize);
            const y_candidate = max_y - y_offset;
            const z_candidate = angular_momentum - x_candidate - y_candidate;
            if (x_candidate == x_power and y_candidate == y_power and z_candidate == z_power) {
                return index;
            }
            index += 1;
        }
    }

    unreachable;
}

fn cartesianComponentsFor(
    comptime angular_momentum: AngularMomentumOrder,
) [gaussian_orbital.cartesianComponentCount(angular_momentum)]gaussian_orbital.AngularMomentum {
    const cartesian_count = comptime gaussian_orbital.cartesianComponentCount(angular_momentum);
    var components: [cartesian_count]gaussian_orbital.AngularMomentum = undefined;

    var index: usize = 0;
    for (0..@as(usize, angular_momentum) + 1) |x_offset_usize| {
        const x_offset: AngularMomentumOrder = @intCast(x_offset_usize);
        const x = angular_momentum - x_offset;
        const max_y = angular_momentum - x;
        for (0..@as(usize, max_y) + 1) |y_offset_usize| {
            const y_offset: AngularMomentumOrder = @intCast(y_offset_usize);
            const y = max_y - y_offset;
            const z = angular_momentum - x - y;
            components[index] = .{ .x = x, .y = y, .z = z };
            index += 1;
        }
    }

    return components;
}

fn signedUnit(comptime T: type, value: AngularMomentumOrder) T {
    return if ((value & 1) == 0) 1 else -1;
}

fn factorial(comptime T: type, value: AngularMomentumOrder) T {
    var result: T = 1;
    var i: AngularMomentumOrder = 2;
    while (i <= value) : (i += 1) {
        result *= @as(T, @floatFromInt(i));
    }
    return result;
}

fn binomial(comptime T: type, n: AngularMomentumOrder, k: AngularMomentumOrder) T {
    if (k > n) return 0;
    return factorial(T, n) / (factorial(T, k) * factorial(T, n - k));
}

fn multinomial3(
    comptime T: type,
    n: AngularMomentumOrder,
    a: AngularMomentumOrder,
    b: AngularMomentumOrder,
    c: AngularMomentumOrder,
) T {
    std.debug.assert(a + b + c == n);
    return factorial(T, n) / (factorial(T, a) * factorial(T, b) * factorial(T, c));
}

/// Transform a Cartesian two-index shell-pair block into AO space.
pub fn transformTwoIndexBlock(
    comptime T: type,
    lhs_shell: gaussian_orbital.Shell(T),
    rhs_shell: gaussian_orbital.Shell(T),
    cartesian_block: []const T,
    out: []T,
) void {
    scalar.assertFloat(T);

    const lhs_cart_count = gaussian_orbital.cartesianComponentCount(lhs_shell.angular_momentum);
    const rhs_cart_count = gaussian_orbital.cartesianComponentCount(rhs_shell.angular_momentum);
    const lhs_ao_count = gaussian_orbital.shellComponentCount(lhs_shell.kind, lhs_shell.angular_momentum);
    const rhs_ao_count = gaussian_orbital.shellComponentCount(rhs_shell.kind, rhs_shell.angular_momentum);

    std.debug.assert(cartesian_block.len == lhs_cart_count * rhs_cart_count);
    std.debug.assert(out.len == lhs_ao_count * rhs_ao_count);

    if (lhs_shell.kind == .cartesian and rhs_shell.kind == .cartesian) {
        @memcpy(out, cartesian_block[0..out.len]);
        return;
    }

    const lhs_transform = ShellTransform(T).init(lhs_shell);
    const rhs_transform = ShellTransform(T).init(rhs_shell);

    for (0..lhs_ao_count) |lhs_ao_index| {
        for (0..rhs_ao_count) |rhs_ao_index| {
            var transformed_value: T = 0;
            for (0..lhs_cart_count) |lhs_cartesian_index| {
                const lhs_coefficient = lhs_transform.at(lhs_cartesian_index, lhs_ao_index);
                if (lhs_coefficient == 0) continue;

                for (0..rhs_cart_count) |rhs_cartesian_index| {
                    const rhs_coefficient = rhs_transform.at(rhs_cartesian_index, rhs_ao_index);
                    if (rhs_coefficient == 0) continue;

                    transformed_value += lhs_coefficient * rhs_coefficient * cartesian_block[
                        rowMajorIndex(lhs_cartesian_index, rhs_cartesian_index, rhs_cart_count)
                    ];
                }
            }
            out[rowMajorIndex(lhs_ao_index, rhs_ao_index, rhs_ao_count)] = transformed_value;
        }
    }
}

/// Compute the (lhs_ao × rhs_ao) AO-space block for one shell pair under a
/// caller-supplied 1e kernel. `computeBlockFn` is monomorphized at the call
/// site, so the indirection is zero-cost; `extra` is forwarded verbatim and
/// is the kernel's only channel for runtime-valued context (e.g. nuclei,
/// allocator).
pub fn shellPairBlock(
    comptime T: type,
    comptime computeBlockFn: anytype,
    lhs_shell: gaussian_orbital.Shell(T),
    rhs_shell: gaussian_orbital.Shell(T),
    extra: anytype,
    out: []T,
) ShellPairBlockError!void {
    try validateShell(T, lhs_shell);
    try validateShell(T, rhs_shell);
    return shellPairBlockAssumeValid(T, computeBlockFn, lhs_shell, rhs_shell, extra, out);
}

/// Compute and transform a shell-pair block after the caller has validated shells.
pub fn shellPairBlockAssumeValid(
    comptime T: type,
    comptime computeBlockFn: anytype,
    lhs_shell: gaussian_orbital.Shell(T),
    rhs_shell: gaussian_orbital.Shell(T),
    extra: anytype,
    out: []T,
) ShellPairBlockError!void {
    const lhs_count = gaussian_orbital.cartesianComponentCount(lhs_shell.angular_momentum);
    const rhs_count = gaussian_orbital.cartesianComponentCount(rhs_shell.angular_momentum);

    var cartesian_block_buffer: [max_cartesian_components_per_shell * max_cartesian_components_per_shell]T = undefined;
    const cartesian_block = cartesian_block_buffer[0 .. lhs_count * rhs_count];

    try computeBlockFn(
        T,
        lhs_shell,
        rhs_shell,
        extra,
        cartesian_block,
    );

    transformTwoIndexBlock(T, lhs_shell, rhs_shell, cartesian_block, out);
}

/// Build the full symmetric AO matrix for a 1e operator by iterating shell
/// pairs and stamping in each `shellPairBlock` result. `computeBlockFn` and
/// `extra` are forwarded straight through.
pub fn oneElectronMatrix(
    comptime T: type,
    comptime computeBlockFn: anytype,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
    extra: anytype,
) OneElectronMatrixError!matrix.SymmetricMatrix(T) {
    scalar.assertFloat(T);
    try validateShells(T, shells);
    return oneElectronMatrixAssumeValid(T, computeBlockFn, allocator, shells, extra);
}

/// Assemble a validated shell list into packed symmetric AO matrix storage.
pub fn oneElectronMatrixAssumeValid(
    comptime T: type,
    comptime computeBlockFn: anytype,
    allocator: std.mem.Allocator,
    shells: []const gaussian_orbital.Shell(T),
    extra: anytype,
) OneElectronMatrixError!matrix.SymmetricMatrix(T) {
    scalar.assertFloat(T);
    const offsets = try gaussian_orbital.shellOffsets(T, allocator, shells);
    defer allocator.free(offsets);

    var result = try matrix.SymmetricMatrix(T).init(allocator, offsets[shells.len]);
    errdefer result.deinit(allocator);

    var block_buffer: [max_components_per_shell * max_components_per_shell]T = undefined;

    for (shells, 0..) |lhs_shell, lhs_shell_index| {
        for (shells[lhs_shell_index..], lhs_shell_index..) |rhs_shell, rhs_shell_index| {
            const lhs_ao_count = gaussian_orbital.shellComponentCount(lhs_shell.kind, lhs_shell.angular_momentum);
            const rhs_ao_count = gaussian_orbital.shellComponentCount(rhs_shell.kind, rhs_shell.angular_momentum);
            const block = block_buffer[0 .. lhs_ao_count * rhs_ao_count];

            try shellPairBlockAssumeValid(T, computeBlockFn, lhs_shell, rhs_shell, extra, block);

            const row_offset = offsets[lhs_shell_index];
            const col_offset = offsets[rhs_shell_index];
            for (0..lhs_ao_count) |lhs_index| {
                for (0..rhs_ao_count) |rhs_index| {
                    result.set(
                        row_offset + lhs_index,
                        col_offset + rhs_index,
                        block[rowMajorIndex(lhs_index, rhs_index, rhs_ao_count)],
                    );
                }
            }
        }
    }

    return result;
}

fn expectSphericalTransformProducesIdentity(comptime angular_momentum: AngularMomentumOrder) !void {
    const Primitive = gaussian_orbital.Primitive(f64);
    const Shell = gaussian_orbital.Shell(f64);
    const cartesian_count = comptime gaussian_orbital.cartesianComponentCount(angular_momentum);
    const spherical_count = comptime gaussian_orbital.sphericalComponentCount(angular_momentum);
    const components = cartesianComponentsFor(angular_momentum);

    var cartesian_overlap: [cartesian_count * cartesian_count]f64 = undefined;
    for (0..cartesian_count) |row| {
        for (0..cartesian_count) |col| {
            cartesian_overlap[rowMajorIndex(row, col, cartesian_count)] =
                normalizedCartesianMetric(f64, components[row], components[col]);
        }
    }

    const primitives = [_]Primitive{.{ .exponent = 1, .coefficient = 1 }};
    const shell = Shell{
        .center = .{ 0, 0, 0 },
        .angular_momentum = angular_momentum,
        .kind = .spherical,
        .primitives = primitives[0..],
    };

    var spherical_overlap: [spherical_count * spherical_count]f64 = undefined;
    transformTwoIndexBlock(f64, shell, shell, &cartesian_overlap, &spherical_overlap);

    for (0..spherical_count) |row| {
        for (0..spherical_count) |col| {
            const expected: f64 = if (row == col) 1 else 0;
            try std.testing.expectApproxEqAbs(
                expected,
                spherical_overlap[rowMajorIndex(row, col, spherical_count)],
                1e-12,
            );
        }
    }
}

test "generated spherical transforms normalize same-center Cartesian overlap" {
    inline for (0..max_supported_angular_momentum + 1) |angular_momentum| {
        try expectSphericalTransformProducesIdentity(@intCast(angular_momentum));
    }
}

test "generated d-shell transform matches PySCF pure-d order" {
    const sqrt3_half = @sqrt(@as(f64, 3)) * 0.5;

    try std.testing.expectApproxEqAbs(
        @as(f64, 1),
        cartesianToAoCoefficient(f64, .spherical, 2, 1, 0),
        1e-12,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, 1),
        cartesianToAoCoefficient(f64, .spherical, 2, 4, 1),
        1e-12,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, -0.5),
        cartesianToAoCoefficient(f64, .spherical, 2, 0, 2),
        1e-12,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, -0.5),
        cartesianToAoCoefficient(f64, .spherical, 2, 3, 2),
        1e-12,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, 1),
        cartesianToAoCoefficient(f64, .spherical, 2, 5, 2),
        1e-12,
    );
    try std.testing.expectApproxEqAbs(
        @as(f64, 1),
        cartesianToAoCoefficient(f64, .spherical, 2, 2, 3),
        1e-12,
    );
    try std.testing.expectApproxEqAbs(
        sqrt3_half,
        cartesianToAoCoefficient(f64, .spherical, 2, 0, 4),
        1e-12,
    );
    try std.testing.expectApproxEqAbs(
        -sqrt3_half,
        cartesianToAoCoefficient(f64, .spherical, 2, 3, 4),
        1e-12,
    );
}
