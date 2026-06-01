//! Gaussian integral kernels (Hermite/McMurchie-Davidson).

const std = @import("std");
const common = @import("integrals/common.zig");
const hermite = @import("integrals/hermite.zig");
const overlap = @import("integrals/overlap.zig");
const kinetic = @import("integrals/kinetic.zig");
const nuclear_attraction = @import("integrals/nuclear_attraction.zig");
const eri = @import("integrals/eri.zig");
const matrix = @import("matrix.zig");

pub const max_supported_angular_momentum = common.max_supported_angular_momentum;
pub const Matrix = matrix.SymmetricMatrix;
pub const ValidateShellError = common.ValidateShellError;
pub const ShellPairBlockError = common.ShellPairBlockError;
pub const OneElectronMatrixError = common.OneElectronMatrixError;

pub const overlapMatrix = overlap.overlapMatrix;
pub const shellPairOverlap = overlap.shellPairOverlap;
pub const kineticMatrix = kinetic.kineticMatrix;
pub const shellPairKinetic = kinetic.shellPairKinetic;
pub const Nucleus = nuclear_attraction.Nucleus;
pub const ValidateNucleiError = nuclear_attraction.ValidateNucleiError;
pub const NuclearAttractionMatrixError = nuclear_attraction.NuclearAttractionMatrixError;
pub const ShellPairNuclearAttractionError = nuclear_attraction.ShellPairNuclearAttractionError;
pub const nuclearAttractionMatrix = nuclear_attraction.nuclearAttractionMatrix;
pub const shellPairNuclearAttraction = nuclear_attraction.shellPairNuclearAttraction;
pub const ElectronRepulsionTensor = eri.ElectronRepulsionTensor;
pub const ElectronRepulsionTensorError = eri.ElectronRepulsionTensorError;
pub const ElectronRepulsionBlocksError = eri.ElectronRepulsionBlocksError;
pub const ElectronRepulsionOptions = eri.ElectronRepulsionOptions;
pub const ElectronRepulsionBlock = eri.ElectronRepulsionBlock;
pub const electronRepulsionTensor = eri.electronRepulsionTensor;
pub const electronRepulsionTensorWithOptions = eri.electronRepulsionTensorWithOptions;
pub const electronRepulsionBlocks = eri.electronRepulsionBlocks;

test {
    std.testing.refAllDecls(common);
    std.testing.refAllDecls(hermite);
    std.testing.refAllDecls(overlap);
    std.testing.refAllDecls(kinetic);
    std.testing.refAllDecls(nuclear_attraction);
    std.testing.refAllDecls(eri);
    _ = @import("integrals/test.zig");
}
