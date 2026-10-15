const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Library module — `@import("zint")` from anywhere else in this
    // package surfaces what's exported from `src/root.zig`.
    const mod = b.addModule("zint", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const fast_reference_options = b.addOptions();
    fast_reference_options.addOption(bool, "run_slow_reference_tests", false);
    mod.addOptions("build_options", fast_reference_options);

    // CLI entry point. Stays separate from the library so consumers
    // can embed `zint` without pulling in the executable's main.
    const exe = b.addExecutable(.{
        .name = "ZInt",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    const fmt = b.addFmt(.{ .paths = b.pathList(&.{ "build.zig", "src" }), .check = true });
    b.step("fmt", "Check Zig formatting").dependOn(&fmt.step);

    // Test artifacts:
    //   - `mod_tests`       exercises the library module itself
    //   - `exe_tests`       exercises the CLI's root module
    //   - `fixture_tests`   exercises internal reference-fixture helpers
    //   - `slow_mod_tests` repeats the library tests with slow reference
    //                      cases enabled
    const mod_tests = b.addTest(.{ .root_module = mod });
    const run_mod_tests = b.addRunArtifact(mod_tests);
    run_mod_tests.has_side_effects = true;

    const exe_tests = b.addTest(.{ .root_module = exe.root_module });
    const run_exe_tests = b.addRunArtifact(exe_tests);

    const fixture_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/reference.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_fixture_tests = b.addRunArtifact(fixture_tests);

    const slow_reference_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });
    const slow_reference_options = b.addOptions();
    slow_reference_options.addOption(bool, "run_slow_reference_tests", true);
    slow_reference_mod.addOptions("build_options", slow_reference_options);
    const slow_mod_tests = b.addTest(.{ .root_module = slow_reference_mod });
    const run_slow_mod_tests = b.addRunArtifact(slow_mod_tests);
    run_slow_mod_tests.has_side_effects = true;

    const database_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/database_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_database_tests = b.addRunArtifact(database_tests);
    run_database_tests.has_side_effects = true;

    const example_default = b.addRunArtifact(exe);
    example_default.has_side_effects = true;
    example_default.expectStdOutMatch("ZInt example: h2_sto-3g");
    example_default.expectStdOutMatch("Overlap matrix S");

    const example_with_args = b.addRunArtifact(exe);
    example_with_args.addArg("testdata/reference/cases/h2_sto-3g.json");
    example_with_args.addArg("data/basis/sto-3g.json");
    example_with_args.has_side_effects = true;
    example_with_args.expectStdOutMatch("Selected electron-repulsion integrals");

    const example_malformed = b.addRunArtifact(exe);
    example_malformed.addArg("testdata/reference/malformed/shape.json");
    example_malformed.has_side_effects = true;
    example_malformed.expectExitCode(1);
    example_malformed.expectStdErrMatch("ShapeMismatch");

    const test_step = b.step("test", "Run fast tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_exe_tests.step);
    test_step.dependOn(&run_fixture_tests.step);
    test_step.dependOn(&example_default.step);
    test_step.dependOn(&example_with_args.step);
    test_step.dependOn(&example_malformed.step);

    const test_reference_step = b.step("test-reference", "Run all committed reference fixture tests");
    test_reference_step.dependOn(&run_slow_mod_tests.step);

    const test_database_step = b.step("test-database", "Run basis database corpus tests");
    test_database_step.dependOn(&run_database_tests.step);

    const test_all_step = b.step("test-all", "Run fast, reference, and database tests");
    test_all_step.dependOn(test_step);
    test_all_step.dependOn(test_reference_step);
    test_all_step.dependOn(test_database_step);

    // Build-time data tooling. Both shell out to `uv run` and are NOT
    // deps of the default build/test — they touch the network and a
    // Python toolchain. Invoke explicitly:
    //   `zig build fetch-basis`, `zig build gen-reference`.
    const fetch_cmd = b.addSystemCommand(&.{ "uv", "run", "scripts/fetch_basis_sets.py" });
    fetch_cmd.setCwd(b.path("data"));
    const fetch_step = b.step("fetch-basis", "Fetch basis sets from the Basis Set Exchange into data/basis/");
    fetch_step.dependOn(&fetch_cmd.step);

    const gen_ref_cmd = b.addSystemCommand(&.{ "uv", "run", "gen_reference.py" });
    gen_ref_cmd.setCwd(b.path("tools"));
    const gen_ref_step = b.step("gen-reference", "Generate PySCF reference data into testdata/reference/cases/");
    gen_ref_step.dependOn(&gen_ref_cmd.step);
}
