//! Compile-time scalar type checks shared across the library.

pub fn assertFloat(comptime T: type) void {
    switch (@typeInfo(T)) {
        .float, .comptime_float => {},
        else => @compileError("scalar type must be a floating-point type"),
    }
}
