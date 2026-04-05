const std = @import("std");

const zml = @import("zml");

const Ministral3 = @import("ministral3.zig");

pub const CompiledModel = struct {
    pub fn init(_: std.mem.Allocator, _: std.Io, _: *zml.Platform, _: Ministral3, _: *std.Progress.Node) CompiledModel {
        return .{};
    }

    pub fn deinit(_: *CompiledModel) void {}
};
