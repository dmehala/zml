const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const inference = @import("inference.zig");
const Ministral3 = @import("ministral3.zig");

pub const Buffers = zml.Bufferized(Ministral3);

pub const LoadedModel = struct {
    inner: Ministral3,
    parsed_config: std.json.Parsed(Ministral3.Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        opts: common.GenerationOptions,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Ministral3.Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        return .{
            .parsed_config = parsed_config,
            .inner = try .init(allocator, store, parsed_config.value, opts),
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        return self.inner.loadBuffers(allocator, io, platform, store, progress, shardings);
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        self.inner.unloadBuffers(buffers, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = self; // autofix
        _ = allocator; // autofix
        _ = io; // autofix
        _ = platform; // autofix
        _ = backend; // autofix
        _ = shardings; // autofix
        _ = seqlen; // autofix
        _ = progress; // autofix
        return error.ModelNotImplemented;
    }
};
