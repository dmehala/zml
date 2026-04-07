const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const Ministral3 = @import("ministral3.zig");

const log = std.log.scoped(.ministral3);

pub const CompilationOptions = struct {
    batch_dim: u32,
    seqlen: usize,
    shardings: common.Shardings,
    cache: Ministral3.KvCache,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,

    pub fn init(config: Ministral3.Config, model: Ministral3, shardings: common.Shardings, backend: zml.attention.attention.Backend, seqlen: usize) CompilationOptions {
        return .{
            .batch_dim = 1,
            .seqlen = seqlen,
            .shardings = shardings,
            .cache = .init(.init(.{
                .layer = config.text_config.num_hidden_layers,
                .k = seqlen,
                .h = config.text_config.num_key_value_heads,
                .hd = config.text_config.head_dim,
            }, model.embed_tokens.weight.dtype())),
            .attention_metadata = .init(.fromBackend(backend, @intCast(seqlen), @intCast(config.text_config.num_attention_heads))),
            .attention_parameters = .init(.fromBackend(backend)),
        };
    }
};

pub const CompiledModel = struct {
    prefill: KernelExe,
    decode: KernelExe,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        model: Ministral3,
        opts: CompilationOptions,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        return .{
            .prefill = try compileKernel(allocator, io, platform, model, opts.shardings, opts, progress),
            .decode = try compileKernel(allocator, io, platform, model, opts.shardings, opts, progress),
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
    }
};

fn compileKernel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, model: Ministral3, shardings: common.Shardings, opts: CompilationOptions, progress: *std.Progress.Node) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling single kernel...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled single kernel [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = opts.seqlen }, .u32);
    const token_position_offset: zml.Tensor = .init(.{ .batch = opts.batch_dim }, .u32);

    const all_shardings = shardings.all();
    const exe = try platform.compile(
        allocator,
        io,
        model,
        .forward,
        .{
            tokens,
            token_position_offset,
            opts.cache,
            opts.attention_metadata,
            opts.attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    return .{ .exe = exe };
}

const Args = struct {
    allocator: std.mem.Allocator,
};

const KernelExe = struct {
    exe: zml.Exe,

    fn deinit(self: KernelExe) void {
        self.exe.deinit();
    }

    pub fn run(self: *const KernelExe, args: Args) !void {
        var exe_args = try self.exe.args(args.allocator);
        defer exe_args.deinit(args.allocator);

        var results = try self.exe.results(args.allocator);
        defer results.deinit(args.allocator);

        exe_args.set(.{});
        self.exe.call(exe_args, &results);

        var tokens, var cache = results.get(struct {
            zml.Buffer,
            zml.Bufferized(Ministral3.KvCache),
        });

        replaceBuffer(args.tokens_buf, &tokens);
        replaceCacheBuffers(args.cache_buffers, &cache);
    }
};

fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn replaceCacheBuffers(dst: *zml.Bufferized(Ministral3.KvCache), src: *zml.Bufferized(Ministral3.KvCache)) void {
    replaceBuffer(&dst.kv.k, &src.kv.k);
    replaceBuffer(&dst.kv.v, &src.kv.v);
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
