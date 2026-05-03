const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const Ministral3 = @import("ministral3.zig");

const log = std.log.scoped(.ministral3);

pub const CompilationOptions = struct {
    batch_dim: u32,
    seqlen: usize,
    rng: zml.Tensor.Rng,
    shardings: common.Shardings,
    kv_cache: Ministral3.KvCache,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,
    channel: u32,
    width: u32,
    height: u32,

    pub fn init(config: Ministral3.Config, model: Ministral3, shardings: common.Shardings, backend: zml.attention.attention.Backend, seqlen: usize) CompilationOptions {
        const batch_dim = 1;
        return .{
            .batch_dim = batch_dim,
            .seqlen = seqlen,
            .rng = .init(),
            .shardings = shardings,
            .kv_cache = .init(config, model, batch_dim, seqlen),
            .attention_metadata = .init(.fromBackend(backend, @intCast(seqlen), @intCast(config.text_config.num_attention_heads))),
            .attention_parameters = .init(.fromBackend(backend)),
            .channel = 3,
            .width = 392,
            .height = 532,
        };
    }
};

pub const CompiledModel = struct {
    prefill: KernelExe,
    decode: KernelExe,
    vision: KernelExe,
    embeddings: KernelExe,
    decode_embeddings: KernelExe,
    params: CompilationOptions,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        model: Ministral3,
        opts: CompilationOptions,
        progress: *std.Progress.Node,
    ) !CompiledModel {
        return .{
            .embeddings = try compileEmbeddingsKernel(allocator, io, platform, model, opts.shardings, opts, progress),
            .decode_embeddings = try compileDecodeEmbeddingsKernel(allocator, io, platform, model, opts.shardings, opts, progress),
            .vision = try compileVisionKernel(allocator, io, platform, model, opts.shardings, opts, progress),
            .prefill = try compileKernel(allocator, io, platform, model, opts.shardings, opts, progress),
            .decode = try compileDecoderKernel(allocator, io, platform, model, opts.shardings, opts, progress),
            .params = opts,
        };
    }

    pub fn deinit(self: *CompiledModel) void {
        self.prefill.deinit();
        self.decode.deinit();
        self.vision.deinit();
        self.embeddings.deinit();
        self.decode_embeddings.deinit();
    }
};

fn compileEmbeddingsKernel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, model: Ministral3, shardings: common.Shardings, opts: CompilationOptions, progress: *std.Progress.Node) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling multi modal embeddings kernel...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled multi modal embeddings kernel [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = opts.seqlen }, .u32);

    const all_shardings = shardings.all();
    const exe = try platform.compile(
        allocator,
        io,
        model.embeds,
        .forward,
        .{
            tokens,
        },
        .{ .shardings = &all_shardings },
    );
    return .{ .exe = exe };
}

fn compileDecodeEmbeddingsKernel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, model: Ministral3, shardings: common.Shardings, opts: CompilationOptions, progress: *std.Progress.Node) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling multi modal embeddings kernel...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled multi modal embeddings kernel [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = 1 }, .u32);

    const all_shardings = shardings.all();
    const exe = try platform.compile(
        allocator,
        io,
        model.embeds,
        .forward,
        .{
            tokens,
        },
        .{ .shardings = &all_shardings },
    );
    return .{ .exe = exe };
}

fn compileKernel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, model: Ministral3, shardings: common.Shardings, opts: CompilationOptions, progress: *std.Progress.Node) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling prefill kernel...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled prefill kernel [{f}]", .{now.untilNow(io, .awake)});

    // const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = opts.seqlen }, .u32);
    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = opts.seqlen, .hidden = 3072 }, .bf16);
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
            opts.rng,
            opts.kv_cache,
            opts.attention_metadata,
            opts.attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    return .{ .exe = exe };
}

fn compileDecoderKernel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, model: Ministral3, shardings: common.Shardings, opts: CompilationOptions, progress: *std.Progress.Node) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling decoder kernel...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled decoder kernel [{f}]", .{now.untilNow(io, .awake)});

    // const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = 1 }, .u32);
    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .seq = 1, .hidden = 3072 }, .bf16);
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
            opts.rng,
            opts.kv_cache,
            opts.attention_metadata,
            opts.attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    return .{ .exe = exe };
}

fn compileVisionKernel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, model: Ministral3, shardings: common.Shardings, opts: CompilationOptions, progress: *std.Progress.Node) !KernelExe {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Compiling vision kernel...", 1);
    defer node.end();
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled vision kernel [{f}]", .{now.untilNow(io, .awake)});

    const tokens: zml.Tensor = .init(.{ .batch = opts.batch_dim, .channel = opts.channel, .width = opts.width, .height = opts.height }, .u32);

    const all_shardings = shardings.all();
    const exe = try platform.compile(
        allocator,
        io,
        model.vision_encoder,
        .forward,
        .{
            tokens,
            opts.attention_metadata,
            opts.attention_parameters,
        },
        .{ .shardings = &all_shardings },
    );
    return .{ .exe = exe };
}

const Args = struct {
    allocator: std.mem.Allocator,
    model_buffers: *zml.Bufferized(Ministral3),
    tokens_buf: *zml.Buffer,
    tokens_pos_buffer: *zml.Buffer,
    rng_buffer: *zml.Bufferized(zml.Tensor.Rng),
    kv_cache_buffers: *zml.Bufferized(Ministral3.KvCache),
    attention_metadata_buffers: zml.Bufferized(zml.attention.attention.Metadata),
};

const ArgsEmbds = struct {
    allocator: std.mem.Allocator,
    model_buffers: *zml.Bufferized(Ministral3),
    tokens_buf: *zml.Buffer,
};

const KernelExe = struct {
    exe: zml.Exe,

    fn deinit(self: KernelExe) void {
        self.exe.deinit();
    }

    pub fn run(self: *const KernelExe, args: Args) !void {
        var exe_args = try self.exe.args(args.allocator);
        defer exe_args.deinit(args.allocator);

        exe_args.set(.{
            args.model_buffers,
            args.tokens_buf,
            args.tokens_pos_buffer,
            args.rng_buffer,
            args.kv_cache_buffers,
            args.attention_metadata_buffers,
        });

        var results = try self.exe.results(args.allocator);
        defer results.deinit(args.allocator);

        self.exe.call(exe_args, &results);

        var tokens, var kv_cache, var rng = results.get(struct {
            zml.Buffer,
            zml.Bufferized(Ministral3.KvCache),
            zml.Bufferized(zml.Tensor.Rng),
        });

        swapBuffer(args.tokens_buf, &tokens);
        swapBuffer(&args.rng_buffer._state, &rng._state);
        swapCacheBuffers(args.kv_cache_buffers, &kv_cache);
    }

    pub fn run_embeddings(self: *const KernelExe, args: ArgsEmbds) !void {
        var exe_args = try self.exe.args(args.allocator);
        defer exe_args.deinit(args.allocator);

        exe_args.set(.{
            args.model_buffers.embeds.embed_tokens,
            args.tokens_buf,
        });

        var results = try self.exe.results(args.allocator);
        defer results.deinit(args.allocator);

        self.exe.call(exe_args, &results);

        var tokens = results.get(zml.Buffer);

        swapBuffer(args.tokens_buf, &tokens);
    }
};

fn swapBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
    if (!sameBufferHandle(dst.*, src.*)) {
        dst.deinit();
    }
    dst.* = src.*;
}

fn swapCacheBuffers(dst: *zml.Bufferized(Ministral3.KvCache), src: *zml.Bufferized(Ministral3.KvCache)) void {
    swapBuffer(&dst.k, &src.k);
    swapBuffer(&dst.v, &src.v);
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
