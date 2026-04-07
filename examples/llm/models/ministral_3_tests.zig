const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const ministral = @import("ministral_3.zig");
const model = @import("ministral_3/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,

    pub const help =
        \\Use ministral_3_tests --model=<path> --activations=<path>
        \\
        \\ Validate the LLaMA implementation against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>            Path to the model repository
        \\   --activations=<path>      Path to activation safetensors
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var repo_model = try ministral.LoadedModel.init(allocator, io, repo, store.view(), .{});
    defer repo_model.deinit(allocator);

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    const tp_mesh: zml.sharding.LogicalMesh = try .init("tp_mesh", .{ .model = .high_bandwidth });
    const tp_strategy: zml.sharding.Strategy = try .suggest(tp_mesh, platform.physical_mesh);
    const shardings: common.Shardings = .{
        .replicated = try zml.sharding.replicatedSharding(platform),
        .model = try .initFromStrategy(platform, tp_mesh, tp_strategy),
    };

    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer repo_model.unloadBuffers(&model_buffers, allocator);

    const backend = zml.attention.attention.Backend.auto(platform);

    const seqlen = 4096;
    var compiled_model = try repo_model.compile(allocator, io, platform, backend, shardings, seqlen, &progress);
    defer compiled_model.deinit();

    progress.end();

    try run(allocator, io, platform, args.activations, repo_model.inner, &model_buffers, shardings.replicated, backend);
}

fn loadBufferFromStore(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: zml.io.TensorStore.View, key: []const u8, sharding: zml.sharding.Sharding) !zml.Buffer {
    const shape = store.getShape(key) orelse return error.NotFound;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.getReader(key, io, &io_buffer);
    defer reader.deinit();

    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    mdl: ministral.Model,
    model_buffers: *model.Buffers,
    sharding: zml.sharding.Sharding,
    backend: zml.attention.attention.Backend,
) !void {
    _ = backend; // autofix
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    try testLayer(
        allocator,
        io,
        platform,
        activation_store.view(),
        "model.model.language_model.embed_tokens",
        mdl.embed_tokens,
        model_buffers.embed_tokens,
        sharding,
        .{ .absolute_tolerance = 1e-3 },
    );

    try testLayerWithTags(
        allocator,
        io,
        platform,
        activation_store.view(),
        "model.lm_head",
        mdl.lm_head,
        model_buffers.lm_head,
        sharding,
        .{ .absolute_tolerance = 2e-2 },
        .{ .batch, .seq, .hidden },
    );

    if (mdl.layers.len == 0) return;

    const layer = mdl.layers[0];
    const layer_buffers = model_buffers.layers[0];

    try testLayerWithTags(
        allocator,
        io,
        platform,
        activation_store.view(),
        "model.model.language_model.layers.0.input_layernorm",
        layer.input_norm,
        layer_buffers.input_norm,
        sharding,
        .{ .absolute_tolerance = 1e-2 },
        .{ .a, .c, .hidden },
    );

    // try testAttentionLayer(
    //     allocator,
    //     io,
    //     platform,
    //     activation_store.view(),
    //     "model.model.language_model.layers.0.self_attn",
    //     layer.self_attn,
    //     layer_buffers.self_attn,
    //     sharding,
    //     .{ .absolute_tolerance = 1e-2 },
    //     .{ .a, .c, .d },
    //     backend,
    // );

    try testLayerWithTags(
        allocator,
        io,
        platform,
        activation_store.view(),
        "model.model.language_model.layers.0.post_attention_layernorm",
        layer.post_attn,
        layer_buffers.post_attn,
        sharding,
        .{ .absolute_tolerance = 2e-2 },
        .{ .a, .c, .hidden },
    );

    try testLayerWithTags(
        allocator,
        io,
        platform,
        activation_store.view(),
        "model.model.language_model.layers.0.mlp",
        layer.feed_fwd,
        layer_buffers.feed_fwd,
        sharding,
        .{ .absolute_tolerance = 2e-2 },
        .{ .a, .c, .hidden },
    );
}

fn testLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: zml.io.TensorStore.View,
    name: []const u8,
    layer: anytype,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    sharding: zml.sharding.Sharding,
    opts: zml.testing.CompareOpts,
) !void {
    const in_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
    defer allocator.free(in_key);
    const in_shape = activation_store.getShape(in_key) orelse return error.NotFound;
    var in_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, in_key, sharding);
    defer in_buffer.deinit();
    const in_tensor = zml.Tensor.fromShape(in_shape);

    const out_key = try std.fmt.allocPrint(allocator, "{s}.out.0", .{name});
    defer allocator.free(out_key);
    var out_buffer_expected = try loadBufferFromStore(allocator, io, platform, activation_store, out_key, sharding);
    defer out_buffer_expected.deinit();

    const exe = try platform.compileFn(allocator, io, @TypeOf(layer).forward, .{ layer, in_tensor }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ layer_weights, in_buffer });

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(args, &res);

    var out_result = res.get(zml.Buffer);
    defer out_result.deinit();
    try zml.testing.expectClose(io, out_result, out_buffer_expected, opts);
}

fn testLayerWithTags(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: zml.io.TensorStore.View,
    name: []const u8,
    layer: anytype,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    sharding: zml.sharding.Sharding,
    opts: zml.testing.CompareOpts,
    tags: anytype,
) !void {
    const in_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
    defer allocator.free(in_key);
    const in_shape = activation_store.getShape(in_key) orelse return error.NotFound;
    var in_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, in_key, sharding);
    defer in_buffer.deinit();
    const in_tensor = zml.Tensor.fromShape(in_shape).withTags(tags);

    const out_key = try std.fmt.allocPrint(allocator, "{s}.out.0", .{name});
    defer allocator.free(out_key);
    var out_buffer_expected = try loadBufferFromStore(allocator, io, platform, activation_store, out_key, sharding);
    defer out_buffer_expected.deinit();

    const exe = try platform.compileFn(allocator, io, @TypeOf(layer).forward, .{ layer, in_tensor }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ layer_weights, in_buffer });

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(args, &res);

    var out_result = res.get(zml.Buffer);
    defer out_result.deinit();
    try zml.testing.expectClose(io, out_result, out_buffer_expected, opts);
}

fn testAttentionLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: zml.io.TensorStore.View,
    name: []const u8,
    layer: anytype,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    sharding: zml.sharding.Sharding,
    opts: zml.testing.CompareOpts,
    tags: anytype,
    backend: zml.attention.attention.Backend,
) !void {
    const in_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{name});
    defer allocator.free(in_key);
    const in_shape = activation_store.getShape(in_key) orelse return error.NotFound;
    var in_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, in_key, sharding);
    defer in_buffer.deinit();
    const in_tensor = zml.Tensor.fromShape(in_shape).withTags(tags);

    const out_key = try std.fmt.allocPrint(allocator, "{s}.out.0", .{name});
    defer allocator.free(out_key);
    var out_buffer_expected = try loadBufferFromStore(allocator, io, platform, activation_store, out_key, sharding);
    defer out_buffer_expected.deinit();

    const cache: ministral.Model.KvCache = .init(.init(.{ 3072, 3072 }, .bf16)); //< find shape of kvcache
    const token_index = zml.Tensor.iota(in_shape, 1);
    const cache_index = zml.Tensor.scalar(@as(u32, 0), .u32);

    const seqlen = 11;
    const attention_heads = 32;
    const attention_metadata: zml.attention.attention.Metadata = .init(.fromBackend(backend, seqlen, attention_heads));
    const attention_params: zml.attention.attention.Parameters = .init(.fromBackend(backend));

    const exe = try platform.compileFn(allocator, io, @TypeOf(layer).forward, .{ layer, in_tensor, token_index, cache, cache_index, attention_metadata, attention_params }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ layer_weights, in_buffer });

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(args, &res);

    var out_result = res.get(zml.Buffer);
    defer out_result.deinit();
    try zml.testing.expectClose(io, out_result, out_buffer_expected, opts);
}
