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

    // const seqlen = 4096;
    // var compiled_model = try repo_model.compile(allocator, io, platform, backend, shardings, seqlen, &progress);
    // defer compiled_model.deinit();

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
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    var ctx = TestContext{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .activation_store = activation_store.view(),
        .sharding = sharding,
        .backend = backend,
    };

    // try ctx.testLayer(
    //     "model.model.language_model.embed_tokens",
    //     mdl.embed_tokens,
    //     model_buffers.embed_tokens,
    //     .{ .absolute_tolerance = 1e-3 },
    // );
    //
    // try ctx.testLayerWithTags(
    //     "model.lm_head",
    //     mdl.lm_head,
    //     model_buffers.lm_head,
    //     .{ .absolute_tolerance = 2e-2 },
    //     .{ .batch, .seq, .hidden },
    // );

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    // expect layers.len == context.layer
    const i: usize = 0;
    const layer = mdl.layers[i];
    _ = layer; // autofix
    // // for (mdl.layers, 0..) |layer, i| {
    const layer_buffers = model_buffers.layers[i];
    _ = layer_buffers; // autofix
    //
    // try ctx.testLayerWithTags(
    //     try std.fmt.allocPrint(arena.allocator(), "model.model.language_model.layers.{d}.input_layernorm", .{i}),
    //     layer.input_norm,
    //     layer_buffers.input_norm,
    //     .{ .absolute_tolerance = 1e-2 },
    //     .{ .batch, .seq, .hidden },
    // );

    // try ctx.testAttentionLayer(
    //     "model.model.language_model.layers.0.self_attn",
    //     layer.self_attn,
    //     layer_buffers.self_attn,
    //     .{ .absolute_tolerance = 1e-2 },
    //     .{ .batch, .seq, .hidden },
    //     backend,
    // );

    // try ctx.testLayerWithTags(
    //     try std.fmt.allocPrint(arena.allocator(), "model.model.language_model.layers.{d}.post_attention_layernorm", .{i}),
    //     layer.post_attn,
    //     layer_buffers.post_attn,
    //     .{ .absolute_tolerance = 2e-2 },
    //     .{ .batch, .seq, .hidden },
    // );
    //
    // try ctx.testLayerWithTags(
    //     try std.fmt.allocPrint(arena.allocator(), "model.model.language_model.layers.{d}.mlp", .{i}),
    //     layer.feed_fwd,
    //     layer_buffers.feed_fwd,
    //     .{ .absolute_tolerance = 2e-2 },
    //     .{ .batch, .seq, .hidden },
    // );
    // }

    try ctx.testLayerWithTags(
        "model.model.vision_tower.patch_conv",
        mdl.vision_encoder.model.patch,
        model_buffers.vision_encoder.model.patch,
        .{ .absolute_tolerance = 2e-2 },
        .{ .batch, .channel, .width, .height },
    );

    try ctx.testLayerWithTags(
        "model.model.vision_tower.ln_pre",
        mdl.vision_encoder.model.ln_pre,
        model_buffers.vision_encoder.model.ln_pre,
        .{ .absolute_tolerance = 2e-2 },
        .{ .batch, .n, .hidden },
    );

    // try ctx.testPositionEmbeddings(
    //     "model.model.vision_tower.patch_positional_embedding",
    //     mdl.vision_encoder.model.rope,
    //     .{ .absolute_tolerance = 2e-4 },
    //     .{.n},
    // );

    const vision_layer = mdl.vision_encoder.model.layers[0];
    const vision_buffers = model_buffers.vision_encoder.model.layers[0];

    try ctx.testLayerWithTags(
        try std.fmt.allocPrint(arena.allocator(), "model.model.vision_tower.transformer.layers.{d}.ffn_norm", .{i}),
        vision_layer.ffn_norm,
        vision_buffers.ffn_norm,
        .{ .absolute_tolerance = 2e-2 },
        .{ .batch, .n, .hidden },
    );

    // try ctx.testViTAttentionLayer(
    //     try std.fmt.allocPrint(arena.allocator(), "model.model.vision_tower.transformer.layers.{d}.attention", .{i}),
    //     vision_layer.self_attn,
    //     vision_buffers.self_attn,
    //     .{ .absolute_tolerance = 2e-2 },
    //     .{ .batch, .n, .hidden },
    // );

    try ctx.testLayerWithTags(
        try std.fmt.allocPrint(arena.allocator(), "model.model.vision_tower.transformer.layers.{d}.attention_norm", .{i}),
        vision_layer.norm_attn,
        vision_buffers.norm_attn,
        .{ .absolute_tolerance = 2e-2 },
        .{ .batch, .n, .hidden },
    );

    try ctx.testLayerWithTags(
        try std.fmt.allocPrint(arena.allocator(), "model.model.vision_tower.transformer.layers.{d}.feed_forward", .{i}),
        vision_layer.feed_fwd,
        vision_buffers.feed_fwd,
        .{ .absolute_tolerance = 2e-2 },
        .{ .batch, .n, .hidden },
    );

    try ctx.testLayerWithTags(
        "model.model.multi_modal_projector.norm",
        mdl.vision_encoder.lm_head.norm,
        model_buffers.vision_encoder.lm_head.norm,
        .{ .absolute_tolerance = 2e-2 },
        .{ .n, .v_hidden },
    );

    try ctx.testLayerWithTags(
        "model.model.multi_modal_projector.linear_1",
        mdl.vision_encoder.lm_head.w1,
        model_buffers.vision_encoder.lm_head.w1,
        .{ .absolute_tolerance = 2e-2 },
        .{ .h, .v_hidden },
    );

    try ctx.testLayerWithTags(
        "model.model.multi_modal_projector.linear_2",
        mdl.vision_encoder.lm_head.w2,
        model_buffers.vision_encoder.lm_head.w2,
        .{ .absolute_tolerance = 2e-2 },
        .{ .h, .i },
    );

    try ctx.testLayerWithTags(
        "model.model.multi_modal_projector.patch_merger",
        mdl.vision_encoder.lm_head.merger,
        model_buffers.vision_encoder.lm_head.merger,
        .{ .absolute_tolerance = 2e-2 },
        .{ .n, .v_hidden },
    );

    try ctx.testLayerWithTags(
        "model.model.multi_modal_projector",
        mdl.vision_encoder.lm_head,
        model_buffers.vision_encoder.lm_head,
        .{ .absolute_tolerance = 2e-2 },
        .{ .n, .v_hidden },
    );
}

const TestContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activation_store: zml.io.TensorStore.View,
    sharding: zml.sharding.Sharding,
    backend: zml.attention.attention.Backend,
    fn testLayer(
        self: TestContext,
        name: []const u8,
        layer: anytype,
        layer_weights: zml.Bufferized(@TypeOf(layer)),
        opts: zml.testing.CompareOpts,
    ) !void {
        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);
        const in_shape = self.activation_store.getShape(in_key) orelse return error.NotFound;
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_shape);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_weights, in_buffer });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();
        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
    }

    fn testLayerWithTags(
        self: TestContext,
        name: []const u8,
        layer: anytype,
        layer_weights: zml.Bufferized(@TypeOf(layer)),
        opts: zml.testing.CompareOpts,
        tags: anytype,
    ) !void {
        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);

        const in_shape = self.activation_store.getShape(in_key) orelse return error.NotFound;
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_shape).withTags(tags);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_weights, in_buffer });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();
        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
    }

    fn testPositionEmbeddings(
        self: TestContext,
        name: []const u8,
        layer: anytype,
        opts: zml.testing.CompareOpts,
        tags: anytype,
    ) !void {
        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.1", .{name});
        defer self.allocator.free(in_key);

        const in_shape = self.activation_store.getShape(in_key) orelse return error.NotFound;
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_shape).withTags(tags);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_cos_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, out_key, self.sharding);
        defer out_cos_buffer_expected.deinit();

        const out_sin_key = try std.fmt.allocPrint(self.allocator, "{s}.out.1", .{name});
        defer self.allocator.free(out_sin_key);
        var out_sin_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, out_sin_key, self.sharding);
        defer out_sin_buffer_expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{in_buffer});

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_cos, var out_sin = res.get(struct { zml.Buffer, zml.Buffer });
        defer out_cos.deinit();
        defer out_sin.deinit();

        try zml.testing.expectClose(self.io, out_cos, out_cos_buffer_expected, opts);
        try zml.testing.expectClose(self.io, out_sin, out_sin_buffer_expected, opts);
    }

    fn testAttentionLayer(
        self: TestContext,
        name: []const u8,
        layer: anytype,
        layer_weights: zml.Bufferized(@TypeOf(layer)),
        opts: zml.testing.CompareOpts,
        tags: anytype,
        backend: zml.attention.attention.Backend,
    ) !void {
        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);

        const in_shape = self.activation_store.getShape(in_key) orelse return error.NotFound;
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_shape).withTags(tags);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const seqlen = 11;
        const head_dim = 128;
        const attention_heads = 32;
        const num_key_value_heads = 8;

        const cache: ministral.Model.KvCache = .init(.init(.{
            .layer = 26,
            .batch = 1,
            .k = seqlen,
            .h = num_key_value_heads,
            .hd = head_dim,
        }, .bf16));

        // const token_index = zml.Tensor.iota(in_shape, 1);
        const token_index: zml.Tensor = .init(.{ .batch = 1 }, .u32);
        const cache_index: zml.Tensor = .init(.{}, .u32);

        const attention_metadata: zml.attention.attention.Metadata = .init(.fromBackend(backend, seqlen, attention_heads));
        const attention_params: zml.attention.attention.Parameters = .init(.fromBackend(backend));

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, token_index, cache, cache_index, attention_metadata, attention_params }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_weights, in_buffer });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();
        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
    }

    fn testViTAttentionLayer(
        self: TestContext,
        name: []const u8,
        layer: anytype,
        layer_weights: zml.Bufferized(@TypeOf(layer)),
        opts: zml.testing.CompareOpts,
        tags: anytype,
    ) !void {
        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);

        const in_shape = self.activation_store.getShape(in_key) orelse return error.NotFound;
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_shape).withTags(tags);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const seqlen = 11;
        const attention_heads = 64;

        const pos_embds = "model.model.vision_tower.patch_positional_embedding.in.1";
        const pos_shape = self.activation_store.getShape(pos_embds) orelse return error.NotFound;

        const token_index = zml.Tensor.fromShape(pos_shape);
        var token_index_buf = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, pos_embds, self.sharding);
        defer token_index_buf.deinit();

        const attention_metadata: zml.attention.attention.Metadata = .init(.fromBackend(self.backend, seqlen, attention_heads));
        const attention_params: zml.attention.attention.Parameters = .init(.fromBackend(self.backend));

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, token_index, attention_metadata, attention_params }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_weights, in_buffer, token_index_buf });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();
        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
    }
};
