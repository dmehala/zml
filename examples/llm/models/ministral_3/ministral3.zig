const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");

const log = std.log.scoped(.lfm);

const RopeParameters = struct {
    beta_fast: f32,
    beta_slow: f32,
    factor: f32,
    llama_4_scaling_beta: f32,
    mscale: f32,
    mscale_all_dim: f32,
    original_max_position_embeddings: u32,
    rope_theta: f32,
    rope_type: []const u8,
    type: []const u8,
};

const TextConfig = struct {
    attention_dropout: f32,
    head_dim: u8,
    hidden_act: []const u8,
    hidden_size: u32,
    initializer_range: f32,
    intermediate_size: u32,
    max_position_embeddings: u32,
    model_type: []const u8,
    num_attention_heads: u8,
    num_hidden_layers: u8,
    num_key_value_heads: u8,
    rms_norm_eps: f32,
    rope_parameters: RopeParameters,
    // sliding_window: ?i64,
    tie_word_embeddings: bool,
    use_cache: bool,
    vocab_size: u32,
};

pub const Config = struct {
    architectures: []const []const u8,
    dtype: []const u8,
    image_token_index: u8,
    model_type: []const u8,
    multimodal_projector_bias: bool,
    projector_hidden_act: []const u8,
    spatial_merge_size: u8,
    text_config: TextConfig,
    transformers_version: []const u8,
};

const Self = @This();

// Entry layer: maps token IDs -> dense vectors
embed_tokens: zml.Tensor,

layers: []Layer,

// Final layer
lm_head: LmHead,

const LmHead = struct {
    norm: RMSNorm,
};

const RMSNorm = struct {
    weights: zml.Tensor,
    eps: f32,
};

const Layer = struct {
    const Mlp = struct {
        down_proj: zml.Tensor,
        gate_proj: zml.Tensor,
        up_proj: zml.Tensor,
    };

    const Attention = struct {
        k_proj: zml.Tensor,
        q_proj: zml.Tensor,
        v_proj: zml.Tensor,
        o_proj: zml.Tensor,
    };

    input_norm: RMSNorm,
    self_attn: Attention,
    post_attn: RMSNorm,
    feed_fwd: Mlp,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Layer {
        const mlp_store = store.withPrefix("mlp");
        const attn_store = store.withPrefix("self_attn");

        return .{
            .input_norm = .{
                .weights = store.withPrefix("input_layernorm").createTensor("weight", .{.d}, null),
                .eps = config.text_config.rms_norm_eps,
            },
            .self_attn = .{
                .k_proj = attn_store.withPrefix("k_proj").createTensor("weight", .{.k_proj}, null),
                .q_proj = attn_store.withPrefix("q_proj").createTensor("weight", .{.q_proj}, null),
                .v_proj = attn_store.withPrefix("v_proj").createTensor("weight", .{.v_proj}, null),
                .o_proj = attn_store.withPrefix("o_proj").createTensor("weight", .{.o_proj}, null),
            },
            .post_attn = .{
                .weights = store.withPrefix("post_attention_layernorm").createTensor("weight", .{.post_attn}, null),
                .eps = config.text_config.rms_norm_eps,
            },
            .feed_fwd = .{
                .down_proj = mlp_store.withPrefix("down_proj").createTensor("weight", .{.down_proj}, null),
                .gate_proj = mlp_store.withPrefix("gate_proj").createTensor("weight", .{.gate_proj}, null),
                .up_proj = mlp_store.withPrefix("up_proj").createTensor("weight", .{.up_proj}, null),
            },
        };
    }

    pub fn deinit(_: *const Layer, _: std.mem.Allocator) void {}
};

pub fn init(
    allocator: std.mem.Allocator,
    store: zml.io.TensorStore.View,
    config: Config,
    _: common.GenerationOptions,
) !Self {
    const model_store = store.withPrefix("language_model.model");

    // TODO: Use `maybeCreateTensor`.
    const embed_tokens = model_store.withPrefix("embed_tokens").createTensor("weight", .{ .a, .b }, null);
    const norm_weight = model_store.withPrefix("norm").createTensor("weight", .{.c}, null);

    stdx.debug.assert(config.text_config.num_hidden_layers != 0, "expected at least one layer", .{});
    const layers = try allocator.alloc(Layer, config.text_config.num_hidden_layers);
    for (layers, 0..) |*layer, i| {
        const layer_store = model_store.withPrefix("layers").withLayer(i);
        layer.* = Layer.init(layer_store, config);
    }

    return .{
        .embed_tokens = embed_tokens,
        .layers = layers,
        .lm_head = .{
            .norm = .{
                .weights = norm_weight,
                .eps = config.text_config.rms_norm_eps,
            },
        },
    };
}

pub fn deinit(self: Self, allocator: std.mem.Allocator) void {
    for (self.layers) |layer| {
        layer.deinit(allocator);
    }
}

pub fn loadBuffers(
    self: *const Self,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    shardings: common.Shardings,
) !zml.Bufferized(Self) {
    const now: std.Io.Timestamp = .now(io, .awake);
    var total_bytes: usize = 0;
    defer {
        const took = now.untilNow(io, .awake);
        const took_ns: usize = @max(1, @as(usize, @intCast(took.toNanoseconds())));
        log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{
            total_bytes,
            took,
            total_bytes * std.time.ns_per_s / took_ns,
        });
    }

    const all_shardings = shardings.all();
    return zml.io.load(Self, self, allocator, io, platform, store, .{
        .dma_chunks = 8,
        .dma_chunk_size = 128 * zml.MiB,
        .progress = progress,
        .parallelism = 16,
        .total_bytes = &total_bytes,
        .shardings = &all_shardings,
    });
}

pub fn unloadBuffers(_: *zml.Bufferized(Self), _: std.mem.Allocator) void {}
