const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");

const log = std.log.scoped(.ministral3);

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

    pub fn unloadBuffers(self: *zml.Bufferized(LmHead)) void {
        RMSNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(self: *LmHead, hidden: zml.Tensor) zml.Tensor {
        const logits = self.norm.forward(hidden, .{.d});
        // TODO: implement sampling strategy
        return logits;
    }
};

const RMSNorm = struct {
    weights: zml.Tensor,
    eps: f32,

    pub fn unloadBuffers(self: *zml.Bufferized(RMSNorm)) void {
        self.weights.deinit();
    }

    pub fn forward(self: RMSNorm, hidden: zml.Tensor, tag: anytype) zml.Tensor {
        const normalized = zml.nn.rmsNorm(hidden, tag, self.eps);
        return normalized.mul(self.weights.broad(hidden.shape())).reuseBuffer(hidden);
    }
};

const Layer = struct {
    const Mlp = struct {
        down_proj: zml.Tensor,
        gate_proj: zml.Tensor,
        up_proj: zml.Tensor,

        pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
            self.down_proj.deinit();
            self.gate_proj.deinit();
            self.up_proj.deinit();
        }

        pub fn forward(self: *Mlp, x: zml.Tensor) zml.Tensor {
            // def mlp(x):
            // gate = x @ W_gate.T
            // up   = x @ W_up.T
            //
            // # SwiGLU activation
            // activated = silu(gate) * up
            //
            // out = activated @ W_down.T
            //
            // return out

            const up = self.up_proj.forward(x);
            var activated = self.gate_proj.forward(x);
            activated = activated.silu().mul(up);
            return self.down_proj.forward(activated);
        }
    };

    const Attention = struct {
        k_proj: zml.Tensor,
        q_proj: zml.Tensor,
        v_proj: zml.Tensor,
        o_proj: zml.Tensor,

        pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
            self.k_proj.deinit();
            self.q_proj.deinit();
            self.v_proj.deinit();
            self.o_proj.deinit();
        }

        pub fn forward(
            self: *Attention,
            x: zml.Tensor,
            token_index: zml.Tensor,
            kv_cache: KvCache,
            attention_metadata: zml.attention.attention.Metadata,
            attention_parameters: zml.attention.attention.Parameters,
        ) zml.Tensor {
            _ = kv_cache; // autofix
            // def self_attention(x):
            //     # Projections
            //     Q = x @ Wq.T
            //     K = x @ Wk.T
            //     V = x @ Wv.T
            //
            //     # reshape to heads
            //     Q = reshape_heads(Q)
            //     K = reshape_heads(K)
            //     V = reshape_heads(V)
            //
            //     # (optional) apply RoPE here
            //     Q, K = apply_rope(Q, K)
            //
            //     # ---- KV cache logic ----
            //     K_cache = concat(prev_K, K)
            //     V_cache = concat(prev_V, V)
            //
            //     # Attention scores
            //     scores = Q @ K_cache.transpose(-1, -2)
            //     scores = scores / sqrt(head_dim)
            //
            //     scores = causal_mask(scores)
            //
            //     probs = softmax(scores)
            //
            //     # Weighted sum
            //     context = probs @ V_cache
            //
            //     # merge heads
            //     context = merge_heads(context)
            //
            //     # Output projection
            //     out = context @ Wo.T
            //
            //     return out
            var Q = self.q_proj.forward(x);
            var K = self.k_proj.forward(x);
            const V = self.v_proj.forward(x);

            // TODO: reshape and apply rope
            const pos_index = token_index;
            Q = zml.nn.rope(Q, pos_index, self.rope_opts);
            K = zml.nn.rope(K, pos_index, self.rope_opts);
            const attn_output = zml.attention.attention.attention(
                Q,
                K,
                V,
                token_index,
                attention_metadata,
                attention_parameters,
            );

            const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
            const delta = self.o_proj.forward(attn)
                .rename(.{ .dout = .d })
                .withPartitioning(.{ .d = .replicated });
            // return .{ delta, new_kv_cache };
            return delta;
        }
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
                .k_proj = attn_store.withPrefix("k_proj").createTensor("weight", .{ .e, .f }, null),
                .q_proj = attn_store.withPrefix("q_proj").createTensor("weight", .{ .g, .h }, null),
                .v_proj = attn_store.withPrefix("v_proj").createTensor("weight", .{ .i, .j }, null),
                .o_proj = attn_store.withPrefix("o_proj").createTensor("weight", .{ .k, .l }, null),
            },
            .post_attn = .{
                .weights = store.withPrefix("post_attention_layernorm").createTensor("weight", .{.post_attn}, null),
                .eps = config.text_config.rms_norm_eps,
            },
            .feed_fwd = .{
                .down_proj = mlp_store.withPrefix("down_proj").createTensor("weight", .{ .m, .n }, null),
                .gate_proj = mlp_store.withPrefix("gate_proj").createTensor("weight", .{ .o, .p }, null),
                .up_proj = mlp_store.withPrefix("up_proj").createTensor("weight", .{ .q, .r }, null),
            },
        };
    }

    pub fn deinit(_: *const Layer, _: std.mem.Allocator) void {}

    pub fn unloadBuffers(self: *zml.Bufferized(Layer)) void {
        RMSNorm.unloadBuffers(&self.input_norm);
        Attention.unloadBuffers(&self.self_attn);
        RMSNorm.unloadBuffers(&self.post_attn);
        Mlp.unloadBuffers(&self.feed_fwd);
    }

    pub fn forward(
        self: *Layer,
        x: zml.Tensor,
        tokens_index: zml.Tensor,
        cur_seq_len: zml.Tensor,
        cache: KvCache,
        cache_index: zml.Tensor,
        conv_cache_index: zml.Tensor,
        kv_cache_index: zml.Tensor,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
        conv_parameters: ConvParameters,
    ) struct { zml.Tensor, KvCache, zml.Tensor, zml.Tensor } {
        _ = cur_seq_len; // autofix
        _ = cache_index; // autofix
        _ = conv_parameters; // autofix
        const attn_norm = self.input_norm.forward(x, .{.a});
        const attn = self.self_attn.forward(attn_norm, tokens_index, cache, attention_metadata, attention_parameters);
        x = x + attn; // residual

        const feed_norm = self.post_attn.forward(x, .{.b});
        const feed_fwd = self.feed_fwd.forward(feed_norm);
        x = x + feed_fwd; // residual

        return .{ x, cache, conv_cache_index, kv_cache_index };
    }
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
    allocator.free(self.layers);
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

pub fn unloadBuffers(self: *zml.Bufferized(Self), allocator: std.mem.Allocator) void {
    self.embed_tokens.deinit();
    for (self.layers) |*layer| {
        Layer.unloadBuffers(layer);
    }
    LmHead.unloadBuffers(&self.lm_head);
    allocator.free(self.layers);
}

const KvCache = struct {};

const ConvParameters = struct {};

pub fn forward(
    self: Self,
    tokens: zml.Tensor,
    tokens_index: zml.Tensor,
    cur_seq_len: zml.Tensor,
    kv_cache: KvCache,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,
    conv_parameters: ConvParameters,
) zml.Tensor {
    var cache = kv_cache;
    var conv_cache_index = zml.Tensor.scalar(@as(u32, 0), .u32);
    var kv_cache_index = zml.Tensor.scalar(@as(u32, 0), .u32);

    var hidden = self.embed_tokens.forward(tokens);

    for (self.layers) |layer| {
        hidden, cache, conv_cache_index, kv_cache_index = layer.forward(
            hidden,
            tokens_index,
            cur_seq_len,
            cache,
            conv_cache_index,
            kv_cache_index,
            attention_metadata,
            attention_parameters,
            conv_parameters,
        );
    }

    const next_tokens = self.lm_head.forward(hidden);
    return next_tokens;
}
