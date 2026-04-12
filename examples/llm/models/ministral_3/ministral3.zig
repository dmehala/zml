const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const common = @import("../common.zig");

const log = std.log.scoped(.ministral3);

const TextConfig = struct {
    attention_dropout: f32,
    head_dim: u32,
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
    rope_parameters: zml.nn.RopeOpts,
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

embed_tokens: zml.nn.TokenEmbedding,

layers: []Layer,

// Final layer
norm_head: RMSNorm,

lm_head: LmHead,

sampling: zml.nn.SamplingStrategy,

const LmHead = struct {
    weight: zml.Tensor,

    pub fn unloadBuffers(self: *zml.Bufferized(LmHead)) void {
        self.weight.deinit();
    }

    pub fn forward(self: LmHead, hidden: zml.Tensor) zml.Tensor {
        return hidden.dot(self.weight, .hidden);
    }
};

const RMSNorm = struct {
    weights: zml.Tensor,
    eps: f32,
    tag: zml.Shape.Tag,

    pub fn unloadBuffers(self: *zml.Bufferized(RMSNorm)) void {
        self.weights.deinit();
    }

    pub fn forward(self: RMSNorm, x: zml.Tensor) zml.Tensor {
        const normalized = zml.nn.rmsNorm(x, self.tag, self.eps);
        return normalized.mul(self.weights.broad(x.shape())).reuseBuffer(x);
    }
};

const Layer = struct {
    const Mlp = struct {
        down_proj: zml.nn.Linear,
        gate_proj: zml.nn.Linear,
        up_proj: zml.nn.Linear,

        pub fn init(store: zml.io.TensorStore.View) Mlp {
            return .{
                .down_proj = .init(store.withPrefix("down_proj").createTensor("weight", .{ .hidden, .dout }, null), null, .dout),
                .gate_proj = .init(store.withPrefix("gate_proj").createTensor("weight", .{ .dout, .hidden }, null), null, .hidden),
                .up_proj = .init(store.withPrefix("up_proj").createTensor("weight", .{ .dout, .hidden }, null), null, .hidden),
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
            self.down_proj.weight.deinit();
            self.gate_proj.weight.deinit();
            self.up_proj.weight.deinit();
        }

        pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
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
        k_proj: zml.nn.Linear,
        q_proj: zml.nn.Linear,
        v_proj: zml.nn.Linear,
        o_proj: zml.nn.Linear,
        rope_opts: zml.nn.RopeOpts,
        head_dim: u32,
        num_kv_heads: u32,

        pub fn init(config: Config, store: zml.io.TensorStore.View) Attention {
            return .{
                .k_proj = .init(store.withPrefix("k_proj").createTensor("weight", .{ .e, .hidden }, null), null, .hidden),
                .q_proj = .init(store.withPrefix("q_proj").createTensor("weight", .{ .dout, .hidden }, null), null, .hidden),
                .v_proj = .init(store.withPrefix("v_proj").createTensor("weight", .{ .e, .hidden }, null), null, .hidden),
                .o_proj = .init(store.withPrefix("o_proj").createTensor("weight", .{ .hidden, .d }, null), null, .d),
                .rope_opts = config.text_config.rope_parameters,
                .head_dim = config.text_config.head_dim,
                .num_kv_heads = config.text_config.num_key_value_heads,
            };
        }

        pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
            self.k_proj.weight.deinit();
            self.q_proj.weight.deinit();
            self.v_proj.weight.deinit();
            self.o_proj.weight.deinit();
        }

        pub fn forward(
            self: Attention,
            x: zml.Tensor,
            token_index: zml.Tensor,
            kv_cache: KvCache,
            cache_index: zml.Tensor,
            attention_metadata: zml.attention.attention.Metadata,
            attention_parameters: zml.attention.attention.Parameters,
        ) struct { zml.Tensor, KvCache } {
            // splitAxis for multi head attention.
            var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = .auto, .hd = self.head_dim });
            var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim });
            var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = self.num_kv_heads, .hd = self.head_dim });

            const token_positions = b: {
                const sh = token_index.shape().insert(.last, .{ .seq = x.dim(.seq) });
                break :b zml.Tensor.iota(sh, .seq).convert(.u32).add(token_index.broad(sh));
            };

            q = zml.nn.rope(q, token_positions, self.rope_opts);
            k = zml.nn.rope(k, token_positions, self.rope_opts);

            // Rename to match kvcache tags
            q = q.rename(.{ .seq = .q });
            k = k.rename(.{ .seq = .k });
            v = v.rename(.{ .seq = .k });

            const new_kv_cache = kv_cache.update(k, v, token_index, cache_index);
            k = new_kv_cache.keys(cache_index);
            v = new_kv_cache.values(cache_index);

            const attn_scores = zml.attention.attention.attention(
                q,
                k,
                v,
                token_index,
                attention_metadata,
                attention_parameters,
            );

            const attn_heads = attn_scores.merge(.{ .d = .{ .h, .hd } });
            return .{ self.o_proj.forward(attn_heads).reuseBuffer(x), new_kv_cache.reuseBuffer(kv_cache) };
        }
    };

    input_norm: RMSNorm,
    self_attn: Attention,
    post_attn: RMSNorm,
    feed_fwd: Mlp,

    pub fn init(config: Config, store: zml.io.TensorStore.View) Layer {
        const mlp_store = store.withPrefix("mlp");
        const attn_store = store.withPrefix("self_attn");

        return .{
            .input_norm = .{
                .weights = store.withPrefix("input_layernorm").createTensor("weight", .{.hidden}, null),
                .eps = config.text_config.rms_norm_eps,
                .tag = zml.Shape.toTag(.hidden),
            },
            .self_attn = .init(config, attn_store),
            .post_attn = .{
                .weights = store.withPrefix("post_attention_layernorm").createTensor("weight", .{.hidden}, null),
                .eps = config.text_config.rms_norm_eps,
                .tag = zml.Shape.toTag(.hidden),
            },
            .feed_fwd = .init(mlp_store),
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
        cache: KvCache,
        cache_index: zml.Tensor,
        attention_metadata: zml.attention.attention.Metadata,
        attention_parameters: zml.attention.attention.Parameters,
    ) struct { zml.Tensor, KvCache, zml.Tensor } {
        const attn_norm = self.input_norm.forward(x);
        const attn, const cache_ = self.self_attn.forward(attn_norm, tokens_index, cache, cache_index, attention_metadata, attention_parameters);
        var x_ = x.add(attn); // residual

        const feed_norm = self.post_attn.forward(x_);
        const feed_fwd = self.feed_fwd.forward(feed_norm);
        x_ = x_.add(feed_fwd); // residual

        const cache_index_ = cache_index.add(zml.Tensor.scalar(@as(u32, 1), .u32));
        return .{ x_, cache_, cache_index_ };
    }
};

pub fn init(
    allocator: std.mem.Allocator,
    store: zml.io.TensorStore.View,
    config: Config,
    opts: common.GenerationOptions,
) !Self {
    const model_store = store.withPrefix("language_model.model");

    // TODO: Use `maybeCreateTensor`.
    const embed_tokens = model_store.withPrefix("embed_tokens").createTensor("weight", .{ .voc, .hidden }, null);
    const norm = model_store.withPrefix("norm").createTensor("weight", .{.hidden}, null);

    stdx.debug.assert(config.text_config.num_hidden_layers != 0, "expected at least one layer", .{});
    const layers = try allocator.alloc(Layer, config.text_config.num_hidden_layers);
    for (layers, 0..) |*layer, i| {
        const layer_store = model_store.withPrefix("layers").withLayer(i);
        layer.* = Layer.init(config, layer_store);
    }

    return .{
        .embed_tokens = .{ .weight = embed_tokens },
        .layers = layers,
        .norm_head = .{
            .weights = norm,
            .eps = config.text_config.rms_norm_eps,
            .tag = zml.Shape.toTag(.hidden),
        },
        .lm_head = .{ .weight = embed_tokens },
        .sampling = opts.sampling_strategy,
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
    self.embed_tokens.weight.deinit();
    for (self.layers) |*layer| {
        Layer.unloadBuffers(layer);
    }
    LmHead.unloadBuffers(&self.lm_head);
    allocator.free(self.layers);
}

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,

    pub fn init(config: Config, model: Self, batch: usize, seqlen: usize) KvCache {
        const kv_shape: zml.Shape = .init(.{
            .layer = config.text_config.num_hidden_layers,
            .batch = batch,
            .k = seqlen,
            .h = config.text_config.num_key_value_heads,
            .hd = config.text_config.head_dim,
        }, model.embed_tokens.weight.dtype());

        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
        };
    }

    pub fn initBuffers(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), sharding, .{}),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
        };
    }

    pub fn keys(self: KvCache, cache_index: zml.Tensor) zml.Tensor {
        return self.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache, cache_index: zml.Tensor) zml.Tensor {
        return self.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = cache_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_position: zml.Tensor, cache_index: zml.Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        const layer = cache_index.broad(token_position.shape());
        return .{
            .k = self.k.scatterSlices(.{ .layer = layer, .k = token_position }, new_k.transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
            .v = self.v.scatterSlices(.{ .layer = layer, .k = token_position }, new_v.transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
        };
    }
};

pub fn forward(
    self: Self,
    tokens: zml.Tensor,
    tokens_index: zml.Tensor,
    rng: zml.Tensor.Rng,
    kv_cache: KvCache,
    attention_metadata: zml.attention.attention.Metadata,
    attention_parameters: zml.attention.attention.Parameters,
) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
    stdx.debug.assert(tokens.shape().hasTags(.{ .batch, .seq }), "Tokens should have tags {{.batch, .seq}}, got {f}", .{tokens.shape()});

    var cache = kv_cache;
    var kv_cache_index = zml.Tensor.scalar(@as(u32, 0), .u32);

    var hidden = self.embed_tokens.forward(tokens).renameTag(.d, .hidden);

    for (self.layers) |*layer| {
        hidden, cache, kv_cache_index = layer.forward(
            hidden,
            tokens_index,
            cache,
            kv_cache_index,
            attention_metadata,
            attention_parameters,
        );
    }

    hidden = self.norm_head.forward(hidden);
    const logits = self.lm_head.forward(hidden);

    const gen_tokens, const new_rng = zml.nn.sampleTokens(logits, self.sampling, rng);
    return .{ gen_tokens.convert(tokens.dtype()).reuseBuffer(tokens), cache.reuseBuffer(kv_cache), new_rng };
}
