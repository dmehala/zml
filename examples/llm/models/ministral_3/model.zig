const std = @import("std");

const zml = @import("zml");

const common = @import("../common.zig");
const inference = @import("inference.zig");

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

pub const Buffers = zml.Bufferized(Model);

pub const Model = struct {
    pub fn init(
        _: std.mem.Allocator,
        _: zml.io.TensorStore.View,
        _: Config,
        _: common.GenerationOptions,
    ) Model {
        return .{};
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        _ = self; // autofix
        _ = allocator; // autofix
    }

    pub fn loadBuffers(
        _: *const Model,
        _: std.mem.Allocator,
        _: std.Io,
        _: *const zml.Platform,
        _: *zml.io.TensorStore,
        _: *std.Progress.Node,
        _: common.Shardings,
    ) !zml.Bufferized(Model) {
        return error.loadBuffersNotImplemented;
    }

    pub fn unloadBuffers(_: *zml.Bufferized(Model), _: std.mem.Allocator) void {}
};

pub const LoadedModel = struct {
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        _: zml.io.TensorStore.View,
        _: common.GenerationOptions,
    ) !LoadedModel {
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        return .{
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, _: std.mem.Allocator) void {
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
        _ = self; // autofix
        _ = allocator; // autofix
        _ = io; // autofix
        _ = platform; // autofix
        _ = store; // autofix
        _ = progress; // autofix
        _ = shardings; // autofix
        return error.ModelNotImplemented;
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self; // autofix
        _ = buffers; // autofix
        _ = allocator; // autofix
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
