const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const Ministral3 = @import("ministral3.zig");
const model = @import("model.zig");

const Self = @This();

allocator: std.mem.Allocator,

io: std.Io,

platform: *const zml.Platform,

rng_buffers: zml.Bufferized(zml.Tensor.Rng),

model_buffers: *model.Buffers,

kv_cache_buffers: zml.Bufferized(Ministral3.KvCache),

attention_metadata_buffers: zml.Bufferized(zml.attention.attention.Metadata),

tokenizer: zml.tokenizer.Tokenizer,

compiled_model: *const inference.CompiledModel,

generated_token_slice: zml.Slice,

seqlen: usize,

pub fn init(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    tokenizer: zml.tokenizer.Tokenizer,
    compiled_model: *const inference.CompiledModel,
    model_buffers: *model.Buffers,
) !Self {
    const shardings = compiled_model.params.shardings;

    var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffers(io, platform, shardings.model);
    errdefer Ministral3.KvCache.unloadBuffers(&kv_cache_buffers);

    var attention_metadata_buffers = try compiled_model.params.attention_metadata.initBuffer(io, platform, shardings.model);
    errdefer zml.attention.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

    const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());

    return .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, compiled_model.params.shardings.replicated),
        .model_buffers = model_buffers,
        .kv_cache_buffers = kv_cache_buffers,
        .attention_metadata_buffers = attention_metadata_buffers,
        .tokenizer = tokenizer,
        .compiled_model = compiled_model,
        .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .batch = 1, .seq = 1 }, .u32)),
        .seqlen = compiled_model.params.seqlen,
    };
}

pub fn deinit(self: *Self) void {
    Ministral3.KvCache.unloadBuffers(&self.kv_cache_buffers);
    zml.attention.attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
    zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
    self.generated_token_slice.free(self.allocator);
}

pub fn tokenizePrompt(self: *const Self, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
    var encoder = try self.tokenizer.encoder();
    defer encoder.deinit();

    // TODO: read system prompt from repo and add it
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
    try tokens.appendSlice(allocator, try encoder.encode(prompt));

    return tokens.toOwnedSlice(allocator);
}

pub fn tokenizeTurn(self: *const Self, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
    return self.tokenizePrompt(allocator, prompt);
}

pub fn runPrefill(self: *Self, all_tokens: []const u32) !void {
    const tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{ .batch = 1, .seq = self.seqlen }, .u32));
    defer tokens_slice.free(self.allocator);

    // TODO: Make sure all_tokens < seqlen?
    @memcpy(tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

    const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);

    var tokens_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, tokens_slice, replicated_sharding);
    defer tokens_buffer.deinit();

    const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{0}));
    var token_pos_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, replicated_sharding);
    defer token_pos_buffer.deinit();

    try self.compiled_model.prefill.run(.{
        .allocator = self.allocator,
        .model_buffers = self.model_buffers,
        .tokens_buf = &tokens_buffer,
        .tokens_pos_buffer = &token_pos_buffer,
        .rng_buffer = &self.rng_buffers,
        .kv_cache_buffers = &self.kv_cache_buffers,
        .attention_metadata_buffers = self.attention_metadata_buffers,
    });

    try tokens_buffer.toSlice(self.io, tokens_slice);
    self.generated_token_slice.items(u32)[0] = tokens_slice.items(u32)[all_tokens.len - 1];
}

pub fn runDecode(self: *Self, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
    var decoder = try self.tokenizer.decoder();
    defer decoder.deinit();

    const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);

    var current_token_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, self.generated_token_slice, replicated_sharding);
    defer current_token_buffer.deinit();

    while (true) {
        const token_id = self.generated_token_slice.items(u32)[0];

        if (try decoder.next(token_id)) |token| {
            try stdout.writeAll(token);
            try stdout.flush();
        }

        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(all_tokens.items.len)}));
        var token_pos_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, replicated_sharding);
        defer token_pos_buffer.deinit();

        try self.compiled_model.decode.run(.{
            .allocator = self.allocator,
            .model_buffers = self.model_buffers,
            .tokens_buf = &current_token_buffer,
            .tokens_pos_buffer = &token_pos_buffer,
            .rng_buffer = &self.rng_buffers,
            .kv_cache_buffers = &self.kv_cache_buffers,
            .attention_metadata_buffers = self.attention_metadata_buffers,
        });

        try current_token_buffer.toSlice(self.io, self.generated_token_slice);
    }
}
