const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const inference = @import("inference.zig");
const Ministral3 = @import("ministral3.zig");
const model = @import("model.zig");

const Self = @This();

const log = std.log.scoped(.session);

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

pub fn tokenizePrompt(self: *const Self, allocator: std.mem.Allocator, prompt: []const u8, img_len: u32) ![]const u32 {
    var encoder = try self.tokenizer.encoder();
    defer encoder.deinit();

    const bos_token = self.tokenizer.tokenToId("<s>") orelse return error.NoSuchToken;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len + img_len);
    try tokens.append(allocator, bos_token);

    if (img_len > 0) {
        const img_bos_token = self.tokenizer.tokenToId("[IMG]") orelse return error.NoSuchToken;
        const img_eos_token = self.tokenizer.tokenToId("[IMG_END]") orelse return error.NoSuchToken;

        try tokens.appendNTimes(allocator, img_bos_token, img_len);
        try tokens.append(allocator, img_eos_token);
    }

    try tokens.appendSlice(allocator, try encoder.encode(prompt));

    return tokens.toOwnedSlice(allocator);
}

pub fn tokenizeTurn(self: *const Self, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
    return self.tokenizePrompt(allocator, prompt, 0);
}

pub fn runPrefill(self: *Self, tokens: []const u32) !void {
    stdx.debug.assert(self.seqlen > tokens.len, "input tokens of size {d} exceed seqlen size of {d}", .{ tokens.len, self.seqlen });

    const tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{ .batch = 1, .seq = self.seqlen }, .u32));
    defer tokens_slice.free(self.allocator);

    @memcpy(tokens_slice.items(u32)[0..tokens.len], tokens);

    const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);

    var tokens_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, tokens_slice, replicated_sharding);
    defer tokens_buffer.deinit();

    // try self.compiled_model.embeddings.run_embeddings(.{
    //     .allocator = self.allocator,
    //     .model_buffers = self.model_buffers,
    //     .tokens_buf = &tokens_buffer,
    // });

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
    self.generated_token_slice.items(u32)[0] = tokens_slice.items(u32)[tokens.len - 1];
}

pub fn runDecode(self: *Self, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
    var decoder = try self.tokenizer.decoder();
    defer decoder.deinit();

    const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);

    var token_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, self.generated_token_slice, replicated_sharding);
    defer token_buffer.deinit();

    const end_token = self.tokenizer.tokenToId("</s>");

    generation: while (true) {
        const token_id = self.generated_token_slice.items(u32)[0];

        if (token_id == end_token) break :generation;

        if (try decoder.next(token_id)) |token| {
            try stdout.writeAll(token);
            try stdout.flush();
        }

        try all_tokens.append(self.allocator, token_id);
        if (all_tokens.items.len >= self.seqlen) break :generation;

        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(all_tokens.items.len)}));
        var token_pos_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, replicated_sharding);
        defer token_pos_buffer.deinit();

        // try self.compiled_model.embeddings.run_embeddings(.{
        //     .allocator = self.allocator,
        //     .model_buffers = self.model_buffers,
        //     .tokens_buf = &token_buffer,
        // });

        try self.compiled_model.decode.run(.{
            .allocator = self.allocator,
            .model_buffers = self.model_buffers,
            .tokens_buf = &token_buffer,
            .tokens_pos_buffer = &token_pos_buffer,
            .rng_buffer = &self.rng_buffers,
            .kv_cache_buffers = &self.kv_cache_buffers,
            .attention_metadata_buffers = self.attention_metadata_buffers,
        });

        try token_buffer.toSlice(self.io, self.generated_token_slice);
    }
}
