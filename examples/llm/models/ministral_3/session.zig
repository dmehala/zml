const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

const Self = @This();

allocator: std.mem.Allocator,

io: std.Io,

tokenizer: zml.tokenizer.Tokenizer,

compiled_model: *const inference.CompiledModel,

// generated_token_slice: zml.Slice,

pub fn init(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    tokenizer: zml.tokenizer.Tokenizer,
    compiled_model: *const inference.CompiledModel,
    model_buffers: *model.Buffers,
) !Self {
    _ = platform;
    _ = model_buffers;
    return .{
        .allocator = allocator,
        .io = io,
        .tokenizer = tokenizer,
        .compiled_model = compiled_model,
    };
}

pub fn deinit(_: *Self) void {}

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

pub fn runDecode(self: *Self, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
    _ = all_tokens;
    _ = stdout;

    // TODO: execute decoder kernel
    // self.compiled_model.decode_exe.args(self.allocator);

    var decoder = try self.tokenizer.decoder();
    defer decoder.deinit();

    // generation: while (true) {
    //     const token_id = self.generated_token_slice.items(u32)[0];
    //
    // }
    return error.SessionNotImplemented;
}

pub fn runPrefill(self: *Self, all_tokens: []const u32) !void {
    _ = self;
    _ = all_tokens;
    return error.SessionNotImplemented;
}
