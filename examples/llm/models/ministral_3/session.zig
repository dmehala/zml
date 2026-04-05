const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

pub const Session = struct {
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        _ = allocator;
        _ = io;
        _ = platform;
        _ = tokenizer;
        _ = compiled_model;
        _ = model_buffers;
        return error.SessionNotImplemented;
    }

    pub fn deinit(_: *Session) void {}

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        _ = self;
        _ = allocator;
        _ = prompt;
        return error.SessionNotImplemented;
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        _ = self;
        _ = allocator;
        _ = prompt;
        return error.SessionNotImplemented;
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        _ = self;
        _ = all_tokens;
        _ = stdout;
        return error.SessionNotImplemented;
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        _ = self;
        _ = all_tokens;
        return error.SessionNotImplemented;
    }
};
