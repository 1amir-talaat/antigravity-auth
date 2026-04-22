import { describe, expect, it, vi } from "vitest";
import {
  transformStreamingPayload,
  deduplicateThinkingText,
  cacheThinkingSignaturesFromResponse,
  createThoughtBuffer,
} from "./transformer";
import { createSignatureStore } from "../../stores/signature-store";

// ─── Helpers ──────────────────────────────────────────────────────────────────

function geminiResponse(parts: unknown[]) {
  return { candidates: [{ content: { role: "model", parts } }] };
}

function thinkingPart(text: string) {
  return { thought: true, text };
}

function textPart(text: string) {
  return { text };
}

// ─── transformStreamingPayload ────────────────────────────────────────────────

describe("transformStreamingPayload", () => {
  it("passes non-data lines through unchanged", () => {
    const line = "event: message";
    expect(transformStreamingPayload(line)).toBe(line);
  });

  it("passes empty data line unchanged", () => {
    expect(transformStreamingPayload("data: ")).toBe("data: ");
  });

  it("passes invalid JSON data line unchanged", () => {
    expect(transformStreamingPayload("data: {not json}")).toBe("data: {not json}");
  });

  it("passes data without response field unchanged", () => {
    const line = `data: ${JSON.stringify({ candidates: [] })}`;
    expect(transformStreamingPayload(line)).toBe(line);
  });

  it("applies transformThinkingParts to response field", () => {
    const inner = { type: "thinking", text: "reasoning" };
    const payload = { response: inner };
    const transform = vi.fn().mockReturnValue({ type: "redacted_thinking" });
    const result = transformStreamingPayload(`data: ${JSON.stringify(payload)}`, transform);
    expect(transform).toHaveBeenCalledWith(inner);
    expect(result).toContain("redacted_thinking");
  });
});

// ─── deduplicateThinkingText — Gemini candidates ──────────────────────────────

describe("deduplicateThinkingText", () => {
  it("returns null input unchanged", () => {
    expect(deduplicateThinkingText(null, createThoughtBuffer())).toBeNull();
  });

  it("passes non-thinking parts through", () => {
    const buf = createThoughtBuffer();
    const resp = geminiResponse([textPart("hello")]);
    const result = deduplicateThinkingText(resp, buf) as typeof resp;
    expect(result.candidates[0].content.parts).toEqual([textPart("hello")]);
  });

  it("emits full thinking text on first call", () => {
    const buf = createThoughtBuffer();
    const result = deduplicateThinkingText(geminiResponse([thinkingPart("hello")]), buf) as any;
    expect(result.candidates[0].content.parts[0].text).toBe("hello");
  });

  it("emits only the new delta on subsequent call with extended text", () => {
    const buf = createThoughtBuffer();
    deduplicateThinkingText(geminiResponse([thinkingPart("alpha")]), buf);
    const result = deduplicateThinkingText(geminiResponse([thinkingPart("alphabeta")]), buf) as any;
    expect(result.candidates[0].content.parts[0].text).toBe("beta");
  });

  it("filters out duplicate thinking when hash set is provided", () => {
    const buf = createThoughtBuffer();
    const seen = new Set<string>();
    const resp = geminiResponse([thinkingPart("same")]);
    deduplicateThinkingText(resp, buf, seen);
    const result2 = deduplicateThinkingText(resp, buf, seen) as any;
    const parts = result2.candidates[0].content.parts;
    expect(parts.some((p: any) => p.thought === true)).toBe(false);
  });
});

// ─── cacheThinkingSignaturesFromResponse ──────────────────────────────────────

describe("cacheThinkingSignaturesFromResponse", () => {
  it("accumulates thinking text in the thought buffer", () => {
    const store = createSignatureStore();
    const buf = createThoughtBuffer();
    cacheThinkingSignaturesFromResponse(geminiResponse([thinkingPart("my thoughts")]), "k", store, buf);
    expect(buf.get(0)).toBe("my thoughts");
  });

  it("fires onCacheSignature with session key, text, and signature", () => {
    const store = createSignatureStore();
    const buf = createThoughtBuffer();
    const onSig = vi.fn();
    cacheThinkingSignaturesFromResponse(
      geminiResponse([thinkingPart("reasoning"), { thoughtSignature: "sig-1" }]),
      "sess",
      store,
      buf,
      onSig,
    );
    expect(onSig).toHaveBeenCalledWith("sess", "reasoning", "sig-1");
  });

  it("stores result in signatureStore keyed by session key", () => {
    const store = createSignatureStore();
    const buf = createThoughtBuffer();
    cacheThinkingSignaturesFromResponse(
      geminiResponse([thinkingPart("t"), { thoughtSignature: "sig-2" }]),
      "session-a",
      store,
      buf,
    );
    expect(store.get("session-a")).toEqual({ text: "t", signature: "sig-2" });
  });

  it("skips firing onCacheSignature when no thinking text was accumulated", () => {
    const store = createSignatureStore();
    const buf = createThoughtBuffer();
    const onSig = vi.fn();
    cacheThinkingSignaturesFromResponse(geminiResponse([{ thoughtSignature: "sig" }]), "k", store, buf, onSig);
    expect(onSig).not.toHaveBeenCalled();
  });
});
