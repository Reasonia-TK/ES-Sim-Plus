import { afterEach, describe, expect, it, vi } from "vitest";
import { parseHealth, useConnection } from "../src/backend/connection";
import { parsePort } from "../src/backend/port";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

afterEach(() => {
  vi.unstubAllGlobals();
  useConnection.setState({ status: "connecting", info: null, schema: null, lastError: null });
});

describe("backend connection", () => {
  it("parses /health", () => {
    expect(parseHealth({ status: "ok", version: "0.1.42", gpu: true, numba: false })).toMatchObject({
      version: "0.1.42",
      gpu: true,
      numba: false,
    });
    expect(parseHealth({ status: "ok", version: "1" }).numba).toBe(true);
    expect(() => parseHealth({ status: "down" })).toThrow();
  });

  it("validates ports", () => {
    expect(parsePort("8317")).toBe(8317);
    expect(parsePort(" 80 ")).toBe(80);
    expect(parsePort("0")).toBeNull();
    expect(parsePort("70000")).toBeNull();
    expect(parsePort("80a")).toBeNull();
  });

  it("connects, fetches the schema once, and notices when the backend goes away", async () => {
    const calls: string[] = [];
    let up = true;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        calls.push(new URL(url).pathname);
        if (!up) throw new TypeError("Failed to fetch");
        if (url.endsWith("/health")) return jsonResponse({ status: "ok", version: "0.1.42", gpu: false, numba: true });
        if (url.endsWith("/v2/schema")) return jsonResponse({ version: "0.1.42", project: { properties: {} } });
        return jsonResponse({ detail: "not found" }, 404);
      }),
    );
    await useConnection.getState().check();
    expect(useConnection.getState().status).toBe("connected");
    expect(useConnection.getState().schema?.version).toBe("0.1.42");
    await useConnection.getState().check();
    expect(calls).toEqual(["/health", "/v2/schema", "/health"]);

    up = false;
    await useConnection.getState().check();
    expect(useConnection.getState().status).toBe("disconnected");
    expect(useConnection.getState().lastError).toMatch(/fetch/);

    up = true;
    await useConnection.getState().check();
    expect(useConnection.getState().status).toBe("connected");
    expect(calls.slice(-2)).toEqual(["/health", "/v2/schema"]);
  });

  it("stays connected to an old backend without /v2/schema", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        url.endsWith("/health") ? jsonResponse({ status: "ok", version: "0.1.0", gpu: false }) : jsonResponse({ detail: "Not Found" }, 404),
      ),
    );
    await useConnection.getState().check();
    expect(useConnection.getState().status).toBe("connected");
    expect(useConnection.getState().schema).toBeNull();
    expect(useConnection.getState().lastError).toMatch(/Not Found/);
  });
});
