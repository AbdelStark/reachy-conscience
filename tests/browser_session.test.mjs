import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { runInNewContext } from "node:vm";

const source = readFileSync(new URL("../src/reachy_conscience/assets/app.js", import.meta.url), "utf8");

function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}

function element() {
  return {
    value: "", textContent: "", hidden: false, disabled: false, children: [],
    classList: { toggle() {} },
    addEventListener() {},
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
  };
}

function response(body) {
  return { ok: true, json: async () => body };
}

function consoleHarness(fetch) {
  const nodes = new Map();
  const byId = (id) => {
    if (!nodes.has(id)) nodes.set(id, element());
    return nodes.get(id);
  };
  const context = {
    document: {
      getElementById: byId,
      createElement: element,
      createTextNode: (text) => ({ textContent: text }),
      querySelectorAll: () => [],
    },
    fetch,
    setInterval() {},
    window: { confirm: () => true },
  };
  runInNewContext(source, context);
  return { context, byId };
}

const policy = {
  rules: "private policy", block_threshold: 0.7, hold_floor: 0.3,
  registered_tools: [], confirm_before: [], preview_available: false,
  approval_available: false,
};

test("disconnect clears owner data and ignores a late policy body", async () => {
  const pendingPolicy = deferred();
  const { context, byId } = consoleHarness((path) => {
    if (path === "/api/policy") return Promise.resolve({ ok: true, json: () => pendingPolicy.promise });
    return Promise.resolve(response({ rows: [] }));
  });
  byId("token").value = "test-owner-token";
  const connecting = context.connect();
  await Promise.resolve();
  context.disconnect();
  pendingPolicy.resolve(policy);
  await connecting;
  assert.equal(byId("workspace").hidden, true);
  assert.equal(byId("rules").value, "");
  assert.equal(byId("ledger-rows").children.length, 0);
  assert.equal(byId("status").textContent, "Token forgotten. Reloading also forgets it.");
});

test("repeated connect during authentication does not start another request", async () => {
  const pendingPolicy = deferred();
  const calls = [];
  const { context, byId } = consoleHarness((path) => {
    calls.push(path);
    if (path === "/api/policy") return pendingPolicy.promise;
    return Promise.resolve(response({ rows: [] }));
  });
  byId("token").value = "test-owner-token";
  const connecting = context.connect();
  await context.connect();
  assert.deepEqual(calls, ["/api/policy", "/api/ledger"]);
  assert.equal(byId("disconnect").disabled, false);
  context.disconnect();
  pendingPolicy.resolve(response(policy));
  await connecting;
  assert.equal(byId("workspace").hidden, true);
});

test("a stale ledger refresh cannot reveal a previous owner's rows after reconnect", async () => {
  const staleLedger = deferred();
  let ledgerCalls = 0;
  const { context, byId } = consoleHarness((path, options) => {
    if (path === "/api/policy") return Promise.resolve(response(policy));
    if (path === "/api/ledger") {
      ledgerCalls += 1;
      if (ledgerCalls === 2) return Promise.resolve({ ok: true, json: () => staleLedger.promise });
      return Promise.resolve(response({ rows: [] }));
    }
    throw new Error(`Unexpected request: ${path} ${options?.method}`);
  });
  byId("token").value = "first-token";
  await context.connect();
  assert.equal(byId("workspace").hidden, false);
  const refreshing = context.loadLedger();
  context.disconnect();
  byId("token").value = "second-token";
  await context.connect();
  staleLedger.resolve({ rows: [{ t: 1, kind: "private", verdict: "hold", reason: "old", latency_ms: 1 }] });
  await assert.rejects(refreshing, (error) => error.name === "StaleSessionError");
  assert.equal(byId("ledger-rows").children.length, 1);
  assert.equal(byId("ledger-rows").children[0].children[0].textContent, "No verdicts yet.");
  assert.equal(byId("workspace").hidden, false);
});

test("camera-sign ledger rows show only a source label, not OCR text", async () => {
  const { context, byId } = consoleHarness((path) => {
    if (path === "/api/policy") return Promise.resolve(response(policy));
    if (path === "/api/ledger") return Promise.resolve(response({ rows: [{
      t: 1, kind: "inbound", source: "camera_sign", verdict: "block",
      reason: "injection", latency_ms: 12, route_choice: null,
    }] }));
    throw new Error(`Unexpected request: ${path}`);
  });
  byId("token").value = "test-owner-token";
  await context.connect();
  const row = byId("ledger-rows").children[0];
  assert.equal(row.children[1].textContent, "camera sign");
  assert.equal(row.children[2].textContent, "block");
  assert.equal(row.children.length, 6);
  assert.equal(byId("workspace").hidden, false);
});
