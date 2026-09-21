"use strict";

let ownerToken = "";
let registeredTools = [];
let approvalAvailable = false;
let pendingApproval = null;
let sessionVersion = 0;
let connecting = false;
const byId = (id) => document.getElementById(id);

class StaleSessionError extends Error {
  constructor() {
    super("Owner session changed.");
    this.name = "StaleSessionError";
  }
}

function assertSession(version) {
  if (!ownerToken || version !== sessionVersion) throw new StaleSessionError();
}

function reportError(error, version) {
  if (version === sessionVersion && !(error instanceof StaleSessionError)) status(error.message, true);
}

function status(message, failed = false) {
  const element = byId("status");
  element.textContent = message;
  element.classList.toggle("error", failed);
}

async function request(path, options = {}, version = sessionVersion) {
  assertSession(version);
  const headers = { Authorization: `Bearer ${ownerToken}`, ...options.headers };
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  assertSession(version);
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try { message = (await response.json()).error || message; } catch (_) { /* no body */ }
    assertSession(version);
    throw new Error(message);
  }
  return response;
}

async function requestJson(path, options = {}, version = sessionVersion) {
  const response = await request(path, options, version);
  const result = await response.json();
  assertSession(version);
  return result;
}

function currentPolicy() {
  return {
    rules: byId("rules").value,
    confirm_before: [...document.querySelectorAll("#tools input:checked")].map((input) => input.value),
    block_threshold: Number(byId("block-threshold").value),
    hold_floor: Number(byId("hold-floor").value),
  };
}

function renderTools(selected) {
  const holder = byId("tools");
  holder.replaceChildren();
  if (registeredTools.length === 0) {
    holder.textContent = "No tools registered in this host.";
    return;
  }
  for (const name of registeredTools) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = name;
    input.checked = selected.includes(name);
    label.append(input, document.createTextNode(` ${name}`));
    holder.append(label);
  }
}

function renderPolicy(policy) {
  byId("rules").value = policy.rules;
  byId("block-threshold").value = policy.block_threshold;
  byId("hold-floor").value = policy.hold_floor;
  registeredTools = policy.registered_tools;
  renderTools(policy.confirm_before);
  byId("preview").disabled = !policy.preview_available;
  byId("redteam").disabled = !policy.preview_available;
  approvalAvailable = policy.approval_available;
  byId("approval-section").hidden = !approvalAvailable;
}

function renderApproval(request) {
  if (request?.request_id !== pendingApproval?.request_id) {
    byId("approval-confirm-name").value = "";
  }
  pendingApproval = request;
  byId("approval-empty").hidden = request !== null;
  byId("approval-request").hidden = request === null;
  if (request === null) return;
  byId("approval-tool").textContent = request.tool;
  byId("approval-arguments").textContent = request.arguments_json;
  byId("approval-digest").textContent = request.digest;
  byId("approval-seconds").textContent = Math.max(0, Math.ceil(request.remaining_s));
  byId("approve-action").disabled = byId("approval-confirm-name").value !== request.tool;
}

async function loadApproval(version = sessionVersion) {
  if (!approvalAvailable || !ownerToken) return;
  renderApproval((await requestJson("/api/approval", {}, version)).pending);
}

async function loadPolicy(version = sessionVersion) {
  renderPolicy(await requestJson("/api/policy", {}, version));
}

async function loadLedger(version = sessionVersion) {
  const { rows } = await requestJson("/api/ledger", {}, version);
  const body = byId("ledger-rows");
  body.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    const route = row.route_choice
      ? `${row.route_choice} (${Math.round(row.route_confidence * 100)}%)${row.route_choice === "fast_path" ? ` · ${row.fast_command}` : ""}`
      : "—";
    const kind = row.source === "camera_sign" ? `camera sign · ${row.kind}` : row.kind;
    for (const value of [new Date(row.t * 1000).toLocaleString(), kind, row.verdict, route, row.reason, `${row.latency_ms.toFixed(1)} ms`]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    body.append(tr);
  }
  if (rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.textContent = "No verdicts yet.";
    tr.append(td);
    body.append(tr);
  }
}

async function connect() {
  if (connecting || ownerToken) return;
  ownerToken = byId("token").value.trim();
  if (!ownerToken) { status("Paste the owner token first.", true); return; }
  const version = ++sessionVersion;
  connecting = true;
  byId("token").value = "";
  byId("connect").disabled = true;
  byId("token").disabled = true;
  byId("disconnect").disabled = false;
  status("Connecting to the local owner console...");
  try {
    await Promise.all([loadPolicy(version), loadLedger(version)]);
    await loadApproval(version);
    assertSession(version);
    byId("workspace").hidden = false;
    status("Connected to the local owner console.");
  } catch (error) {
    if (version === sessionVersion) disconnect(error.message, true);
  } finally {
    if (version === sessionVersion) connecting = false;
  }
}

function disconnect(message = "Token forgotten. Reloading also forgets it.", failed = false) {
  ++sessionVersion;
  ownerToken = "";
  connecting = false;
  registeredTools = [];
  approvalAvailable = false;
  pendingApproval = null;
  byId("workspace").hidden = true;
  byId("connect").disabled = false;
  byId("token").disabled = false;
  byId("disconnect").disabled = true;
  byId("rules").value = "";
  byId("block-threshold").value = "";
  byId("hold-floor").value = "";
  byId("tools").replaceChildren();
  byId("ledger-rows").replaceChildren();
  byId("approval-confirm-name").value = "";
  byId("approval-tool").textContent = "";
  byId("approval-arguments").textContent = "";
  byId("approval-digest").textContent = "";
  byId("approval-seconds").textContent = "";
  byId("approval-section").hidden = true;
  byId("approval-empty").hidden = false;
  byId("approval-request").hidden = true;
  byId("approve-action").disabled = true;
  byId("preview").disabled = true;
  byId("redteam").disabled = true;
  byId("preview-results").replaceChildren();
  byId("redteam-results").replaceChildren();
  status(message, failed);
}

async function decideApproval(approve) {
  if (!pendingApproval) return;
  const version = sessionVersion;
  const { request_id, digest, tool } = pendingApproval;
  if (approve && byId("approval-confirm-name").value !== tool) return;
  if (approve && !window.confirm(`Approve ${tool} with exactly the JSON arguments shown?`)) return;
  try {
    await requestJson("/api/approval", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id, digest, approve }),
    }, version);
    renderApproval(null);
    status(approve ? "Approval recorded; execution is not confirmed." : "Action denied.");
  } catch (error) {
    if (version !== sessionVersion) return;
    try { await loadApproval(version); } catch (_) { /* preserve original error */ }
    reportError(error, version);
  }
}

async function save() {
  const version = sessionVersion;
  try {
    const policy = await requestJson("/api/policy", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(currentPolicy()),
    }, version);
    renderPolicy(policy);
    status("Policy saved. Restart or reload the guard host to apply it to live turns.");
  } catch (error) { reportError(error, version); }
}

async function preview() {
  if (!window.confirm("Run five synthetic cases through the configured guard? This can make five billable calls.")) return;
  const version = sessionVersion;
  const button = byId("preview");
  button.disabled = true;
  try {
    const { results } = await requestJson("/api/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ policy: currentPolicy(), confirm_live_calls: true }),
    }, version);
    const holder = byId("preview-results");
    holder.replaceChildren();
    const heading = document.createElement("h3");
    heading.textContent = "Preview results";
    holder.append(heading);
    for (const result of results) {
      const line = document.createElement("p");
      line.textContent = `${result.case_id}: ${result.verdict} · ${result.reason} · ${result.latency_ms.toFixed(1)} ms`;
      holder.append(line);
    }
    status("Preview complete. No action was dispatched.");
  } catch (error) { reportError(error, version); }
  finally { if (version === sessionVersion) button.disabled = false; }
}

async function redteam() {
  if (!window.confirm("Run 20 self-authored synthetic cases through the configured guard? This can make 20 billable calls. No action will execute.")) return;
  const version = sessionVersion;
  const button = byId("redteam");
  button.disabled = true;
  try {
    const report = await requestJson("/api/redteam", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ policy: currentPolicy(), confirm_live_calls: true }),
    }, version);
    const holder = byId("redteam-results");
    holder.replaceChildren();
    const heading = document.createElement("h3");
    heading.textContent = "Synthetic red-team run";
    const summary = document.createElement("p");
    summary.textContent = `20 self-authored cases · approved ${report.counts.approve} · held ${report.counts.hold} · blocked ${report.counts.block} · observed p95 ${report.p95_latency_ms.toFixed(1)} ms`;
    const caveat = document.createElement("p");
    caveat.textContent = "Case intent is not a correctness label. These counts are not a confusion matrix, safety rate, or live robot measurement.";
    holder.append(heading, summary, caveat);
    for (const item of report.results) {
      const line = document.createElement("p");
      line.textContent = `${item.case_id} (${item.channel}, ${item.intent}): ${item.verdict} · ${item.reason} · ${item.latency_ms.toFixed(1)} ms`;
      holder.append(line);
    }
    status("Synthetic guard run complete. No action was dispatched.");
  } catch (error) { reportError(error, version); }
  finally { if (version === sessionVersion) button.disabled = false; }
}

async function exportLedger() {
  const version = sessionVersion;
  try {
    const response = await request("/api/export", {}, version);
    const blob = await response.blob();
    assertSession(version);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "conscience-ledger.jsonl";
    link.click();
    URL.revokeObjectURL(link.href);
    status("Exported JSONL without summaries.");
  } catch (error) { reportError(error, version); }
}

byId("connect").addEventListener("click", connect);
byId("token").addEventListener("keydown", (event) => { if (event.key === "Enter") connect(); });
byId("disconnect").addEventListener("click", () => disconnect());
byId("save").addEventListener("click", save);
byId("preview").addEventListener("click", preview);
byId("redteam").addEventListener("click", redteam);
byId("refresh").addEventListener("click", async () => {
  const version = sessionVersion;
  try { await loadLedger(version); status("Ledger refreshed."); } catch (error) { reportError(error, version); }
});
byId("export").addEventListener("click", exportLedger);
byId("approval-confirm-name").addEventListener("input", () => {
  byId("approve-action").disabled = !pendingApproval || byId("approval-confirm-name").value !== pendingApproval.tool;
});
byId("approve-action").addEventListener("click", () => decideApproval(true));
byId("deny-action").addEventListener("click", () => decideApproval(false));
setInterval(() => {
  if (!ownerToken || !approvalAvailable) return;
  const version = sessionVersion;
  loadApproval(version).catch((error) => {
    if (version === sessionVersion && !(error instanceof StaleSessionError)) status("Approval channel unavailable.", true);
  });
}, 1000);
