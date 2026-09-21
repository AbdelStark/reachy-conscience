"use strict";

let ownerToken = "";
let registeredTools = [];
let approvalAvailable = false;
let pendingApproval = null;
const byId = (id) => document.getElementById(id);

function status(message, failed = false) {
  const element = byId("status");
  element.textContent = message;
  element.classList.toggle("error", failed);
}

async function request(path, options = {}) {
  const headers = { Authorization: `Bearer ${ownerToken}`, ...options.headers };
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try { message = (await response.json()).error || message; } catch (_) { /* no body */ }
    throw new Error(message);
  }
  return response;
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

async function loadApproval() {
  if (!approvalAvailable || !ownerToken) return;
  const response = await request("/api/approval");
  renderApproval((await response.json()).pending);
}

async function loadPolicy() {
  const response = await request("/api/policy");
  renderPolicy(await response.json());
}

async function loadLedger() {
  const response = await request("/api/ledger");
  const { rows } = await response.json();
  const body = byId("ledger-rows");
  body.replaceChildren();
  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const value of [new Date(row.t * 1000).toLocaleString(), row.kind, row.verdict, row.reason, `${row.latency_ms.toFixed(1)} ms`]) {
      const td = document.createElement("td");
      td.textContent = value;
      tr.append(td);
    }
    body.append(tr);
  }
  if (rows.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No verdicts yet.";
    tr.append(td);
    body.append(tr);
  }
}

async function connect() {
  ownerToken = byId("token").value.trim();
  if (!ownerToken) { status("Paste the owner token first.", true); return; }
  byId("token").value = "";
  try {
    await Promise.all([loadPolicy(), loadLedger()]);
    await loadApproval();
    byId("workspace").hidden = false;
    byId("connect").disabled = true;
    byId("token").disabled = true;
    byId("disconnect").disabled = false;
    status("Connected to the local owner console.");
  } catch (error) {
    ownerToken = "";
    byId("workspace").hidden = true;
    status(error.message, true);
  }
}

function disconnect() {
  ownerToken = "";
  registeredTools = [];
  approvalAvailable = false;
  pendingApproval = null;
  byId("workspace").hidden = true;
  byId("connect").disabled = false;
  byId("token").disabled = false;
  byId("disconnect").disabled = true;
  byId("preview-results").replaceChildren();
  status("Token forgotten. Reloading also forgets it.");
}

async function decideApproval(approve) {
  if (!pendingApproval) return;
  const { request_id, digest, tool } = pendingApproval;
  if (approve && byId("approval-confirm-name").value !== tool) return;
  if (approve && !window.confirm(`Approve ${tool} with exactly the JSON arguments shown?`)) return;
  try {
    await request("/api/approval", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id, digest, approve }),
    });
    renderApproval(null);
    status(approve ? "Approval recorded; execution is not confirmed." : "Action denied.");
  } catch (error) {
    await loadApproval();
    status(error.message, true);
  }
}

async function save() {
  try {
    const response = await request("/api/policy", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(currentPolicy()),
    });
    renderPolicy(await response.json());
    status("Policy saved. Restart or reload the guard host to apply it to live turns.");
  } catch (error) { status(error.message, true); }
}

async function preview() {
  if (!window.confirm("Run five synthetic cases through the configured guard? This can make five billable calls.")) return;
  const button = byId("preview");
  button.disabled = true;
  try {
    const response = await request("/api/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ policy: currentPolicy(), confirm_live_calls: true }),
    });
    const { results } = await response.json();
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
  } catch (error) { status(error.message, true); }
  finally { button.disabled = false; }
}

async function exportLedger() {
  try {
    const response = await request("/api/export");
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "conscience-ledger.jsonl";
    link.click();
    URL.revokeObjectURL(link.href);
    status("Exported JSONL without summaries.");
  } catch (error) { status(error.message, true); }
}

byId("connect").addEventListener("click", connect);
byId("token").addEventListener("keydown", (event) => { if (event.key === "Enter") connect(); });
byId("disconnect").addEventListener("click", disconnect);
byId("save").addEventListener("click", save);
byId("preview").addEventListener("click", preview);
byId("refresh").addEventListener("click", async () => {
  try { await loadLedger(); status("Ledger refreshed."); } catch (error) { status(error.message, true); }
});
byId("export").addEventListener("click", exportLedger);
byId("approval-confirm-name").addEventListener("input", () => {
  byId("approve-action").disabled = !pendingApproval || byId("approval-confirm-name").value !== pendingApproval.tool;
});
byId("approve-action").addEventListener("click", () => decideApproval(true));
byId("deny-action").addEventListener("click", () => decideApproval(false));
setInterval(() => { if (ownerToken && approvalAvailable) loadApproval().catch(() => status("Approval channel unavailable.", true)); }, 1000);
