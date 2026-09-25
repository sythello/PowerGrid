// Exercise the browser export workflow without adding a DOM dependency.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function harness(fetch) {
  const downloads = [];
  const messages = [];
  const timers = [];
  let currentBlob;
  const context = {
    fetch, Blob, console,
    document: {
      addEventListener() {},
      body: { append() {} },
      createElement() {
        return { click() { downloads.push({ filename: this.download, blob: currentBlob }); }, remove() {} };
      },
    },
    URL: { createObjectURL(blob) { currentBlob = blob; return "blob:test"; }, revokeObjectURL() {} },
    setTimeout(fn, delay) { timers.push({ fn, delay }); return timers.length; },
    clearTimeout() {},
    recordToast(message) { messages.push(message); },
  };
  const source = fs.readFileSync(path.join(__dirname, "../src/powergrid/web/static/app.js"), "utf8");
  // Expose the existing IIFE's functions only inside this isolated test VM.
  const instrumented = source.replace(/\}\)\(\);\s*$/, `
    renderHeader = () => {};
    renderActionConsole = () => {};
    showToast = recordToast;
    globalThis.app = { ui, api, exportGameLog, setSnapshot, scheduleAi };
  })();`);
  vm.runInNewContext(instrumented, context);
  context.app.ui.snapshot = {
    has_game: true, state: { round_number: 1, step: 1, phase: "auction" },
    request: null, needs_ai_advance: true,
  };
  return { ...context.app, downloads, messages, timers };
}

function response(payload, ok = true) {
  return { ok, status: ok ? 200 : 400, async json() { return payload; } };
}

test("export waits for an in-flight action, preserves the displayed state, and leaves AI paused", async () => {
  const pendingAction = deferred();
  const requests = [];
  const h = harness((url) => {
    requests.push(url);
    if (url === "/api/advance") return pendingAction.promise;
    return Promise.resolve(response({ download_filename: "capture.json", game_log: [{ index: 1 }] }));
  });
  h.ui.aiWorking = true;
  h.ui.buildCities = ["test-city"];
  h.ui.layoutEditor.resumeAi = true;
  const action = h.api("/api/advance", { method: "POST", body: {} }).then((state) => h.setSnapshot(state));
  const capture = h.exportGameLog();
  await h.exportGameLog(); // Double-click must not create a second download.
  assert.deepEqual(requests, ["/api/advance"]);
  assert.equal(h.ui.aiPaused, true);
  assert.equal(h.ui.layoutEditor.resumeAi, false);
  pendingAction.resolve(response({
    has_game: true, state: { round_number: 2, step: 1, phase: "buy_resources" },
    request: null, needs_ai_advance: false,
  }));
  await Promise.all([action, capture]);
  assert.deepEqual(requests, ["/api/advance", "/api/game-log"]);
  assert.equal(h.downloads.length, 1);
  assert.equal(h.downloads[0].filename, "capture.json");
  const payload = JSON.parse(await h.downloads[0].blob.text());
  assert.equal(payload.browser_context.displayed_snapshot.state.round_number, 1);
  assert.deepEqual(payload.browser_context.selections.build_cities, ["test-city"]);
  assert.equal(payload.browser_context.pending_request_count, 1);
  assert.equal(h.ui.snapshot.state.round_number, 2);
  assert.equal(h.ui.aiPaused, true);
  assert.equal(h.ui.exportingLog, false);
  assert.equal(h.ui.pendingRequests.size, 0);
  h.ui.snapshot.needs_ai_advance = true;
  h.scheduleAi();
  assert.deepEqual(h.timers.map((timer) => timer.delay), [10000]); // Blob cleanup only.
});

test("a failed capture leaves AI paused and permits retry after game completion", async () => {
  let calls = 0;
  const h = harness(async () => ++calls === 1
    ? response({ error: "temporary failure" }, false)
    : response({ download_filename: "finished.json", winner_result: { winner_ids: ["p1"] } }));
  h.ui.snapshot.winner = { winner_ids: ["p1"] };
  h.ui.snapshot.needs_ai_advance = false;
  await h.exportGameLog();
  assert.equal(h.downloads.length, 0);
  assert.equal(h.ui.exportingLog, false);
  assert.equal(h.ui.aiPaused, true);
  assert.match(h.messages[0], /temporary failure/);
  await h.exportGameLog();
  assert.equal(h.downloads.length, 1);
  const payload = JSON.parse(await h.downloads[0].blob.text());
  assert.deepEqual(payload.winner_result.winner_ids, ["p1"]);
});

test("a rejected in-flight request does not prevent diagnostic export", async () => {
  const pendingAction = deferred();
  const h = harness((url) => url === "/api/intent" ? pendingAction.promise
    : Promise.resolve(response({ download_filename: "error.json" })));
  const action = h.api("/api/intent", { method: "POST", body: {} }).catch(() => {});
  const capture = h.exportGameLog();
  pendingAction.reject(new Error("connection lost"));
  await Promise.all([action, capture]);
  assert.equal(h.downloads.length, 1);
  assert.equal(h.ui.pendingRequests.size, 0);
});
