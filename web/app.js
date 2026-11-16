// Arm play check: page logic. The measuring runs in worker.js; this file only reads the form,
// shows progress and lays out the card it gets back. Every number shown is a string the engine
// formatted from the card; nothing is recomputed here.

const $ = (id) => document.getElementById(id);
const ENGINE_TIMEOUT_MS = 240000;
const JOINT_NAMES = {
  shoulder_pan: "Shoulder pan", shoulder_lift: "Shoulder lift", elbow_flex: "Elbow flex",
  wrist_flex: "Wrist flex", wrist_roll: "Wrist roll",
};

let worker = null;
let warm = false;
let running = false;
let engineTimer = null;
let urls = [];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

// ---- input -----------------------------------------------------------------------------------

const REPO = /^[A-Za-z0-9][A-Za-z0-9._-]*\/[A-Za-z0-9][A-Za-z0-9._-]*$/;

function parseInput(raw, rawSub) {
  let s = raw.trim().replace(/[?#].*$/, "");
  let sub = rawSub.trim().replace(/^\/+|\/+$/g, "");
  const m = /^(?:https?:\/\/)?(?:www\.)?(?:hf\.co|huggingface\.co)\/(?:datasets\/)?(.*)$/i.exec(s);
  if (m) s = m[1];
  s = s.replace(/^datasets\//, "").replace(/\/+$/, "");
  let parts = s.split("/").filter(Boolean);
  if (parts.length > 2) {
    // user/name/tree/main/folder..., user/name/folder...
    let rest = parts.slice(2);
    if ((rest[0] === "tree" || rest[0] === "blob" || rest[0] === "resolve") && rest.length >= 2) rest = rest.slice(2);
    rest = rest.filter((x) => x !== "meta" && x !== "data" && !x.endsWith(".json") && !x.endsWith(".parquet"));
    if (!sub && rest.length) sub = rest.join("/");
    parts = parts.slice(0, 2);
  }
  const repo = parts.join("/");
  if (!REPO.test(repo)) return { error: "Give the dataset as user/name, for example lerobot/svla_so101_pickplace." };
  if (sub.split("/").some((x) => x === ".." || x === ".")) return { error: "The folder cannot contain . or .. parts." };
  return { repo, sub };
}

// ---- worker ----------------------------------------------------------------------------------

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL("worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (ev) => onMessage(ev.data);
  worker.onerror = (ev) => {
    ev.preventDefault();
    if (running) fail({ kind: "engine", message: "the worker did not start" });
  };
  return worker;
}

function warmUp() {
  if (warm || worker) return;
  getWorker().postMessage({ type: "warm" });
}

function stopWorker() {
  if (worker) worker.terminate();
  worker = null;
  warm = false;
}

// ---- progress --------------------------------------------------------------------------------

const ORDER = ["info", "data", "engine", "measure"];

function step(name, state) {
  const li = document.querySelector(`#hc-steps li[data-step="${name}"]`);
  if (!li) return;
  li.classList.remove("is-active", "is-done", "is-fail");
  if (state) li.classList.add("is-" + state);
}

function announce(text) {
  $("hc-live").textContent = text;
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function mb(n) {
  return n >= 1e6 ? (n / 1e6).toFixed(1) + " MB" : Math.max(1, Math.round(n / 1e3)) + " kB";
}

function onMessage(m) {
  if (m.type === "warm") {
    warm = true;
    if (running) step("engine", "done");
    return;
  }
  if (!running) return;
  if (m.type === "progress") {
    if (m.step === "info") { step("info", "active"); announce("Reading the dataset's description."); }
    if (m.step === "list") {
      clearTimeout(engineTimer);
      warm = true;
      step("info", "done");
      step("engine", "done");
    }
    if (m.step === "data") {
      step("data", "active");
      const bar = $("hc-data-bar");
      bar.max = Math.max(1, m.total || 1);
      bar.value = Math.min(m.done || 0, bar.max);
      $("hc-data-detail").textContent = `${mb(m.done || 0)} of ${mb(m.total || 0)}, ${m.files} file${m.files === 1 ? "" : "s"}`;
      if (!m.done) announce(`Downloading ${m.files} data file${m.files === 1 ? "" : "s"}, ${mb(m.total || 0)}.`);
    }
    if (m.step === "engine-wait") {
      step("info", "done");
      if (!warm) {
        step("engine", "active");
        announce("Starting the measuring engine.");
        clearTimeout(engineTimer);
        engineTimer = setTimeout(() => {
          if (running) { stopWorker(); fail({ kind: "engine", message: "it did not start within four minutes" }); }
        }, ENGINE_TIMEOUT_MS);
      }
    }
    if (m.step === "measure") {
      step("data", "done");
      step("measure", "active");
      announce("Measuring every hold.");
    }
    return;
  }
  clearTimeout(engineTimer);
  if (m.type === "result") show(m.view);
  if (m.type === "fail") fail(m);
}

// ---- states ----------------------------------------------------------------------------------

let current = null;

function setRunning(on) {
  running = on;
  $("hc-go").disabled = on;
  $("hc-go").textContent = on ? "Checking…" : "Check my arm";
  $("hc-stop").hidden = !on;
  document.body.classList.toggle("is-running", on);
}

function reset() {
  for (const u of urls) URL.revokeObjectURL(u);
  urls = [];
  $("hc-message").hidden = true;
  $("hc-result").hidden = true;
  $("hc-status").hidden = true;
  for (const s of ORDER) step(s, null);
  $("hc-data-bar").value = 0;
  $("hc-data-detail").textContent = "";
  announce("");
  setPlate(null);
}

function message(label, title, paras, opts = {}) {
  setRunning(false);
  $("hc-status").hidden = !opts.keepSteps;
  $("hc-result").hidden = true;
  $("hc-message-label").textContent = label;
  $("hc-message-title").textContent = title;
  const body = $("hc-message-body");
  body.replaceChildren();
  for (const p of paras) body.append(typeof p === "string" ? el("p", null, p) : p);
  if (opts.view) body.append(downloads(opts.view, true));
  $("hc-message").hidden = false;
  $("hc-message-title").setAttribute("tabindex", "-1");
  $("hc-message-title").focus({ preventScroll: false });
}

function idText() {
  return current.sub ? `${current.repo} (folder ${current.sub})` : current.repo;
}

function fail(m) {
  const stepNow = ORDER.find((s) => document.querySelector(`#hc-steps li[data-step="${s}"].is-active`));
  if (stepNow) step(stepNow, "fail");
  const id = idText();
  const k = m.kind;
  if (k === "missing" && m.message === "subfolder") {
    return message("Not found", "That folder has no dataset in it.", [
      `${current.repo} has no meta/info.json under ${current.sub}. Give the folder that holds meta/ and data/, as it appears in the dataset's file list on Hugging Face.`,
    ]);
  }
  if (k === "missing") {
    return message("Not found", "Hugging Face did not give us that dataset.", [
      `No public dataset answers to ${current.repo}. Check the id (user/name, as in the dataset's address). Private and gated datasets cannot be read from a browser page; this check only reads public ones.`,
    ]);
  }
  if (k === "layout" && m.message === "no-columns") {
    return message("Not an arm recording", "This dataset has no leader and follower columns.", [
      `${id} does not record both action and observation.state, so there is no command to compare the arm with. The check reads LeRobot teleoperation recordings, where a leader arm drives a follower.`,
    ]);
  }
  if (k === "layout" && m.message === "no-data") {
    return message("No recordings", "No joint recordings found.", [
      `${id} has a description but no data/*.parquet files next to it. If the recording sits in a folder of a larger repository, give that folder.`,
    ]);
  }
  if (k === "layout") return message("Not readable", "The dataset's description could not be read.", [m.message]);
  if (k === "large") {
    return message("Too large for a browser tab", "This dataset is too large to measure here.", [
      `${id} has ${Number(m.frames || 0).toLocaleString("en-US")} frames and ${mb(m.total || 0)} of joint data. A browser tab can run out of memory above 600,000 frames or 400 MB. One dataset of an hour or less with the same arm is enough for the figures.`,
    ]);
  }
  if (k === "network") {
    return message("No connection", "Hugging Face could not be reached.", [
      `${m.message} Check your connection and try again. Some networks and extensions block huggingface.co or its file servers.`,
    ], { keepSteps: true });
  }
  return message("Engine", "The measuring engine stopped in this tab.", [
    `In this browser the engine failed: ${m.message}. The check runs in a current Chrome, Edge, Firefox or Safari with WebAssembly on; try one of those, or reload the page.`,
  ], { keepSteps: true });
}

// ---- result ----------------------------------------------------------------------------------

function strip(v, pct, label) {
  // Placement against other arms; lower is tighter. Scale: 0 to a little past the 90th percentile
  // (or past this arm's value), capped at three times the 90th percentile.
  const wrap = el("div", "hc-strip");
  if (!pct) return null;
  const top = Math.min(Math.max(pct.p90 * 1.25, v * 1.08), pct.p90 * 3);
  const x = (n) => Math.max(0, Math.min(100, (n / top) * 100));
  const wide = el("span", "hc-strip-wide");
  wide.style.setProperty("--a", x(pct.p10) + "%");
  wide.style.setProperty("--b", x(pct.p90) + "%");
  const mid = el("span", "hc-strip-band");
  mid.style.setProperty("--a", x(pct.p25) + "%");
  mid.style.setProperty("--b", x(pct.p75) + "%");
  const med = el("span", "hc-strip-mid");
  med.style.setProperty("--a", x(pct.p50) + "%");
  const you = el("span", "hc-strip-you" + (v > top ? " is-off" : ""));
  you.style.setProperty("--a", x(v) + "%");
  wrap.append(wide, mid, med, you);
  wrap.setAttribute("role", "img");
  wrap.setAttribute("aria-label", label);
  return wrap;
}

function tipFigure(name, fig, arms) {
  const d = el("div", "hc-fig");
  d.append(el("span", "label", name));
  const n = el("p", "hc-fig-num", fig.text);
  n.append(el("small", null, " mm"));
  d.append(n);
  d.append(el("p", "hc-fig-what", "95% of arrivals land within this distance of the target, sideways."));
  const s = strip(fig.v, fig.pct, `${fig.text} millimetres; ${fig.place || "no comparison"} among ${arms || "other"} arms`);
  if (s) d.append(s);
  if (fig.place) d.append(el("p", "hc-place", fig.place + (arms ? `, ${arms} arms` : "")));
  return d;
}

function setPlate(tip) {
  const fig = $("hc-plate");
  const yours = tip && tip.one_side_p95_mm ? [tip.one_side_p95_mm.v, tip.either_side_p95_mm.v] : null;
  drawPlate($("hc-plate-draw"), yours);
  const cap = $("hc-plate-cap");
  cap.replaceChildren();
  if (yours) {
    cap.append(el("span", "label", "Your arm and a typical one, from above"));
    cap.append(`95% of arrivals land inside each circle. Yours (blue): ${tip.one_side_p95_mm.text} mm from one side, ${tip.either_side_p95_mm.text} mm from any side. Typical (grey): ${fig.dataset.one} and ${fig.dataset.either} mm. Grid 5 mm.`);
    $("hc-plate-draw").setAttribute("aria-label", `Seen from above at the gripper. This arm: 95 percent of arrivals within ${tip.one_side_p95_mm.text} millimetres from one side and ${tip.either_side_p95_mm.text} from any side. A typical SO arm: ${fig.dataset.one} and ${fig.dataset.either}.`);
  } else {
    cap.append(el("span", "label", "A typical SO arm, from above"));
    cap.append(`95% of arrivals at a target land inside each circle. Inner: approaching from one side, ${fig.dataset.one} mm. Outer: from any side, ${fig.dataset.either} mm. Grid 5 mm.`);
    $("hc-plate-draw").setAttribute("aria-label", `A typical SO arm, seen from above at the gripper: 95 percent of arrivals land within ${fig.dataset.one} millimetres of the target when the arm approaches from one side, and within ${fig.dataset.either} millimetres when it approaches from any side.`);
  }
}

const NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs, cls) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, String(v));
  if (cls) e.setAttribute("class", cls);
  return e;
}

function drawPlate(host, yours) {
  const fig = $("hc-plate");
  const typ = [Number(fig.dataset.oneV), Number(fig.dataset.eitherV)];
  const big = Math.max(typ[1], yours ? yours[1] : 0);
  const R = Math.max(12.5, Math.ceil((big * 1.12) / 5) * 5);
  const s = svg("svg", { viewBox: `${-R} ${-R} ${2 * R} ${2 * R}`, "aria-hidden": "true", focusable: "false" });
  for (let g = -Math.floor(R / 5) * 5; g <= R; g += 5) {
    s.append(svg("line", { x1: g, y1: -R, x2: g, y2: R, "vector-effect": "non-scaling-stroke" }, g === 0 ? "hc-axis" : "hc-grid"));
    s.append(svg("line", { x1: -R, y1: g, x2: R, y2: g, "vector-effect": "non-scaling-stroke" }, g === 0 ? "hc-axis" : "hc-grid"));
  }
  const ring = (r, cls) => s.append(svg("circle", { cx: 0, cy: 0, r, "vector-effect": "non-scaling-stroke" }, cls));
  ring(typ[1], yours ? "hc-ring-typ hc-ring-out" : "hc-ring hc-ring-out");
  ring(typ[0], yours ? "hc-ring-typ" : "hc-ring");
  if (yours) {
    ring(yours[1], "hc-ring-you hc-ring-out");
    ring(yours[0], "hc-ring-you");
  }
  s.append(svg("circle", { cx: 0, cy: 0, r: R / 60 }, "hc-target"));
  host.replaceChildren(s);
}

function downloads(view, compact) {
  const box = el("div", "hc-downloads");
  const base = ("holdcheck-" + current.repo + (current.sub ? "-" + current.sub : "")).replace(/[^A-Za-z0-9._-]+/g, "_").slice(0, 120);
  const add = (text, type, name, label, cls) => {
    const u = URL.createObjectURL(new Blob([text], { type }));
    urls.push(u);
    const a = el("a", cls, label);
    a.href = u;
    a.download = name;
    box.append(a);
    return a;
  };
  const j = add(view.json + "\n", "application/json", base + ".json", "Download card (JSON)", "hc-dl");
  j.id = compact ? "hc-dl-json-2" : "hc-dl-json";
  if (!compact) add(view.text + "\n", "text/plain", base + ".txt", "Download as text", "hc-dl hc-dl-2").id = "hc-dl-text";
  return box;
}

function show(view) {
  window.__hcView = view; // read by the parity check harness
  if (view.state === "error") {
    return message("Not readable", "The recording could not be measured.", [
      `The engine could not read ${idText()}: ${view.message}`,
    ], { view });
  }
  if (view.state === "skip") {
    const what = view.robot_type ? `It is recorded as ${view.robot_type}` : "Its description names no robot type";
    return message("Not an SO-style arm", "This is not a single SO-100 or SO-101 arm.", [
      `${what}, and its joints are not the six of one SO-style arm (shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper). The check measures one SO-100 or SO-101 follower; two-arm rigs and other arms are left out.`,
      `Reason on the card: ${view.message}.`,
    ], { view });
  }
  if (view.state === "few") {
    const ul = el("ul", "hc-why");
    for (const r of view.rows) ul.append(el("li", null, `${JOINT_NAMES[r.joint]}: not measured, ${r.why}`));
    return message("Too few holds", "Not enough still moments to measure.", [
      `${idText()} has ${plural(view.episodes, "episode")} and ${view.frames.toLocaleString("en-US")} frames, but no joint has 8 holds arriving from above and 8 from below. A hold is the leader kept still for half a second after a move of at least a degree.`,
      ul,
      "Record or pick a dataset with more pauses: placing, aligning and waiting all count. A few dozen episodes of pick and place is usually enough.",
    ], { view });
  }
  setRunning(false);
  $("hc-status").hidden = true;
  $("hc-message").hidden = true;

  $("hc-for").textContent = `${idText()} · ${view.robot_type || "no robot type"} · ${plural(view.episodes, "episode")}, ${view.frames.toLocaleString("en-US")} frames at ${view.fps} fps`;

  const tipBox = $("hc-tip");
  tipBox.replaceChildren();
  const t = view.tip;
  if (t.one_side_p95_mm) {
    tipBox.append(tipFigure("Arriving from one side", t.one_side_p95_mm, t.arms));
    tipBox.append(tipFigure("Arriving from any side", t.either_side_p95_mm, t.arms));
    $("hc-do").textContent = `Leave at least ${t.either_side_p95_mm.text} mm of lead-in or tolerance around each target, or approach every target from the same side and ${t.one_side_p95_mm.text} mm will do.`;
    $("hc-tip-limit").parentElement.hidden = false;
  } else {
    tipBox.append(el("p", "hc-fig-none", `At the gripper: ${t.not_computed}. The gripper figures need all five arm joints measured.`));
    $("hc-do").textContent = "Use the per-joint figures below; record more holds on the joints that were not measured to get the gripper figures.";
    $("hc-tip-limit").parentElement.hidden = true;
  }
  setPlate(t.one_side_p95_mm ? t : null);

  const tbl = el("table", "hc-joints");
  const cap = el("caption", "hc-vh", "Lost motion, scatter and bias per joint, each placed among other SO arms");
  const head = el("thead");
  const hr = el("tr");
  for (const [h, c] of [["Joint", ""], ["Lost motion, deg", "num"], ["Scatter, deg", "num"], ["Bias, deg", "num"]]) {
    const th = el("th", c, h);
    th.scope = "col";
    hr.append(th);
  }
  head.append(hr);
  const body = el("tbody");
  let anyFree = false, anyAll = false;
  for (const r of view.rows) {
    const tr = el("tr");
    const th = el("th");
    th.scope = "row";
    th.append(el("span", "hc-jn", JOINT_NAMES[r.joint]));
    if (r.holds != null) th.append(el("span", "hc-jm", `${r.holds} holds${r.arms ? ` · vs ${r.arms} arms` : ""}`));
    tr.append(th);
    if (r.why) {
      const td = el("td", "hc-nm", `Not measured: ${r.why}`);
      td.colSpan = 3;
      tr.append(td);
      body.append(tr);
      continue;
    }
    const lm = el("td", "num hc-cell");
    lm.dataset.label = "Lost motion, deg";
    lm.append(el("span", "hc-v", r.lost_motion), el("span", "hc-ci", `90%: ${r.ci90}`));
    const s1 = strip(r.lost_motion_v, r.lost_motion_pct, `lost motion ${r.lost_motion} degrees, ${r.lost_motion_place}`);
    if (s1) lm.append(s1);
    if (r.lost_motion_place) lm.append(el("span", "hc-place", r.lost_motion_place));
    const sc = el("td", "num hc-cell");
    sc.dataset.label = "Scatter, deg";
    sc.append(el("span", "hc-v", r.scatter), el("span", "hc-ci", r.scatter_from));
    const s2 = strip(r.scatter_v, r.scatter_pct, `scatter ${r.scatter} degrees, ${r.scatter_place}`);
    if (s2) sc.append(s2);
    if (r.scatter_place) sc.append(el("span", "hc-place", r.scatter_place));
    if (r.scatter_from === "free holds") anyFree = true; else anyAll = true;
    const bi = el("td", "num hc-cell hc-bias");
    bi.dataset.label = "Bias, deg";
    bi.append(el("span", "hc-v", r.bias));
    tr.append(lm, sc, bi);
    body.append(tr);
  }
  tbl.append(cap, head, body);
  $("hc-joints").replaceChildren(tbl);
  $("hc-scatter-note").hidden = !(anyFree || anyAll);

  const dl = downloads(view, false);
  document.querySelector("#hc-result .hc-downloads").replaceWith(dl);

  const facts = $("hc-facts");
  facts.replaceChildren();
  const fact = (k, v) => { facts.append(el("dt", null, k), el("dd", null, v)); };
  fact("Dataset", current.repo);
  if (current.sub) fact("Folder", current.sub);
  fact("Recorded as", view.robot_type || "none given");
  fact("Layout", view.codebase_version || "not given");
  fact("Episodes", String(view.episodes));
  fact("Frames", view.frames.toLocaleString("en-US"));

  $("hc-result").hidden = false;
  const h = $("hc-tip-h");
  h.setAttribute("tabindex", "-1");
  h.focus({ preventScroll: false });
  announce("Measured. The card is below.");
}

// ---- wiring ----------------------------------------------------------------------------------

function start(ev) {
  ev.preventDefault();
  const p = parseInput($("hc-id").value, $("hc-sub").value);
  const err = $("hc-id-error");
  if (p.error) {
    err.textContent = p.error;
    err.hidden = false;
    $("hc-id").setAttribute("aria-invalid", "true");
    $("hc-id").focus();
    return;
  }
  err.hidden = true;
  $("hc-id").removeAttribute("aria-invalid");
  $("hc-id").value = p.repo;
  $("hc-sub").value = p.sub;
  current = p;
  reset();
  const q = new URLSearchParams({ dataset: p.repo });
  if (p.sub) q.set("folder", p.sub);
  history.replaceState(null, "", "?" + q.toString());
  setRunning(true);
  $("hc-status").hidden = false;
  if (warm) step("engine", "done");
  step("info", "active");
  getWorker().postMessage({ type: "check", repo: p.repo, sub: p.sub });
}

function stop() {
  stopWorker();
  clearTimeout(engineTimer);
  setRunning(false);
  reset();
  $("hc-id").focus();
}

function init() {
  const q = new URLSearchParams(location.search);
  if (q.get("dataset")) $("hc-id").value = q.get("dataset").slice(0, 300);
  if (q.get("folder")) $("hc-sub").value = q.get("folder").slice(0, 300);
  setPlate(null);
  $("hc-form").addEventListener("submit", start);
  $("hc-stop").addEventListener("click", stop);
  // Start the engine download once someone reaches for the form, so it is ready sooner.
  for (const e of ["focusin", "pointerdown"]) $("hc-form").addEventListener(e, warmUp, { once: true });
}

init();
