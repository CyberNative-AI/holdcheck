// Runs in a module worker: fetches one LeRobot dataset's meta/info.json and data/*.parquet from
// Hugging Face, then measures it with holdcheck.py inside Pyodide. Nothing is sent anywhere else.
import { loadPyodide } from "./pyodide/pyodide.mjs";

const HF = "https://huggingface.co";
const COLS = ["action", "observation.state"];
const MAX_FRAMES = 600000; // above this a browser tab runs short of memory turning columns into arrays
const MAX_BYTES = 400e6;
const PARALLEL = 4;

let py = null;
let booting = null;

const post = (type, data) => self.postMessage({ type, ...data });
const progress = (step, data = {}) => post("progress", { step, ...data });

class Fail extends Error {
  constructor(kind, message, extra = {}) { super(message); this.kind = kind; Object.assign(this, extra); }
}

const quote = (path) => path.split("/").map(encodeURIComponent).join("/");

async function get(url, what) {
  let r;
  try {
    r = await fetch(url, { credentials: "omit", referrerPolicy: "no-referrer" });
  } catch (e) {
    throw new Fail("network", `Could not reach Hugging Face for ${what}.`);
  }
  if (r.status === 401 || r.status === 403 || r.status === 404) throw new Fail("missing", what, { status: r.status });
  if (!r.ok) throw new Fail("network", `Hugging Face answered ${r.status} for ${what}.`);
  return r;
}

async function tree(repo, path) {
  let url = `${HF}/api/datasets/${repo}/tree/main/${quote(path)}?recursive=true`;
  const all = [];
  for (let page = 0; url && page < 200; page++) {
    const r = await get(url, "the list of data files");
    all.push(...(await r.json()));
    const m = /<([^>]+)>;\s*rel="next"/.exec(r.headers.get("Link") || "");
    url = m ? m[1] : null;
  }
  return all;
}

async function boot() {
  if (py) return py;
  if (!booting) {
    booting = (async () => {
      const base = new URL("./", import.meta.url);
      const p = await loadPyodide({ indexURL: new URL("pyodide/", base).href, stdout: () => {}, stderr: () => {} });
      await p.loadPackage(
        ["numpy-2.4.6-cp314-cp314-pyemscripten_2026_0_wasm32.whl", "pyodide_unix_timezones-1.0.0-py3-none-any.whl", "pyarrow-22.0.0-cp314-cp314-pyemscripten_2026_0_wasm32.whl"]
          .map((f) => new URL("pyodide/" + f, base).href),
        { messageCallback: () => {}, errorCallback: () => {} },
      );
      p.FS.mkdirTree("/hc");
      for (const f of ["holdcheck.py", "engine.py", "tipmap.json", "seatmap.json", "corpus_summary.json"]) {
        const r = await fetch(new URL("py/" + f, base));
        if (!r.ok) throw new Error("engine file " + f);
        p.FS.writeFile("/hc/" + f, new Uint8Array(await r.arrayBuffer()));
      }
      p.runPython("import sys; sys.dont_write_bytecode = True; sys.path.insert(0, '/hc'); import engine");
      py = p;
      return p;
    })();
    booting.catch(() => { booting = null; });
  }
  return booting;
}

async function download(repo, files, total) {
  const blobs = new Array(files.length);
  let done = 0, next = 0;
  const one = async () => {
    while (next < files.length) {
      const i = next++;
      const r = await get(`${HF}/datasets/${repo}/resolve/main/${quote(files[i].path)}`, "a data file");
      const reader = r.body.getReader();
      const parts = [];
      let n = 0;
      for (;;) {
        const { done: end, value } = await reader.read();
        if (end) break;
        parts.push(value); n += value.length; done += value.length;
        progress("data", { done, total, file: i + 1, files: files.length });
      }
      const buf = new Uint8Array(n);
      let o = 0;
      for (const v of parts) { buf.set(v, o); o += v.length; }
      blobs[i] = buf;
    }
  };
  await Promise.all(Array.from({ length: Math.min(PARALLEL, files.length) }, one));
  return blobs;
}

async function check({ repo, sub }) {
  const engine = boot();
  engine.catch(() => {});
  const pre = sub ? sub + "/" : "";
  progress("info");
  let info, infoBytes;
  try {
    const r = await get(`${HF}/datasets/${repo}/resolve/main/${quote(pre + "meta/info.json")}`, "info");
    infoBytes = new Uint8Array(await r.arrayBuffer());
    info = JSON.parse(new TextDecoder().decode(infoBytes));
  } catch (e) {
    if (e instanceof Fail && e.kind === "missing") throw new Fail("missing", sub ? "subfolder" : "dataset");
    if (e instanceof Fail) throw e;
    throw new Fail("layout", "The dataset's meta/info.json is not readable JSON.");
  }
  const feats = (info && info.features) || {};
  if (!COLS.every((c) => feats[c])) throw new Fail("layout", "no-columns", { robot: info.robot_type || null });
  const listing = tree(repo, pre + "data");
  listing.catch(() => {});
  // The command-line tool decides a skip from meta/info.json before reading any data; so does the page.
  progress("engine-wait");
  const p = await engine;
  const pc = p.globals.get("engine").precheck;
  const skipped = pc(repo, sub, infoBytes);
  pc.destroy();
  if (skipped) return JSON.parse(skipped);
  progress("list");
  const files = (await listing).filter((x) => x.type === "file" && x.path.endsWith(".parquet"));
  if (!files.length) throw new Fail("layout", "no-data");
  const total = files.reduce((s, x) => s + (x.size || 0), 0);
  const frames = Number(info.total_frames) || 0;
  if (frames > MAX_FRAMES || total > MAX_BYTES) throw new Fail("large", "large", { frames, total });
  progress("data", { done: 0, total, file: 0, files: files.length });
  const blobs = await download(repo, files, total);
  progress("measure", { frames });
  // Let the progress message paint before Python holds the worker.
  await new Promise((r) => setTimeout(r, 30));
  const run = p.globals.get("engine").run;
  const out = run(repo, sub, infoBytes, files.map((f) => f.path), blobs);
  run.destroy();
  return JSON.parse(out);
}

self.onmessage = async (ev) => {
  const m = ev.data || {};
  if (m.type === "warm") {
    boot().then(() => post("warm"), () => {});
    return;
  }
  if (m.type !== "check") return;
  try {
    const view = await check(m);
    post("result", { view });
  } catch (e) {
    if (e instanceof Fail) post("fail", { kind: e.kind, message: e.message, status: e.status, frames: e.frames, total: e.total, robot: e.robot });
    else post("fail", { kind: "engine", message: String((e && e.message) || e).slice(-400) });
  }
};
