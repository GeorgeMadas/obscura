/* Obscura: frontend for the Qwen Image 2.1 studio. No framework, no build step. */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const h = (tag, attrs = {}, ...kids) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "style") el.style.cssText = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else if (k === "html") el.innerHTML = v;
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(kid));
    return el;
  };
  const icon = (name, fill) => h("i", { class: `${fill ? "ph-fill" : "ph"} ph-${name}`, "aria-hidden": "true" });
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const store = {
    get(k, d) { try { const v = localStorage.getItem("obscura:" + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("obscura:" + k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
  };

  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    let body = null;
    try { body = await res.json(); } catch { /* empty body */ }
    if (!res.ok) throw new Error((body && (body.error || body.detail)) || `Request failed (${res.status})`);
    return body;
  }

  // ---------------------------------------------------------------- state
  const S = {
    cfg: null,
    mode: "Text to Image",
    ratio: "9:16",
    steps: 40,
    lora: null,
    strength: 1,
    seed: 42,
    seedLocked: false,
    res: "4MP",
    batch: 1,                      // images per click (random seed only)
    jobBatch: new Map(),           // job id -> batch record { total, left }
    status: null,                  // server model status
    refs: [],                      // [{ blob, url }] in order: image 1, image 2, ... (Blend takes up to 4)
    feed: { items: [], total: 0, done: false, loading: false, loaded: false },
    failed: [],                    // { id, error, aspect } render failures shown in the grid until dismissed
    jobs: new Map(),
    lib: { items: [], total: 0, q: "", loading: false, done: false, loaded: false },
    liked: { items: [], loading: false, loaded: false },
    covers: {},
    styleFilter: "All",
  };

  const PLACEHOLDERS = {
    "Text to Image": "A lighthouse at blue hour, fog rolling over black rocks…",
    "Edit Image (1 Ref)": "Describe the change, like “make it golden hour and add light rain”",
    "Transform & Swap (2 Refs)": "Describe how to combine them, like “put the jacket from image 2 on the person in image 1”",
    "Transparent PNG": "A glass perfume bottle, studio lit, isolated subject",
  };
  const IDEAS = [
    ["A neon cyberpunk samurai in a rain-soaked Tokyo alley, glowing katana, volumetric mist", "Anime Consistency"],
    ["A highland lake surrounded by autumn pines at golden hour, still reflections", "Natural Exposure (Photorealism)"],
    ["A small steampunk robot holding a glass flower, brass gears, macro detail", "Viggle Turbo (4-Step Acceleration)"],
  ];
  const CAT_ICONS = {
    Base: "circle-half", Style: "palette", "Turbo Speed": "lightning", Transform: "magic-wand",
    Lighting: "sun-dim", Color: "film-strip", Utility: "arrows-out", "Two Images": "users-three", Custom: "plug",
  };

  // ---------------------------------------------------------------- helpers
  const modeInfo = (m = S.mode) => S.cfg.modes.find((x) => x.value === m);
  const loraSpec = (id) => S.cfg.loras.find((l) => l.id === id);
  function ratioInfo(r = S.ratio, res = S.res) {
    const x = S.cfg.ratios.find((y) => y.ratio === r) || S.cfg.ratios[0];
    const [width, height] = x.sizes[res] || x.sizes["1MP"];
    return { ratio: x.ratio, width, height };
  }
  const megapixels = (r) => ((r.width * r.height) / 1e6).toFixed(1);
  function styleName(id) {
    if (!id || id === S.cfg.none_lora || id === "None") return "Base model";
    if (id === S.cfg.custom_lora) return "Hugging Face LoRA";
    if (id === S.cfg.local_lora) return "Local LoRA";
    const spec = loraSpec(id);
    return spec ? spec.name : id;
  }
  function styleCategory(id) {
    if (!id || id === S.cfg.none_lora) return "Base";
    if (id === S.cfg.custom_lora || id === S.cfg.local_lora) return "Custom";
    return (loraSpec(id) || {}).category || "Style";
  }
  const isVideo = (rec) => rec && rec.media === "video";
  const isComfy = (rec) => rec && rec.kind === "comfy";
  const fmtDur = (s) => (s == null ? "" : s < 60 ? `${s.toFixed(1)}s` : `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`);
  // Video tiles show their first frame and play muted while hovered
  function hoverPlay(el, rec) {
    let v = null;
    el.addEventListener("pointerenter", () => {
      if (v || reduceMotion) return;
      v = h("video", { class: "tile-video", src: rec.url, muted: true, loop: true, playsinline: true, autoplay: true, preload: "auto" });
      v.muted = true;
      v.addEventListener("playing", () => v && v.classList.add("on"));
      // Right above the poster image, below the badges and hover actions
      el.querySelector("img").after(v);
      v.play().catch(() => {});
    });
    el.addEventListener("pointerleave", () => { if (v) { v.pause(); v.remove(); v = null; } });
  }
  const videoBadge = (rec) => h("span", { class: "video-badge" }, icon("play", true), rec.duration ? fmtDur(rec.duration) : "Video");
  const fmtSecs = (s) => (s < 60 ? `${Math.max(1, Math.round(s))}s` : `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`);
  const fmtDate = (t) => (t ? new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(t * 1000)) : "");
  const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

  // ---------------------------------------------------------------- toasts
  function toast(msg, { type = "info", action, timeout = 4200 } = {}) {
    const icons = { info: "info", error: "warning-circle", ok: "check-circle" };
    const el = h("div", { class: `toast ${type}`, role: type === "error" ? "alert" : "status" }, icon(icons[type], true), h("span", {}, msg));
    const close = () => { el.classList.add("out"); setTimeout(() => el.remove(), 250); };
    if (action) el.append(h("button", { class: "btn btn-quiet", type: "button", onclick: () => { action.fn(); close(); } }, action.label));
    $("#toasts").append(el);
    setTimeout(close, timeout);
    return close;
  }

  // ---------------------------------------------------------------- router
  const VIEWS = ["create", "liked", "library", "styles", "prompts", "workflows"];
  function route() {
    const [path] = location.hash.replace(/^#\/?/, "").split("?");
    const view = VIEWS.includes(path) ? path : "create";
    const swap = () => {
      $("#app").dataset.view = view;
      for (const v of VIEWS) $(`#view-${v}`).hidden = v !== view;
      $$(".rail-link").forEach((a) => a.classList.toggle("active", a.dataset.nav === view));
      if (view === "library") ensureLibrary();
      if (view === "liked") ensureLiked();
      if (view === "styles") renderStyles();
      if (view === "prompts") ensurePrompts();
      if (view === "workflows") ensureWorkflows(); else stopWfStatus();
      if (view === "create") requestAnimationFrame(() => { placeThumb(); if (feedColumns() !== FEED.cols) renderFeed(); });
      closeInspector();
    };
    if (document.startViewTransition && !reduceMotion && $("#app").dataset.view !== view) document.startViewTransition(swap);
    else swap();
  }
  const go = (view) => { location.hash = `#/${view}`; };

  // ---------------------------------------------------------------- mode switch
  function buildModes() {
    const wrap = $("#modeSwitch");
    wrap.append(h("span", { class: "thumb", "aria-hidden": "true" }));
    for (const m of S.cfg.modes) {
      wrap.append(h("button", {
        class: "seg-btn", type: "button", role: "tab", "data-mode": m.value, title: m.blurb,
        onclick: () => setMode(m.value),
      }, icon(m.icon), h("span", {}, m.label)));
    }
  }
  function placeThumb() {
    const btn = $(`.seg-btn[aria-selected="true"]`);
    const thumb = $("#modeSwitch .thumb");
    if (!btn || !thumb || !btn.offsetWidth) return;
    thumb.style.width = btn.offsetWidth + "px";
    thumb.style.transform = `translateX(${btn.offsetLeft}px)`;
  }
  function setMode(value, { quiet } = {}) {
    S.mode = value;
    $$(".seg-btn").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.mode === value)));
    placeThumb();
    $("#prompt").placeholder = PLACEHOLDERS[value];
    const spec = loraSpec(S.lora);
    if (spec && spec.two_images && value !== "Transform & Swap (2 Refs)") {
      selectStyle(S.cfg.none_lora, { quiet: true });
      if (!quiet) toast(`${spec.name} needs two images, so the style was reset.`);
    }
    renderRefs();
    save();
  }

  // ---------------------------------------------------------------- reference images
  const BLEND = "Transform & Swap (2 Refs)";
  const maxRefs = () => modeInfo().refs;
  function renderRefs() {
    const max = maxRefs();
    const box = $("#refs");
    box.hidden = max === 0;
    box.replaceChildren();
    if (!max) return;
    const blend = S.mode === BLEND;
    const used = Math.min(S.refs.length, max);
    // Blend: every filled slot, plus one empty slot until the limit (at least two slots in total)
    const slots = blend ? Math.min(max, Math.max(2, used + 1)) : 1;
    for (let i = 0; i < slots; i++) {
      const ref = i < used ? S.refs[i] : null;
      const label = blend ? `Image ${i + 1}` : "Image to edit";
      const slot = ref
        ? h("div", { class: "ref-slot", title: label },
            h("img", { src: ref.url, alt: label }),
            h("span", { class: "tag" }, String(i + 1)),
            h("button", { class: "remove", type: "button", "aria-label": `Remove ${label.toLowerCase()}`, onclick: () => setRef(i, null) }, icon("x")))
        : h("button", { class: "ref-slot empty", type: "button", onclick: () => pickFile(i) }, h("span", {}, icon("plus"), i >= 2 ? "Add image" : label));
      slot.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); slot.classList.add("over"); hideDrop(); });
      slot.addEventListener("dragleave", () => slot.classList.remove("over"));
      slot.addEventListener("drop", (e) => {
        e.preventDefault(); e.stopPropagation(); slot.classList.remove("over"); hideDrop();
        const files = [...(e.dataTransfer.files || [])].filter((x) => x.type.startsWith("image/"));
        if (!files.length) return;
        setRef(i, files[0]);
        if (blend) files.slice(1).forEach((f) => { if (S.refs.length < max) setRef(S.refs.length, f); });
      });
      box.append(slot);
    }
    const note = blend
      ? `Refer to them as image 1, image 2 and so on in your prompt. Up to ${max} images.`
      : "Drop, paste or pick the image you want to change.";
    box.append(h("p", { class: "refs-note" }, note));
  }
  // Replace slot i, append when i is past the end, or remove it (later images move up one number)
  function setRef(i, blob) {
    if (!blob) {
      const [gone] = S.refs.splice(i, 1);
      if (gone) URL.revokeObjectURL(gone.url);
    } else {
      const ref = { blob, url: URL.createObjectURL(blob) };
      if (i < S.refs.length) { URL.revokeObjectURL(S.refs[i].url); S.refs[i] = ref; } else S.refs.push(ref);
    }
    renderRefs();
  }
  let pickSlot = 0;
  function pickFile(i) { pickSlot = i; $("#filePicker").click(); }
  function addImages(files) {
    const imgs = files.filter((f) => f.type.startsWith("image/"));
    if (!imgs.length) return toast("That file is not an image.", { type: "error" });
    if (imgs.length > 1 || S.mode === BLEND) {
      if (S.mode !== BLEND) setMode(BLEND, { quiet: true });
      const max = maxRefs();
      const room = max - S.refs.length;
      imgs.slice(0, Math.max(0, room)).forEach((f) => setRef(S.refs.length, f));
      if (imgs.length > room) toast(`Blend holds up to ${max} images. Remove one to add another.`, { type: room > 0 ? "info" : "error" });
      else toast(imgs.length > 1 ? `${imgs.length} references added.` : "Reference added.", { type: "ok" });
    } else {
      if (S.mode !== "Edit Image (1 Ref)") setMode("Edit Image (1 Ref)", { quiet: true });
      setRef(0, imgs[0]);
      toast("Reference added.", { type: "ok" });
    }
    if (location.hash.indexOf("create") < 0 && location.hash !== "") go("create");
    $("#prompt").focus();
  }

  // ---------------------------------------------------------------- styles
  function coverFor(id, cat) {
    const url = S.covers[id];
    return url
      ? h("img", { src: url, alt: "", loading: "lazy" })
      : h("div", { class: "art", "data-cat": cat }, icon(CAT_ICONS[cat] || "palette"));
  }
  function allStyles() {
    const c = S.cfg;
    return [
      { id: c.none_lora, name: "Base model", category: "Base", blurb: "Qwen Image 2.1 on its own. Most faithful to your prompt.", steps: 40, strength: null, loaded: true },
      ...c.loras,
      { id: c.custom_lora, name: "Hugging Face LoRA", category: "Custom", blurb: "Load any Qwen Image LoRA by repository and filename.", custom: true },
      { id: c.local_lora, name: "Local LoRA", category: "Custom", blurb: `Use a .safetensors file from ${c.lora_dir}.`, custom: true, loaded: true },
    ];
  }
  function selectStyle(id, { quiet } = {}) {
    const prev = { lora: S.lora, prompt: $("#prompt").value, steps: S.steps, strength: S.strength };
    S.lora = id;
    const spec = loraSpec(id);
    let promptChanged = false;
    if (spec) {
      S.steps = spec.steps;
      S.strength = spec.strength;
      const p = $("#prompt").value;
      if (spec.preset && !p.includes(spec.preset)) {
        $("#prompt").value = p.trim() ? `${p.trim()}, ${spec.preset}` : spec.preset;
        promptChanged = true;
        autosize();
      }
      if (spec.two_images && S.mode !== "Transform & Swap (2 Refs)") setMode("Transform & Swap (2 Refs)", { quiet: true });
    } else {
      S.strength = 1;
    }
    syncControls();
    if (!quiet) {
      const undo = () => {
        S.lora = prev.lora; S.steps = prev.steps; S.strength = prev.strength;
        $("#prompt").value = prev.prompt; autosize(); syncControls();
      };
      toast(promptChanged ? `${styleName(id)} selected. Its prompt cue was added.` : `${styleName(id)} selected.`,
        { type: "ok", action: { label: "Undo", fn: undo } });
    }
  }
  function renderStyleCurrent() {
    const cat = styleCategory(S.lora);
    const spec = loraSpec(S.lora);
    const sub = spec ? spec.blurb : S.lora === S.cfg.none_lora ? "No LoRA applied" : "Custom adapter";
    $("#styleCurrent").replaceChildren(
      h("span", { class: "cover", style: "position:relative" }, coverFor(S.lora, cat)),
      h("span", { class: "t" }, h("strong", {}, styleName(S.lora)), h("small", {}, sub)),
      icon("caret-right"),
    );
    $("#customLora").hidden = S.lora !== S.cfg.custom_lora;
    $("#localLora").hidden = S.lora !== S.cfg.local_lora;
    $("#strengthField").hidden = S.lora === S.cfg.none_lora;
  }
  function renderStyles() {
    const styles = allStyles();
    const cats = ["All", ...new Set(styles.map((s) => s.category))];
    const count = (c) => (c === "All" ? styles.length : styles.filter((s) => s.category === c).length);
    $("#styleFilters").replaceChildren(...cats.map((c) => h("button", {
      class: "filter", role: "tab", type: "button", "aria-selected": String(c === S.styleFilter),
      onclick: () => { S.styleFilter = c; renderStyles(); },
    }, c === "Turbo Speed" ? "Turbo" : c, h("span", { class: "n" }, String(count(c))))));
    const shown = styles.filter((s) => S.styleFilter === "All" || s.category === S.styleFilter);
    $("#styleGrid").replaceChildren(...shown.map((s, i) => {
      const selected = s.id === S.lora;
      const specs = [];
      if (s.steps) specs.push(h("span", { class: "badge" }, icon("stack"), `${s.steps} steps`));
      if (s.strength) specs.push(h("span", { class: "badge" }, `strength ${s.strength}`));
      if (s.category === "Turbo Speed") specs.push(h("span", { class: "badge accent" }, icon("lightning", true), "Fast"));
      if (s.two_images) specs.push(h("span", { class: "badge" }, icon("images"), "2 images"));
      if (!s.custom && s.id !== S.cfg.none_lora && s.loaded) specs.push(h("span", { class: "badge" }, icon("check"), "Ready"));
      return h("button", {
        class: `style-card${selected ? " selected" : ""}`, type: "button", style: `--i:${i}`,
        "aria-pressed": String(selected),
        onclick: () => { selectStyle(s.id); go("create"); },
      },
      h("div", { class: "cover" }, coverFor(s.id, s.category)),
      h("div", { class: "body" },
        h("h3", {}, s.name, selected ? icon("check-circle", true) : null),
        h("p", {}, s.blurb),
        h("div", { class: "specs" }, specs)));
    }));
    $$("#styleGrid .ph-check-circle").forEach((i) => i.classList.add("check"));
  }

  // ---------------------------------------------------------------- saved prompts
  const P = { items: [], loaded: false, editing: null };
  async function copyPrompt(text) {
    let ok = false;
    try { await navigator.clipboard.writeText(text); ok = true; }
    catch {
      // navigator.clipboard only exists on https or localhost; fall back to a hidden textarea
      const ta = h("textarea", { style: "position:fixed;top:0;left:0;opacity:0", readonly: true });
      ta.value = text;
      document.body.append(ta);
      ta.select();
      try { ok = document.execCommand("copy"); } catch { /* unsupported */ }
      ta.remove();
    }
    toast(ok ? "Prompt copied." : "Clipboard is not available here.", { type: ok ? "ok" : "error" });
  }
  async function ensurePrompts() {
    if (!P.loaded) {
      try { P.items = (await api("/api/prompts")).items; P.loaded = true; }
      catch (e) { toast(e.message, { type: "error" }); }
    }
    renderPrompts();
  }
  function openPromptEditor(item = null) {
    P.editing = item;
    $("#promptTitle").value = item ? item.title : "";
    $("#promptText").value = item ? item.prompt : "";
    $("#promptSave").textContent = item ? "Save changes" : "Save prompt";
    $("#promptEditor").hidden = false;
    $("#promptsEmpty").hidden = true;
    $("#promptTitle").focus();
  }
  function closePromptEditor() {
    P.editing = null;
    $("#promptEditor").hidden = true;
    renderPrompts();
  }
  async function savePrompt() {
    const body = JSON.stringify({ title: $("#promptTitle").value.trim(), prompt: $("#promptText").value.trim() });
    const opts = { headers: { "Content-Type": "application/json" }, body };
    try {
      if (P.editing) {
        const item = await api(`/api/prompts/${P.editing.id}`, { ...opts, method: "PUT" });
        P.items = P.items.map((x) => (x.id === item.id ? item : x));
      } else {
        P.items.unshift(await api("/api/prompts", { ...opts, method: "POST" }));
      }
      toast("Prompt saved.", { type: "ok" });
      closePromptEditor();
    } catch (e) { toast(e.message, { type: "error" }); }
  }
  async function deletePrompt(item) {
    try {
      await api(`/api/prompts/${item.id}`, { method: "DELETE" });
      P.items = P.items.filter((x) => x.id !== item.id);
      renderPrompts();
    } catch (e) { toast(e.message, { type: "error" }); }
  }
  // "Show full prompt" only appears on prompts that are actually cut off
  function measurePrompts() {
    $$("#promptList .prompt-card").forEach((card) => {
      const body = $(".prompt-body", card);
      if (!body.classList.contains("clamped")) return;
      $(".prompt-more", card).hidden = body.scrollHeight <= body.clientHeight + 1;
    });
  }
  function renderPrompts() {
    $("#promptsEmpty").hidden = !P.loaded || P.items.length > 0 || !$("#promptEditor").hidden;
    $("#promptList").replaceChildren(...P.items.map((item, i) => {
      const body = h("p", { class: "prompt-body clamped" }, item.prompt);
      const more = h("button", {
        class: "link-sm prompt-more", type: "button", hidden: true, "aria-expanded": "false",
        onclick: () => {
          const open = body.classList.toggle("clamped") === false;
          more.classList.toggle("open", open);
          more.setAttribute("aria-expanded", String(open));
          more.lastChild.textContent = open ? "Collapse" : "Show full prompt";
        },
      }, icon("caret-down"), h("span", {}, "Show full prompt"));
      const trash = h("button", {
        class: "icon-btn", type: "button", "aria-label": "Delete", title: "Delete",
        onclick: () => {
          // Two clicks: the first arms the button, the second deletes
          if (!trash.classList.contains("confirm")) {
            trash.classList.add("confirm");
            trash.title = "Click again to delete";
            setTimeout(() => { trash.classList.remove("confirm"); trash.title = "Delete"; }, 3000);
            return;
          }
          deletePrompt(item);
        },
      }, icon("trash"));
      return h("article", { class: "prompt-card", style: `--i:${i}` },
        h("header", {},
          h("h3", {}, item.title),
          h("div", { class: "prompt-card-tools" },
            h("button", { class: "icon-btn", type: "button", "aria-label": "Edit", title: "Edit", onclick: () => openPromptEditor(item) }, icon("pencil-simple")),
            trash)),
        body,
        h("footer", {},
          more,
          h("div", { class: "prompt-card-actions" },
            h("button", {
              class: "btn btn-ghost btn-sm", type: "button", title: "Put this prompt in the Create composer",
              onclick: () => { $("#prompt").value = item.prompt; go("create"); requestAnimationFrame(autosize); },
            }, icon("arrow-up-right"), "Use"),
            h("button", { class: "btn btn-quiet btn-sm", type: "button", onclick: () => copyPrompt(item.prompt) }, icon("copy"), "Copy"))));
    }));
    requestAnimationFrame(measurePrompts);
  }

  // ---------------------------------------------------------------- inspector controls
  function buildRatios() {
    $("#ratios").replaceChildren(...S.cfg.ratios.map(({ ratio }) => {
      const r = ratioInfo(ratio);
      const max = 22;
      const w = r.width >= r.height ? max : Math.round((max * r.width) / r.height);
      const hh = r.height >= r.width ? max : Math.round((max * r.height) / r.width);
      return h("button", {
        class: "ratio", type: "button", role: "radio", "data-ratio": r.ratio,
        onclick: () => { S.ratio = r.ratio; syncControls(); },
      }, h("span", { class: "shape", style: `width:${w}px;height:${hh}px` }), r.ratio);
    }));
  }
  function setRangeFill(el) {
    const p = ((el.value - el.min) / (el.max - el.min)) * 100;
    el.style.setProperty("--p", p + "%");
  }
  function syncControls() {
    $$(".ratio").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.ratio === S.ratio)));
    const r = ratioInfo();
    $("#ratioVal").textContent = `${r.width} × ${r.height}`;
    $$(".ratio").forEach((b) => { const x = ratioInfo(b.dataset.ratio); b.title = `${x.width} × ${x.height}`; });
    $$("#resOpts .opt").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.res === S.res)));
    $("#resHint").textContent = S.res === "4MP"
      ? "Qwen's native size. Sharper detail, about 6 times slower per render."
      : "Fast previews. Switch to 4 MP for final images.";
    const steps = $("#steps"); steps.value = S.steps; $("#stepsVal").textContent = S.steps; setRangeFill(steps);
    const st = $("#strength"); st.value = S.strength; $("#strengthVal").textContent = Number(S.strength).toFixed(2); setRangeFill(st);
    const lock = $("#seedLock");
    lock.setAttribute("aria-pressed", String(S.seedLocked));
    lock.replaceChildren(icon(S.seedLocked ? "lock-simple" : "dice-five"), h("span", {}, S.seedLocked ? "Fixed" : "Random"));
    const count = S.seedLocked ? 1 : S.batch;
    $$("#batchOpts .count").forEach((b) => {
      b.setAttribute("aria-checked", String(+b.dataset.n === count));
      b.disabled = S.seedLocked && b.dataset.n !== "1";
    });
    $("#batchVal").textContent = count === 1 ? "1 image" : `${count} images`;
    $("#batchHint").textContent = S.seedLocked
      ? "A fixed seed renders one image per click, since every copy would be identical. Switch the seed to Random for batches."
      : count > 1 ? "Each image gets its own random seed and joins the queue." : "";
    $("#generateBtn span").textContent = count > 1 ? `Generate ${count}` : "Generate";
    $("#seed").value = S.seed;
    const spec = loraSpec(S.lora);
    const turbo = spec && spec.category === "Turbo Speed";
    const hint = $("#stepsHint");
    hint.className = "hint";
    if (turbo && S.steps > 8) { hint.textContent = "Turbo styles are tuned for about 4 steps."; hint.classList.add("accent"); }
    else if (!turbo && S.steps <= 8) { hint.textContent = "Very low step counts look rough without a Turbo style."; hint.classList.add("accent"); }
    else hint.textContent = "";
    renderStyleCurrent();
    save();
  }
  function focusField(sel) {
    openInspector();
    const el = $(sel);
    el.scrollIntoView({ block: "center", behavior: reduceMotion ? "auto" : "smooth" });
    const target = el.matches("input") ? el : el.querySelector("[aria-checked=true]") || el.querySelector("button");
    setTimeout(() => target && target.focus({ preventScroll: true }), 250);
  }
  const narrow = () => matchMedia("(max-width: 1180px)").matches;
  function openInspector() {
    if (!narrow()) return setCollapsed(false);
    $("#inspector").classList.add("open");
    $("#scrim").hidden = false;
  }
  function setCollapsed(on) {
    $("#app").classList.toggle("inspector-collapsed", on);
    $("#inspector").inert = on && !narrow();
    $("#settingsToggle").setAttribute("aria-expanded", String(!on));
    store.set("inspectorCollapsed", on);
  }
  function toggleSettings() {
    if (narrow()) return $("#inspector").classList.contains("open") ? closeInspector() : openInspector();
    setCollapsed(!$("#app").classList.contains("inspector-collapsed"));
  }
  function closeInspector() { $("#inspector").classList.remove("open"); $("#scrim").hidden = true; }

  function save() {
    store.set("prefs", { v: 3, mode: S.mode, ratio: S.ratio, res: S.res, steps: S.steps, lora: S.lora, strength: S.strength, seed: S.seed, seedLocked: S.seedLocked, batch: S.batch,
      customRepo: $("#customRepo").value, customFile: $("#customFile").value, localLora: $("#localLoraInput").value });
  }

  // ---------------------------------------------------------------- resolution
  function buildResolution() {
    const opts = [["1MP", "1 MP", "Fast"], ["4MP", "4 MP", "Native"]];
    $("#resOpts").replaceChildren(...opts.map(([res, label, sub]) => h("button", {
      class: "opt", type: "button", role: "radio", "data-res": res,
      onclick: () => { S.res = res; syncControls(); },
    }, h("strong", {}, label), h("small", {}, sub))));
  }

  // ---------------------------------------------------------------- model switching
  const modelFor = (file) => (S.status ? S.status.models.find((m) => m.file === file) : null);
  let statusTimer = null;
  function renderModels() {
    const st = S.status;
    const sw = st.switch;
    const loading = sw.state === "loading";
    $("#modelOpts").replaceChildren(...st.models.map((m) => h("button", {
      class: `opt${loading && sw.target === m.file ? " pending" : ""}`, type: "button", role: "radio",
      "aria-checked": String(m.file === st.checkpoint),
      disabled: loading || !m.available || null,
      title: m.available ? m.file : "Not downloaded on this machine",
      onclick: () => switchModel(m),
    }, h("strong", {}, m.label), h("small", {}, m.available ? `${m.detail}  ${m.size_gb} GB` : "Not downloaded"))));
    const active = modelFor(st.checkpoint);
    $("#modelVal").textContent = active ? active.detail : "";
    $("#modelTag").textContent = `Qwen Image 2.1${active && active.id === "small" ? "  Small" : ""}`;
    $("#engineInfo").textContent = st.checkpoint;
    const box = $("#modelStatus");
    if (loading) {
      const target = modelFor(sw.target);
      const secs = sw.started ? Math.round(sw.server_time - sw.started) : 0;
      box.className = "model-status";
      box.replaceChildren(h("span", { class: "queue-ring" }),
        h("span", {}, h("b", {}, `Loading ${target ? target.label : "model"}`), h("br"), sw.phase || "Starting"),
        h("span", { class: "mono" }, `${secs}s`));
      box.hidden = false;
    } else if (sw.state === "error") {
      box.className = "model-status error";
      box.replaceChildren(icon("warning-circle", true), h("span", {}, sw.error));
      box.hidden = false;
    } else box.hidden = true;
  }
  async function switchModel(m) {
    if (m.file === S.status.checkpoint) return;
    try {
      S.status = await api("/api/model", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ checkpoint: m.file }) });
      renderModels();
      toast(`Switching to the ${m.label} model. New renders wait until it is ready.`);
      pollStatus();
    } catch (e) { toast(e.message, { type: "error" }); }
  }
  function pollStatus() {
    if (statusTimer) return;
    statusTimer = setInterval(async () => {
      try { S.status = await api("/api/status"); } catch { return; }
      renderModels();
      if (S.status.switch.state !== "loading") {
        clearInterval(statusTimer); statusTimer = null;
        if (S.status.switch.state === "idle") {
          const m = modelFor(S.status.checkpoint);
          toast(`${m ? m.label : "New"} model is ready.`, { type: "ok" });
          // Styles must reload onto the new weights
          S.cfg.loras.forEach((l) => { l.loaded = false; });
        } else toast(S.status.switch.error || "Model switch failed.", { type: "error", timeout: 8000 });
      }
    }, 1500);
  }

  // ---------------------------------------------------------------- prompt
  function autosize() {
    const t = $("#prompt");
    t.style.height = "auto";
    t.style.height = Math.min(t.scrollHeight, 180) + "px";
  }

  // ---------------------------------------------------------------- generation
  // rewrite: route the composer prompt through the Qwen rewriter popup first (only the composer asks for this)
  // prompt/original: set when the popup confirms a rewritten prompt
  async function generate({ seedOverride, rewrite = false, prompt: confirmed, original = "", refs = S.refs } = {}) {
    const prompt = (confirmed ?? $("#prompt").value).trim();
    const need = S.mode === BLEND ? Math.min(refs.length, maxRefs()) : maxRefs();
    const composer = $("#composer");
    const fail = (msg) => {
      composer.classList.remove("shake"); void composer.offsetWidth; composer.classList.add("shake");
      toast(msg, { type: "error" });
    };
    if (!prompt) { $("#prompt").focus(); return fail("Describe what you want to see first."); }
    if (S.mode === BLEND && need < 2) return fail("Blend needs at least two reference images.");
    for (let i = 0; i < need; i++) if (!refs[i]) return fail("Add the image you want to edit.");
    if (S.lora === S.cfg.custom_lora && (!$("#customRepo").value.trim() || !$("#customFile").value.trim())) {
      focusField("#customRepo"); return fail("Enter the Hugging Face repo and weights file.");
    }
    if (S.lora === S.cfg.local_lora && !$("#localLoraInput").value.trim()) { focusField("#localLoraInput"); return fail("Pick a local LoRA file."); }

    if (rewrite && S.rewriteOn && !RW.done.has(prompt)) return openRewrite(prompt, need, seedOverride);

    const random = seedOverride == null && !S.seedLocked;
    const count = random ? S.batch : 1;
    const params = {
      prompt, original_prompt: original, mode: S.mode, ratio: S.ratio, resolution: S.res, steps: S.steps, lora: S.lora, lora_strength: S.strength,
      seed: seedOverride ?? S.seed, randomize_seed: random,
      custom_repo: $("#customRepo").value.trim(), custom_file: $("#customFile").value.trim(), local_lora: $("#localLoraInput").value.trim(),
    };
    const btn = $("#generateBtn");
    btn.disabled = true;
    const batch = count > 1 ? { total: count, left: count } : null;
    let sent = 0;
    try {
      // One job per image: each gets its own random seed, progress tile and queue slot
      for (let n = 0; n < count; n++) {
        const fd = new FormData();
        fd.append("params", JSON.stringify(params));
        for (let i = 0; i < need; i++) fd.append(`ref${i + 1}`, refs[i].blob, `ref${i + 1}.png`);
        const job = await api("/api/generate", { method: "POST", body: fd });
        sent++;
        S.jobs.set(job.id, job);
        if (batch) S.jobBatch.set(job.id, batch);
        renderFeed();
        startPolling();
      }
      scrollFeedTop();
    } catch (e) {
      // Only the images that reached the queue count toward the batch
      if (batch) { batch.total -= count - sent; batch.left -= count - sent; }
      fail(e.message);
    } finally {
      setTimeout(() => { btn.disabled = false; }, 250);
    }
  }

  // ---------------------------------------------------------------- prompt rewrite (official Qwen PE-T2I / PE-I2I)
  const RW = {
    job: null,          // rewrite job id being shown
    pending: null,      // { prompt, need, refs, seedOverride, key } the popup will generate with
    timer: null,
    cache: new Map(),   // key -> result, so reopening the same prompt doesn't wait again
    done: new Set(),    // prompts that already are rewrites; generating them again skips the popup
    suggested: null,    // ratio the rewriter suggests, mapped to one the app supports
    min: false,         // popup minimized to the dock while the rewrite runs in the background
  };
  const blobIds = new WeakMap();
  let blobSeq = 0;
  const blobId = (b) => { if (!blobIds.has(b)) blobIds.set(b, ++blobSeq); return blobIds.get(b); };
  const rwKey = (prompt, refs) => JSON.stringify([S.mode, prompt, refs.map((r) => blobId(r.blob))]);

  function setRewriteOn(on) {
    S.rewriteOn = on;
    store.set("rewrite", on);
    const b = $("#rewriteToggle");
    b.setAttribute("aria-pressed", String(on));
    b.dataset.tip = on ? "Rewrite with Qwen: on" : "Rewrite with Qwen: off";
  }

  function rwShow(state) {
    $("#rwLoading").hidden = state !== "loading";
    $("#rwError").hidden = state !== "error";
    $("#rwResult").hidden = state !== "done";
    $("#rwAgain").hidden = state === "loading";
    $("#rwConfirm").disabled = state !== "done";
  }

  function minimizeRewrite() {
    RW.min = true;
    $("#rewrite").hidden = true;
    $("#rwDock").hidden = false;
    $("#rwDockText").textContent = $("#rwPhase").textContent || "Rewriting prompt";
    $("#rwDockTime").textContent = $("#rwTime").textContent;
    $("#prompt").focus();
  }

  function expandRewrite() {
    RW.min = false;
    $("#rwDock").hidden = true;
    $("#rewrite").hidden = false;
  }

  function openRewrite(prompt, need, seedOverride) {
    const refs = S.refs.slice(0, need);
    const key = rwKey(prompt, refs);
    // Same prompt again while it rewrites minimized: just bring it back. A different prompt replaces it.
    if (RW.min && RW.pending && RW.pending.key === key) return expandRewrite();
    stopRewrite({ cancel: true });
    if (RW.min) expandRewrite();
    RW.pending = { prompt, need, refs, seedOverride, key };
    $("#rwBadge").textContent = refs.length ? "PE-I2I" : "PE-T2I";
    $("#rwOriginal").textContent = prompt;
    $("#rewrite").hidden = false;
    const cached = RW.cache.get(key);
    if (cached) showRewrite(cached);
    else startRewrite();
  }

  async function startRewrite() {
    const pend = RW.pending;
    stopRewrite();
    rwShow("loading");
    $("#rwPhase").textContent = "Starting";
    $("#rwTime").textContent = "";
    $("#rwThinking").textContent = "";
    const fd = new FormData();
    fd.append("params", JSON.stringify({ prompt: pend.prompt, mode: S.mode }));
    pend.refs.forEach((r, i) => fd.append(`ref${i + 1}`, r.blob, `ref${i + 1}.png`));
    try {
      const job = await api("/api/rewrite", { method: "POST", body: fd });
      if (RW.pending !== pend) { api(`/api/rewrite/${job.id}/cancel`, { method: "POST" }).catch(() => {}); return; }
      RW.job = job.id;
      const t0 = Date.now() - (job.server_time - job.created) * 1000;
      RW.timer = setInterval(() => pollRewrite(job.id, t0), 700);
    } catch (e) { rewriteFailed(e.message); }
  }

  async function pollRewrite(id, t0) {
    $("#rwTime").textContent = $("#rwDockTime").textContent = fmtSecs((Date.now() - t0) / 1000);
    let job;
    try { job = await api(`/api/rewrite/${id}`); } catch (e) { if (RW.job === id) { stopRewrite(); rewriteFailed(e.message); } return; }
    if (RW.job !== id) return;
    $("#rwPhase").textContent = $("#rwDockText").textContent = job.status === "queued" ? "Waiting for another rewrite" : job.phase;
    $("#rwThinking").textContent = job.thinking || "";
    if (job.status === "done") {
      stopRewrite();
      RW.cache.set(RW.pending.key, job.result);
      if (RW.cache.size > 30) RW.cache.delete(RW.cache.keys().next().value);
      if (RW.min) expandRewrite();
      showRewrite(job.result);
    } else if (job.status === "error") {
      stopRewrite(); rewriteFailed(job.error);
      if (RW.min) expandRewrite();
    } else if (job.status === "cancelled") {
      stopRewrite();
    }
  }

  function stopRewrite({ cancel = false } = {}) {
    clearInterval(RW.timer); RW.timer = null;
    if (cancel && RW.job) api(`/api/rewrite/${RW.job}/cancel`, { method: "POST" }).catch(() => {});
    RW.job = null;
  }

  function rewriteFailed(msg) {
    $("#rwErrorText").textContent = msg || "Prompt rewrite failed.";
    rwShow("error");
  }

  // Nearest ratio the app renders, by log distance (2:3 -> 3:4, 3:2 -> 4:3)
  function nearestRatio(aspect) {
    let best = null, dist = Infinity;
    for (const { ratio } of S.cfg.ratios) {
      const [w, hh] = ratio.split(":").map(Number);
      const d = Math.abs(Math.log(aspect) - Math.log(w / hh));
      if (d < dist) { dist = d; best = ratio; }
    }
    return best;
  }

  async function suggestRatio(res) {
    const btn = $("#rwRatio");
    btn.hidden = true;
    RW.suggested = null;
    let aspect = null, label = "";
    const m = /^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$/.exec(res.wh_ratio || "");
    if (m) { aspect = +m[1] / +m[2]; label = res.wh_ratio; }
    const follow = /^<image(\d+)>$/.exec(res.ratio_follow || "");
    if (!aspect && follow) {
      const ref = RW.pending.refs[+follow[1] - 1];
      if (ref) {
        try { const bmp = await createImageBitmap(ref.blob); aspect = bmp.width / bmp.height; bmp.close(); label = `image ${follow[1]}`; } catch { /* unreadable */ }
      }
    }
    if (!aspect) return;
    const ratio = nearestRatio(aspect);
    if (!ratio || ratio === S.ratio) return;
    RW.suggested = ratio;
    btn.setAttribute("aria-pressed", "false");
    btn.querySelector("span").textContent = label === ratio ? `Use suggested ${ratio}` : `Use ${ratio} (suggested ${label})`;
    btn.hidden = false;
  }

  function showRewrite(res) {
    const box = $("#rwPrompt");
    box.value = res.prompt;
    rwCount();
    rwShow("done");
    suggestRatio(res);
    requestAnimationFrame(() => { box.scrollTop = 0; $("#rwConfirm").focus(); });
  }

  function rwCount() {
    const words = $("#rwPrompt").value.trim().split(/\s+/).filter(Boolean).length;
    $("#rwCount").textContent = `${words} words`;
  }

  function closeRewrite() {
    stopRewrite({ cancel: true });
    RW.min = false;
    $("#rwDock").hidden = true;
    $("#rewrite").hidden = true;
    RW.pending = null;
    $("#prompt").focus();
  }

  function confirmRewrite({ mine = false } = {}) {
    const pend = RW.pending;
    if (!pend) return;
    const text = mine ? pend.prompt : $("#rwPrompt").value.trim();
    if (!text) return $("#rwPrompt").focus();
    if (!mine && RW.suggested && $("#rwRatio").getAttribute("aria-pressed") === "true") { S.ratio = RW.suggested; syncControls(); }
    stopRewrite({ cancel: true });
    $("#rewrite").hidden = true;
    RW.pending = null;
    if (!mine) RW.done.add(text);
    // The references may have changed while the popup was open; render with the ones that were rewritten
    return generate({ seedOverride: pend.seedOverride, prompt: text, original: mine ? "" : pend.prompt, refs: pend.refs });
  }

  function wireRewrite() {
    setRewriteOn(store.get("rewrite", true) !== false);
    $("#rewriteToggle").addEventListener("click", () => {
      setRewriteOn(!S.rewriteOn);
      toast(S.rewriteOn ? "Prompts go through the Qwen rewriter before generating." : "Prompts are sent as you write them.");
    });
    $("#rwClose").addEventListener("click", closeRewrite);
    $("#rwMin").addEventListener("click", minimizeRewrite);
    $("#rwDock").addEventListener("click", expandRewrite);
    // Clicking outside minimizes instead of cancelling, so a stray click doesn't throw the rewrite away
    $("#rewrite").addEventListener("click", (e) => { if (e.target.id === "rewrite") minimizeRewrite(); });
    $("#rwMine").addEventListener("click", () => confirmRewrite({ mine: true }));
    $("#rwAgain").addEventListener("click", () => { RW.cache.delete(RW.pending.key); startRewrite(); });
    $("#rwConfirm").addEventListener("click", () => confirmRewrite());
    $("#rwPrompt").addEventListener("input", rwCount);
    $("#rwRatio").addEventListener("click", (e) => {
      const b = e.currentTarget;
      b.setAttribute("aria-pressed", String(b.getAttribute("aria-pressed") !== "true"));
    });
    $("#rewrite").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !$("#rwConfirm").disabled) { e.preventDefault(); confirmRewrite(); }
    });
  }

  let pollTimer = null;
  function activeJobs() { return [...S.jobs.values()].filter((j) => j.status === "queued" || j.status === "running"); }
  function startPolling() { if (!pollTimer) pollTimer = setInterval(poll, 600); updateQueue(); }
  async function poll() {
    const ids = activeJobs().map((j) => j.id);
    if (!ids.length) { clearInterval(pollTimer); pollTimer = null; updateQueue(); return; }
    let res;
    try { res = await api(`/api/jobs?ids=${ids.join(",")}`); } catch { return; }
    for (const job of res.items) {
      const prev = S.jobs.get(job.id);
      S.jobs.set(job.id, job);
      if (prev && prev.status !== job.status) onJobChange(job);
    }
    // Jobs the server forgot (restart) are treated as failed
    for (const id of ids) if (!res.items.find((j) => j.id === id)) onJobChange({ ...S.jobs.get(id), status: "error", error: "The server restarted before this finished." });
    for (const job of activeJobs()) updateJobTile(job);
    updateQueue();
    renderWfRuns();
  }
  function onJobChange(job) {
    S.jobs.set(job.id, job);
    const away = $("#app").dataset.view !== "create" || document.hidden;
    const batch = S.jobBatch.get(job.id);
    if (job.status === "done") {
      const recs = job.results && job.results.length ? job.results : [job.result];
      const rec = recs[0];
      S.jobs.delete(job.id);
      for (const r of [...recs].reverse()) {
        FEED.fresh.add(r.file);
        S.feed.items.unshift(r);
        S.feed.total++;
        if (S.lib.loaded) { S.lib.items.unshift(r); S.lib.total++; }
      }
      if (S.lib.loaded && !S.lib.q) renderLibrary(true);
      if (!isComfy(rec)) {
        if (!S.seedLocked) { S.seed = rec.seed; $("#seed").value = S.seed; }
        const spec = loraSpec(rec.lora); if (spec) spec.loaded = true;
        if (rec.lora && !S.covers[rec.lora]) S.covers[rec.lora] = rec.thumb;
      }
      const view = { label: "View", fn: () => openLightbox(S.feed.items, S.feed.items.indexOf(rec), "feed") };
      const wfAway = $("#app").dataset.view !== "create" && $("#app").dataset.view !== "workflows";
      if (batch) {
        // Batches report once, when their last image lands
        if (--batch.left === 0 && away) toast(`${batch.total} images ready.`, { type: "ok", action: view });
      } else if (isComfy(rec)) {
        if (wfAway || document.hidden || $("#app").dataset.view === "workflows") toast(`${rec.workflow_name || "Workflow"} finished.`, { type: "ok", action: view });
      } else if (away) toast(!rec.upscaler ? "A render finished." : rec.upscaler.startsWith("Enhanced") ? "Enhance finished." : "Upscale finished.", { type: "ok", action: view });
    } else if (job.status === "error" || job.status === "cancelled") {
      if (batch) { batch.left--; if (job.status === "cancelled") batch.total--; }
      if (job.status === "error") {
        S.failed.unshift({ id: job.id, error: job.error || "Generation failed.", aspect: jobAspect(job) });
        toast(job.error || "Generation failed.", { type: "error", timeout: 7000 });
      }
      S.jobs.delete(job.id);
    }
    renderFeed();
  }
  async function cancelJob(id) {
    try { await api(`/api/jobs/${id}/cancel`, { method: "POST" }); } catch (e) { toast(e.message, { type: "error" }); }
    const job = S.jobs.get(id);
    if (job && job.status === "queued") onJobChange({ ...job, status: "cancelled" });
    else if (job) { job.phase = "Stopping"; updateJobTile(job); }
    renderWfRuns();
  }
  function updateQueue() {
    const n = activeJobs().length;
    $("#queuePill").hidden = n === 0;
    $("#queueCount").textContent = n;
    $("#queuePill").title = `${n} render${n === 1 ? "" : "s"} in progress`;
    document.title = n ? `(${n}) Obscura` : "Obscura";
  }

  // ---------------------------------------------------------------- create feed (5-column grid)
  // Rendering jobs sit at the top, newest first, followed by finished images. Tiles are cached by key so
  // re-layouts move existing nodes instead of rebuilding them (no image reloads or flicker).
  const FEED = { cache: new Map(), cols: 0, fresh: new Set() };
  const jobAspect = (job) => job.aspect || (() => { const r = ratioInfo(job.ratio, job.resolution || "1MP"); return r.width / r.height; })();
  const recAspect = (rec) => (rec.width && rec.height ? rec.width / rec.height : 1);
  function feedColumns() {
    const w = $("#feed").clientWidth;
    return w >= 860 ? 5 : w >= 600 ? 3 : 2;
  }
  function scrollFeedTop() {
    const f = $("#feed");
    if (f.scrollTop > 0) f.scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" });
  }
  function jobTile(job) {
    const el = h("div", { class: "ftile ftile-job", "data-job": job.id },
      h("div", { class: "develop-grain" }), h("div", { class: "develop-sweep" }),
      h("div", { class: "fjob-meta" },
        h("div", { class: "fjob-count" }, h("span", { class: "n" }, "0"), h("span", { class: "dim" }, "/", h("span", { class: "t" }, String(job.steps)))),
        h("div", { class: "fjob-phase" }, "Queued"),
        h("div", { class: "develop-bar" }, h("span")),
        h("div", { class: "fjob-eta" })),
      h("button", { class: "icon-btn fjob-x", type: "button", "aria-label": "Cancel", title: "Cancel", onclick: () => cancelJob(job.id) }, icon("x")));
    return el;
  }
  function updateJobTile(job) {
    const el = FEED.cache.get("job:" + job.id);
    if (!el) return;
    const p = job.steps ? job.step / job.steps : 0;
    el.style.setProperty("--p", p);
    el.querySelector(".develop-bar").style.setProperty("--p", p);
    el.querySelector(".n").textContent = job.step;
    el.querySelector(".t").textContent = job.steps;
    let phase = job.phase, eta = "";
    if (job.status === "queued") {
      phase = job.ahead ? `Waiting, ${job.ahead} ahead` : "Up next";
    } else if (job.kind === "upscale" && job.step >= 1 && job.step < job.steps) {
      phase = `Upscaling, tile ${job.step} of ${job.steps}`;
      if (job.step >= 2 && job.first_step_at) eta = `About ${fmtSecs(((job.server_time - job.first_step_at) / (job.step - 1)) * (job.steps - job.step))} left`;
    } else if (job.step >= 2 && job.first_step_at) {
      const left = ((job.server_time - job.first_step_at) / (job.step - 1)) * (job.steps - job.step);
      eta = job.step < job.steps ? `About ${fmtSecs(left)} left` : "Almost there";
    } else if (job.started) {
      eta = `${fmtSecs(job.server_time - job.started)} elapsed`;
    }
    el.querySelector(".fjob-phase").textContent = phase;
    el.querySelector(".fjob-eta").textContent = eta;
    el.classList.toggle("queued", job.status === "queued");
  }
  function imageTile(rec) {
    const img = h("img", { src: rec.thumb, alt: rec.prompt || "Render", loading: "lazy", decoding: "async" });
    img.addEventListener("load", () => img.classList.add("loaded"));
    const act = (ic, tip, fn) => h("button", { class: "icon-btn", type: "button", "aria-label": tip, title: tip,
      onclick: (e) => { e.stopPropagation(); fn(); } }, icon(ic));
    const open = () => openLightbox(S.feed.items, S.feed.items.indexOf(rec), "feed");
    const trash = act("trash", "Delete", () => {
      // Two clicks: the first arms the button, the second deletes
      if (!trash.classList.contains("confirm")) {
        trash.classList.add("confirm");
        trash.title = "Click again to delete";
        setTimeout(() => { trash.classList.remove("confirm"); trash.title = "Delete"; }, 3000);
        return;
      }
      deleteFiles([rec.file]);
    });
    const video = isVideo(rec);
    const reuseBtn = isComfy(rec)
      ? act("arrow-counter-clockwise", "Reuse workflow settings", () => reuse(rec))
      : act("arrow-counter-clockwise", "Reuse settings", () => { reuse(rec); toast("Settings restored, seed fixed.", { type: "ok" }); });
    const el = h("div", {
      class: `ftile ftile-img${rec.mode === "Transparent PNG" ? " alpha" : ""}${SEL.files.has(rec.file) ? " selected" : ""}`,
      role: "button", tabindex: "0", "data-file": rec.file,
      "aria-label": rec.prompt ? `Open: ${rec.prompt.slice(0, 80)}` : video ? "Open video" : "Open render",
      onclick: (e) => tileClick(e, rec, S.feed.items, open),
      onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } },
    }, img,
    rec.upscaler ? h("span", { class: "ftile-badge" }, rec.upscaler) : null,
    video ? videoBadge(rec) : null,
    h("span", { class: "sel-check", "aria-hidden": "true" }, icon("check", false)),
    likeBtn(rec),
    h("div", { class: "ftile-over" },
      h("p", {}, rec.prompt || rec.workflow_name || "Untitled"),
      h("div", { class: "ftile-actions" }, reuseBtn, act("download-simple", "Download", () => downloadFiles([rec])), trash)));
    if (video) hoverPlay(el, rec);
    return el;
  }
  function failedTile(f) {
    return h("div", { class: "ftile ftile-failed" },
      icon("warning-circle", true),
      h("p", {}, f.error),
      h("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: () => { S.failed = S.failed.filter((x) => x !== f); FEED.cache.delete("fail:" + f.id); renderFeed(); } }, "Dismiss"));
  }
  function renderFeed() {
    if (!S.cfg) return;
    const entries = [];
    for (const f of S.failed) entries.push({ key: "fail:" + f.id, ar: f.aspect, make: () => failedTile(f) });
    for (const job of activeJobs().sort((a, b) => b.created - a.created)) entries.push({ key: "job:" + job.id, ar: jobAspect(job), make: () => jobTile(job), job });
    for (const rec of S.feed.items) entries.push({ key: "img:" + rec.file, ar: recAspect(rec), make: () => imageTile(rec), rec });

    const grid = $("#feedGrid");
    const empty = !entries.length && S.feed.loaded;
    $("#stageEmpty").hidden = !empty;
    grid.hidden = empty;
    const n = feedColumns();
    FEED.cols = n;
    const cols = Array.from({ length: n }, () => h("div", { class: "feed-col" }));
    const heights = new Array(n).fill(0);
    const keep = new Set();
    entries.forEach((e, i) => {
      let el = FEED.cache.get(e.key);
      if (!el) {
        el = e.make();
        el.style.setProperty("--i", String(Math.min(i, 12)));
        FEED.cache.set(e.key, el);
        if (e.rec && FEED.fresh.has(e.rec.file)) { el.classList.add("developing"); FEED.fresh.delete(e.rec.file); }
      }
      el.style.aspectRatio = String(e.ar);
      keep.add(e.key);
      // Shortest column first keeps the newest work along the top rows
      let k = 0;
      for (let c = 1; c < n; c++) if (heights[c] < heights[k] - 0.02) k = c;
      cols[k].append(el);
      heights[k] += 1 / e.ar + 0.06;
      if (e.job) updateJobTile(e.job);
    });
    for (const key of [...FEED.cache.keys()]) if (!keep.has(key)) FEED.cache.delete(key);
    grid.replaceChildren(...cols);
  }
  async function loadFeed(reset) {
    if (S.feed.loading || (!reset && S.feed.done)) return;
    S.feed.loading = true;
    try {
      const offset = reset ? 0 : S.feed.items.length;
      const res = await api(`/api/history?offset=${offset}&limit=60`);
      if (reset) S.feed.items = [];
      const seen = new Set(S.feed.items.map((r) => r.file));
      S.feed.items.push(...res.items.filter((r) => !seen.has(r.file)));
      S.feed.total = res.total;
      S.feed.done = S.feed.items.length >= res.total;
      S.feed.loaded = true;
      renderFeed();
    } catch (e) {
      toast(e.message, { type: "error" });
    } finally {
      S.feed.loading = false;
    }
  }

  // ---------------------------------------------------------------- record actions
  function reuse(rec, { withPrompt = true } = {}) {
    if (isComfy(rec)) return reuseWorkflow(rec);
    if (withPrompt) { $("#prompt").value = rec.prompt; autosize(); }
    // A prompt that came out of the rewriter goes straight to the renderer next time
    if (withPrompt && rec.original_prompt) RW.done.add(rec.prompt.trim());
    if (rec.mode && rec.mode !== S.mode && modeInfo(rec.mode)) setMode(rec.mode, { quiet: true });
    if (rec.ratio) S.ratio = rec.ratio;
    if (rec.resolution) S.res = rec.resolution;
    if (rec.steps) S.steps = rec.steps;
    const known = allStyles().some((s) => s.id === rec.lora);
    if (known) S.lora = rec.lora;
    if (rec.lora_strength != null) S.strength = rec.lora_strength;
    if (rec.seed != null) { S.seed = rec.seed; S.seedLocked = true; }
    syncControls();
    renderRefs();
  }
  async function editRecord(rec) {
    try {
      const blob = await (await fetch(rec.url)).blob();
      setMode("Edit Image (1 Ref)", { quiet: true });
      setRef(0, blob);
      $("#prompt").value = "";
      autosize();
      closeLightbox();
      go("create");
      toast("Image loaded for editing. Describe the change.", { type: "ok" });
      setTimeout(() => $("#prompt").focus(), 50);
    } catch { toast("Could not load that image.", { type: "error" }); }
  }

  // ---------------------------------------------------------------- upscaling
  const canUpscale = (rec) => rec.width && rec.height && rec.width * rec.height <= S.cfg.upscale_max_input;
  const canEnhance = (rec) => rec.width && rec.height && rec.width * rec.height <= S.cfg.enhance.max_input;
  // Every image opens at the default; the slider only changes the enhance you're about to start
  const ENHANCE_DEFAULT = 0.3;
  const enhStrength = () => ENHANCE_DEFAULT;
  // Mirrors the server's plan: one 4MP pass for normal renders, overlapping 2048px tiles for large upscales
  function enhanceEstimate(rec, strength) {
    const e = S.cfg.enhance;
    const axis = (L) => (L <= e.tile ? 1 : Math.ceil((L - e.overlap) / (e.tile - e.overlap)));
    const tiles = rec.width * rec.height <= e.single_max ? 1 : axis(rec.width) * axis(rec.height);
    const steps = Math.max(4, Math.round(S.steps * strength));
    const secs = tiles * (steps * 1.8 + 6) + (tiles === 1 && rec.width * rec.height < 3.5e6 ? 5 : 0);
    return { tiles, steps, secs };
  }
  function renderEnhanceHint() {
    const rec = LB.list[LB.i];
    if (!rec || !rec.width) return;
    const s = Number($("#enhStrength").value);
    $("#enhVal").textContent = s.toFixed(2);
    setRangeFill($("#enhStrength"));
    const est = enhanceEstimate(rec, s);
    const where = est.tiles === 1
      ? (rec.width * rec.height < 3.5e6 ? "Enlarges to 4 MP, then repaints fine detail" : "Repaints fine detail at 4 MP")
      : `Large image: ${est.tiles} tiles at full size`;
    $("#enhHint").textContent = `${where}. ${est.steps} steps per pass, about ${fmtSecs(est.secs)}.`;
  }
  async function enhance(rec, strength = enhStrength(), prompt = S.cfg.enhance.default_prompt) {
    try {
      const job = await api("/api/enhance", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ file: rec.file, strength, steps: S.steps, prompt }) });
      S.jobs.set(job.id, job);
      closeLightbox();
      go("create");
      renderFeed();
      scrollFeedTop();
      startPolling();
      toast(`Enhancing detail at ${strength.toFixed(2)}.`);
    } catch (e) { toast(e.message, { type: "error" }); }
  }
  async function upscale(rec, key) {
    const label = (S.cfg.upscalers.find((u) => u.id === key) || {}).label || key;
    try {
      const job = await api("/api/upscale", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ file: rec.file, upscaler: key }) });
      S.jobs.set(job.id, job);
      closeLightbox();
      go("create");
      renderFeed();
      scrollFeedTop();
      startPolling();
      toast(`Upscaling 4x with ${label}.`);
    } catch (e) { toast(e.message, { type: "error" }); }
  }

  // ---------------------------------------------------------------- library
  function ensureLibrary() {
    if (!S.lib.loaded) loadLibrary(true);
    else renderLibrary(true);
    setTimeout(() => $("#librarySearch").focus({ preventScroll: true }), 0);
  }
  async function loadLibrary(reset) {
    if (S.lib.loading) return;
    if (reset) { S.lib.items = []; S.lib.done = false; }
    if (S.lib.done) return;
    S.lib.loading = true;
    if (reset) showLibrarySkeleton();
    try {
      const q = encodeURIComponent(S.lib.q);
      const res = await api(`/api/history?offset=${S.lib.items.length}&limit=48&q=${q}`);
      const start = S.lib.items.length;
      S.lib.items.push(...res.items);
      S.lib.total = res.total;
      S.lib.done = S.lib.items.length >= res.total;
      S.lib.loaded = true;
      renderLibrary(reset, reset ? 0 : start);
    } catch (e) {
      toast(e.message, { type: "error" });
    } finally {
      S.lib.loading = false;
    }
  }
  function showLibrarySkeleton() {
    masonrySkeleton($("#masonry"));
    $("#libraryEmpty").hidden = true;
  }
  function tileFor(rec, i, source = "library") {
    const ar = rec.width && rec.height ? `${rec.width}/${rec.height}` : "1";
    const img = h("img", { src: rec.thumb, alt: rec.prompt, loading: "lazy", decoding: "async", style: `aspect-ratio:${ar}` });
    img.addEventListener("load", () => img.classList.add("loaded"));
    const label = isComfy(rec) ? rec.workflow_name || "Workflow" : rec.upscaler || styleName(rec.lora);
    const list = () => (source === "liked" ? S.liked.items : S.lib.items);
    const open = () => openLightbox(list(), list().indexOf(rec), source);
    const el = h("div", {
      class: `tile${SEL.files.has(rec.file) ? " selected" : ""}`, role: "button", tabindex: "0", style: `--i:${i % 12}`, "data-file": rec.file,
      "aria-label": rec.prompt ? `Open: ${rec.prompt.slice(0, 80)}` : "Open render",
      onclick: (e) => tileClick(e, rec, list(), open),
      onkeydown: (e) => { if (e.target === el && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); open(); } },
    }, img, isVideo(rec) ? videoBadge(rec) : null, h("span", { class: "sel-check", "aria-hidden": "true" }, icon("check", false)), likeBtn(rec), h("div", { class: "tile-over" },
      h("p", {}, rec.prompt || rec.workflow_name || "Untitled"),
      h("span", {}, `${label}  ${rec.width || ""}×${rec.height || ""}`)));
    if (isVideo(rec)) hoverPlay(el, rec);
    return el;
  }
  // Library and Liked grids: same column count as the Create feed, and each tile goes into the
  // shortest column in order, so newest-first reads left to right across the top rows
  const masonryColumns = (m) => { const w = m.clientWidth; return w >= 860 ? 5 : w >= 600 ? 3 : 2; };
  const recRatio = (rec) => (rec.width && rec.height ? rec.height / rec.width : 1);
  function layoutMasonry(m, recs, source, from = 0) {
    const n = masonryColumns(m);
    let st = m._layout;
    if (!st || from === 0 || st.n !== n || st.count !== from) {
      st = m._layout = { n, count: 0, heights: new Array(n).fill(0), cols: Array.from({ length: n }, () => h("div", { class: "masonry-col" })) };
      m.replaceChildren(...st.cols);
      from = 0;
    }
    recs.slice(from).forEach((rec, j) => {
      let k = 0;
      for (let c = 1; c < n; c++) if (st.heights[c] < st.heights[k] - 0.02) k = c;
      st.cols[k].append(tileFor(rec, from + j, source));
      st.heights[k] += recRatio(rec) + 0.05;
    });
    st.count = recs.length;
  }
  function masonrySkeleton(m) {
    m._layout = null;
    const n = masonryColumns(m);
    const ars = [1, 0.75, 1.33, 1, 0.56, 1.78, 1, 0.75, 1, 1.33];
    m.replaceChildren(...Array.from({ length: n }, (_, c) => h("div", { class: "masonry-col" },
      ...[0, 1].map((r) => h("div", { class: "tile skeleton", style: `aspect-ratio:${ars[(c + r * n) % ars.length]};cursor:default` })))));
  }
  function relayoutMasonry() {
    const view = $("#app").dataset.view;
    const m = view === "library" ? $("#masonry") : view === "liked" ? $("#likedMasonry") : null;
    if (!m || !m._layout || masonryColumns(m) === m._layout.n) return;
    if (view === "library") layoutMasonry(m, S.lib.items, "library");
    else layoutMasonry(m, S.liked.items, "liked");
  }
  function renderLibrary(reset, from = 0) {
    layoutMasonry($("#masonry"), S.lib.items, "library", reset ? 0 : from);
    const n = S.lib.total;
    $("#libraryCount").textContent = S.lib.q
      ? `${n} result${n === 1 ? "" : "s"} for “${S.lib.q}”`
      : n ? `${n} render${n === 1 ? "" : "s"} on this machine, newest first.` : "Everything you have rendered on this machine.";
    const empty = $("#libraryEmpty");
    empty.hidden = S.lib.items.length > 0;
    if (!S.lib.items.length && S.lib.q) {
      empty.querySelector("h2").textContent = "No matches";
      empty.querySelector("p").textContent = "Try a different word from the prompt or a style name.";
    } else {
      empty.querySelector("h2").textContent = "Nothing here yet";
      empty.querySelector("p").textContent = "Your renders land here automatically, newest first.";
    }
  }

  // ---------------------------------------------------------------- likes
  function likeBtn(rec) {
    const b = h("button", { class: "icon-btn like-btn", type: "button", "data-like": rec.file,
      onclick: (e) => { e.stopPropagation(); toggleLike(rec); },
      onkeydown: (e) => e.stopPropagation() }, icon("heart"));
    paintLike(b, !!rec.liked);
    return b;
  }
  function paintLike(b, on) {
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", String(on));
    b.setAttribute("aria-label", on ? "Unlike" : "Like");
    b.title = on ? "Unlike" : "Like";
    b.querySelector("i").className = `${on ? "ph-fill" : "ph"} ph-heart`;
  }
  function markLiked(file, on) {
    for (const r of [...S.feed.items, ...S.lib.items, ...S.liked.items, ...LB.list]) if (r.file === file) r.liked = on;
    $$(".like-btn").forEach((b) => { if (b.dataset.like === file) paintLike(b, on); });
    const cur = LB.list[LB.i];
    if (!$("#lightbox").hidden && cur && cur.file === file) paintLike($("#lbLike"), on);
  }
  async function toggleLike(rec) {
    const on = !rec.liked;
    markLiked(rec.file, on);
    try {
      await api("/api/likes", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ file: rec.file, liked: on }) });
    } catch (e) {
      markLiked(rec.file, !on);
      toast(e.message, { type: "error" });
      return;
    }
    if (on) S.liked.loaded = false;
    else S.liked.items = S.liked.items.filter((r) => r.file !== rec.file);
    if ($("#app").dataset.view === "liked") on ? loadLiked() : renderLiked();
  }
  function ensureLiked() {
    if (!S.liked.loaded) loadLiked();
    else renderLiked();
  }
  async function loadLiked() {
    if (S.liked.loading) return;
    S.liked.loading = true;
    if (!S.liked.items.length) {
      masonrySkeleton($("#likedMasonry"));
      $("#likedEmpty").hidden = true;
    }
    try {
      const items = [];
      for (;;) {
        const res = await api(`/api/history?liked=1&offset=${items.length}&limit=200`);
        items.push(...res.items);
        if (!res.items.length || items.length >= res.total) break;
      }
      S.liked.items = items;
      S.liked.loaded = true;
      renderLiked();
    } catch (e) {
      toast(e.message, { type: "error" });
    } finally {
      S.liked.loading = false;
    }
  }
  function renderLiked() {
    const n = S.liked.items.length;
    layoutMasonry($("#likedMasonry"), S.liked.items, "liked");
    $("#likedCount").textContent = n ? `${n} liked render${n === 1 ? "" : "s"}, newest first.` : "The renders you liked.";
    $("#likedEmpty").hidden = n > 0;
  }

  // ---------------------------------------------------------------- lightbox
  const LB = { list: [], i: 0, source: null };
  let lastFocus = null;
  function openLightbox(list, i, source = null) {
    if (i < 0 || !list.length) return;
    LB.list = list; LB.i = i; LB.source = source;
    lastFocus = document.activeElement;
    $("#lightbox").hidden = false;
    renderLightbox();
    if (!Z.hinted) {
      Z.hinted = true;
      const badge = $("#lbZoom");
      badge.textContent = "Scroll to zoom, drag to move";
      badge.classList.add("show");
      setTimeout(() => { if (Z.s <= 1.001) badge.classList.remove("show"); }, 2600);
    }
    $("#lbClose").focus();
  }
  function closeLightbox() {
    if ($("#lightbox").hidden) return;
    $("#lightbox").hidden = true;
    const vid = $("#lbVideo");
    vid.pause(); vid.removeAttribute("src"); vid.load();
    resetDelete();
    lastFocus && lastFocus.focus && lastFocus.focus({ preventScroll: true });
  }
  function renderLightbox() {
    const rec = LB.list[LB.i];
    if (!rec) return closeLightbox();
    const img = $("#lbImg");
    const vid = $("#lbVideo");
    const video = isVideo(rec);
    Z.full = false;
    img.hidden = video;
    vid.hidden = !video;
    $("#lbStage").classList.toggle("is-video", video);
    if (video) {
      img.removeAttribute("src");
      vid.poster = rec.thumb;
      if (vid.getAttribute("src") !== rec.url) vid.src = rec.url;
      vid.muted = store.get("videoMuted", true);
      vid.play().catch(() => {});
    } else {
      vid.pause();
      vid.removeAttribute("src");
      vid.load();
      img.src = rec.thumb;
      const full = new Image();
      const target = rec.preview || rec.url;
      full.onload = () => { if (LB.list[LB.i] === rec && !Z.full) img.src = target; };
      full.src = target;
    }
    img.alt = rec.prompt;
    const layer = $("#lbLayer");
    layer.style.animation = "none"; void layer.offsetWidth; layer.style.animation = "";
    lbFit();
    loadCompare(rec);
    $("#lbStage").classList.toggle("alpha", rec.mode === "Transparent PNG");
    $("#lbDate").textContent = fmtDate(rec.created);
    const promptEl = $("#lbPrompt");
    promptEl.textContent = rec.prompt || "No prompt recorded.";
    // Full prompt by default; the toggle appears only when the prompt is long enough to collapse
    promptEl.classList.add("clamped");
    const collapsible = promptEl.scrollHeight > promptEl.clientHeight + 2;
    promptEl.classList.remove("clamped");
    const more = $("#lbMore");
    more.hidden = !collapsible;
    more.classList.add("open");
    more.querySelector("span").textContent = "Collapse prompt";
    const mode = modeInfo(rec.mode);
    const spec = (k, v, wide) => h("div", { class: wide ? "wide" : "" }, h("dt", {}, k), h("dd", { title: String(v) }, String(v)));
    if (isComfy(rec)) $("#lbSpecs").replaceChildren(
      spec("Workflow", rec.workflow_name || rec.workflow || "?", true),
      spec("Size", `${rec.width || "?"}×${rec.height || "?"}`),
      ...(video ? [spec("Length", rec.frames ? `${rec.frames} frames` : "?"), spec("FPS", rec.fps ?? "?"), spec("Duration", fmtDur(rec.duration) || "?")] : []),
      ...(rec.seed != null ? [spec("Seed", rec.seed)] : []),
      ...(rec.steps ? [spec("Steps", rec.steps)] : []),
      ...(rec.elapsed ? [spec("Render time", fmtSecs(rec.elapsed))] : []),
      ...(rec.values || []).filter(([k]) => !["Width", "Height", "Frames", "FPS", "Steps"].includes(k)).map(([k, v]) => spec(k, v, String(v).length > 14)),
    );
    else $("#lbSpecs").replaceChildren(
      spec("Style", styleName(rec.lora), true),
      spec("Mode", mode ? mode.label : rec.mode),
      spec("Size", `${rec.width || "?"}×${rec.height || "?"}`),
      ...(rec.checkpoint && modelFor(rec.checkpoint) ? [spec("Model", modelFor(rec.checkpoint).label)] : []),
      spec("Seed", rec.seed ?? "?"),
      spec("Steps", rec.steps ?? "?"),
      ...(rec.lora_strength != null && rec.lora !== S.cfg.none_lora ? [spec("Strength", Number(rec.lora_strength).toFixed(2))] : []),
      ...(rec.elapsed ? [spec(!rec.upscaler ? "Render time" : rec.upscaler.startsWith("Enhanced") ? "Enhance time" : "Upscale time", fmtSecs(rec.elapsed))] : []),
      ...(rec.upscaler ? [spec(rec.upscaler.startsWith("Enhanced") ? "Enhanced" : "Upscaler", rec.upscaler.startsWith("Enhanced") ? `strength ${rec.upscaler.split(" ")[1]}` : rec.upscaler, true)] : []),
      ...(rec.enhance_prompt ? [spec("Enhance prompt", rec.enhance_prompt, true)] : []),
      ...(rec.original_prompt ? [spec("Your prompt (before rewrite)", rec.original_prompt, true)] : []),
    );
    // A lone item on the last row spans both columns instead of leaving an empty cell
    const cells = $$("#lbSpecs > div:not(.wide)");
    if (cells.length % 2) cells[cells.length - 1].classList.add("wide");
    $("#lbUpscale").hidden = video || !canUpscale(rec);
    $("#lbEnhanceBox").hidden = video || !canEnhance(rec);
    $("#lbEdit").hidden = video;
    $("#lbAnimate").hidden = video;
    $("#lbReuse").lastChild.textContent = isComfy(rec) ? "Reuse workflow settings" : "Reuse settings";
    $("#enhStrength").value = enhStrength();
    $("#enhPrompt").value = S.cfg.enhance.default_prompt;
    $("#enhAddOriginal").hidden = !rec.prompt;
    renderEnhanceHint();
    $("#lbDownload").href = rec.url + "?download=1";
    paintLike($("#lbLike"), !!rec.liked);
    $("#lbPrev").disabled = LB.i <= 0;
    $("#lbNext").disabled = LB.i >= LB.list.length - 1 && !(LB.source === "library" && !S.lib.done) && !(LB.source === "feed" && !S.feed.done);
    resetDelete();
  }
  // Lightbox zoom: scroll zooms around the cursor, drag pans, click toggles fit / actual pixels
  const Z = { s: 1, tx: 0, ty: 0, max: 8, fitW: 0, fitH: 0, full: false, drag: null, hinted: false };
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  function lbFit() {
    const rec = LB.list[LB.i];
    const stage = $("#lbStage"), layer = $("#lbLayer");
    if (!rec || $("#lightbox").hidden) return;
    const cs = getComputedStyle(stage);
    const W = stage.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
    const H = stage.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom);
    const ar = rec.width && rec.height ? rec.width / rec.height : 1;
    let w = W, hh = w / ar;
    if (hh > H) { hh = H; w = hh * ar; }
    Z.fitW = w; Z.fitH = hh;
    layer.style.width = w + "px";
    layer.style.height = hh + "px";
    C.x = null; // recentre the divider on the new fit
    // Up to 2x actual pixels, and never less than 4x the fitted size
    Z.max = Math.max(4, ((rec.width || w) / w) * 2);
    zoomReset(false);
  }
  function clampPan() {
    const stage = $("#lbStage"), layer = $("#lbLayer");
    const sw = stage.clientWidth, sh = stage.clientHeight;
    const lx = layer.offsetLeft, ly = layer.offsetTop;
    const w = Z.fitW * Z.s, hh = Z.fitH * Z.s;
    Z.tx = w <= sw ? (sw - w) / 2 - lx : clamp(Z.tx, sw - w - lx, -lx);
    Z.ty = hh <= sh ? (sh - hh) / 2 - ly : clamp(Z.ty, sh - hh - ly, -ly);
  }
  function applyZoom(animate) {
    const img = $("#lbImg"), layer = $("#lbLayer"), rec = LB.list[LB.i];
    layer.style.transition = animate && !reduceMotion ? "transform 0.35s var(--ease)" : "none";
    layer.style.transform = `translate(${Z.tx}px, ${Z.ty}px) scale(${Z.s})`;
    updateDivider(animate);
    const zoomed = Z.s > 1.001;
    $("#lbStage").classList.toggle("zoomed", zoomed);
    const badge = $("#lbZoom");
    if (zoomed) {
      const pct = rec && rec.width ? Math.round(((Z.s * Z.fitW) / rec.width) * 100) : Math.round(Z.s * 100);
      badge.textContent = `${pct}%`;
      badge.classList.add("show");
    } else badge.classList.remove("show");
    // Swap in the full-resolution file once the zoom outgrows what is loaded
    if (rec && zoomed && !Z.full && Z.s * Z.fitW > (img.naturalWidth || 0) * 1.05 && !img.src.endsWith(rec.url)) {
      Z.full = true;
      badge.classList.add("loading");
      const full = new Image();
      full.onload = () => { if (LB.list[LB.i] === rec) img.src = rec.url; badge.classList.remove("loading"); };
      full.onerror = () => badge.classList.remove("loading");
      full.src = rec.url;
    }
  }
  function zoomAt(cx, cy, s, animate) {
    const stage = $("#lbStage"), layer = $("#lbLayer");
    const r = stage.getBoundingClientRect();
    const px = cx - r.left, py = cy - r.top;
    const lx = layer.offsetLeft, ly = layer.offsetTop;
    const u = (px - lx - Z.tx) / Z.s, v = (py - ly - Z.ty) / Z.s;
    Z.s = clamp(s, 1, Z.max);
    Z.tx = px - lx - Z.s * u;
    Z.ty = py - ly - Z.s * v;
    clampPan();
    applyZoom(animate);
  }
  function zoomReset(animate) { Z.s = 1; clampPan(); applyZoom(animate); }
  function zoomCenter(factor) {
    const r = $("#lbStage").getBoundingClientRect();
    zoomAt(r.left + r.width / 2, r.top + r.height / 2, Z.s * factor, true);
  }
  function actualPixelsScale() {
    const rec = LB.list[LB.i];
    const s = rec && rec.width ? rec.width / Z.fitW : 2;
    return s < 1.5 ? 2 : s;
  }
  function wireZoom() {
    const stage = $("#lbStage"), img = $("#lbImg");
    const onVideo = () => isVideo(LB.list[LB.i]);
    stage.addEventListener("wheel", (e) => {
      if (onVideo()) return;
      e.preventDefault();
      const dy = e.deltaY * (e.deltaMode === 1 ? 16 : 1);
      // Trackpad pinch arrives as ctrl+wheel with small deltas
      zoomAt(e.clientX, e.clientY, Z.s * Math.exp(-dy * (e.ctrlKey ? 0.01 : 0.0018)), false);
    }, { passive: false });
    stage.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || e.target.closest(".lb-nav")) return;
      if (onVideo()) { if (e.target === stage) closeLightbox(); return; }
      if (e.target.closest("#lbDivider, .lb-side")) return;
      Z.drag = { x: e.clientX, y: e.clientY, tx: Z.tx, ty: Z.ty, moved: false, onImg: !!e.target.closest("#lbLayer"), id: e.pointerId };
      stage.setPointerCapture(e.pointerId);
    });
    stage.addEventListener("pointermove", (e) => {
      const d = Z.drag;
      if (!d || d.id !== e.pointerId) return;
      const dx = e.clientX - d.x, dy = e.clientY - d.y;
      if (!d.moved && Math.hypot(dx, dy) > 4) { d.moved = true; if (Z.s > 1.001) stage.classList.add("dragging"); }
      if (d.moved && Z.s > 1.001) { Z.tx = d.tx + dx; Z.ty = d.ty + dy; clampPan(); applyZoom(false); }
    });
    const end = (e) => {
      const d = Z.drag;
      if (!d || d.id !== e.pointerId) return;
      Z.drag = null;
      stage.classList.remove("dragging");
      if (d.moved || e.type === "pointercancel") return;
      if (d.onImg) {
        if (Z.s > 1.001) zoomReset(true);
        else zoomAt(e.clientX, e.clientY, actualPixelsScale(), true);
      } else if (Z.s <= 1.001) closeLightbox();
    };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
    $("#lbLayer").addEventListener("dragstart", (e) => e.preventDefault());
    wireCompare();
    window.addEventListener("resize", () => lbFit());
  }

  // Before / after slider: the "before" image sits under the upscale, which is clipped at the divider
  const C = { on: true, avail: false, x: null, before: null, options: [], token: 0 };
  function updateDivider(animate) {
    const div = $("#lbDivider"), after = $("#lbImg"), before = $("#lbBefore");
    const show = C.avail && C.on && C.before;
    div.hidden = !show; before.hidden = !show;
    $("#lbBeforePick").hidden = !show; $("#lbAfterLabel").hidden = !show;
    if (!show) { after.style.clipPath = ""; return; }
    const stage = $("#lbStage"), layer = $("#lbLayer");
    const left = layer.offsetLeft + Z.tx, width = Z.fitW * Z.s;
    const lo = Math.max(0, left), hi = Math.min(stage.clientWidth, left + width);
    if (C.x == null) C.x = (lo + hi) / 2;
    C.x = clamp(C.x, lo, hi);
    div.style.transition = animate && !reduceMotion ? "left 0.35s var(--ease)" : "none";
    div.style.left = C.x + "px";
    const cut = (C.x - left) / Z.s; // divider position in the layer's own pixels
    after.style.transition = div.style.transition.replace("left", "clip-path");
    after.style.clipPath = `inset(0 0 0 ${Math.max(0, cut)}px)`;
    $("#lbHandle").setAttribute("aria-valuenow", String(Math.round(((C.x - left) / width) * 100)));
  }
  async function loadCompare(rec) {
    const token = ++C.token;
    C.avail = false; C.before = null; C.options = [];
    $("#lbCompareToggle").hidden = true;
    updateDivider(false);
    if (!rec.upscaled_from) return;
    let rel;
    try { rel = await api(`/api/related/${encodeURIComponent(rec.file)}`); } catch { return; }
    if (token !== C.token) return;
    const opts = [];
    if (rel.source) opts.push({ label: "Original", rec: rel.source });
    for (const u of rel.upscales) if (u.file !== rec.file) opts.push({ label: u.upscaler || "Upscale", rec: u });
    if (!opts.length) return;
    C.options = opts;
    C.avail = true;
    $("#lbCompareToggle").hidden = false;
    $("#lbAfterLabel").textContent = rec.upscaler || "Upscale";
    setBefore(opts[0]);
  }
  function setBefore(opt) {
    C.before = opt;
    const before = $("#lbBefore");
    before.src = opt.rec.preview || opt.rec.url;
    before.alt = `${opt.label} for comparison`;
    $("#lbBeforePick").replaceChildren(h("span", { class: "tag-lbl" }, "vs"), ...C.options.map((o) => h("button", {
      type: "button", role: "radio", "aria-checked": String(o === opt),
      onclick: () => setBefore(o),
    }, o.label)));
    updateDivider(false);
  }
  function wireCompare() {
    const div = $("#lbDivider"), stage = $("#lbStage");
    let dragId = null;
    div.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      dragId = e.pointerId;
      div.setPointerCapture(e.pointerId);
      div.classList.add("active");
    });
    div.addEventListener("pointermove", (e) => {
      if (e.pointerId !== dragId) return;
      C.x = e.clientX - stage.getBoundingClientRect().left;
      updateDivider(false);
    });
    const end = (e) => { if (e.pointerId === dragId) { dragId = null; div.classList.remove("active"); } };
    div.addEventListener("pointerup", end);
    div.addEventListener("pointercancel", end);
    $("#lbHandle").addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault(); e.stopPropagation();
      C.x = (C.x ?? 0) + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 80 : 20);
      updateDivider(false);
    });
    $("#lbCompareToggle").addEventListener("click", () => {
      C.on = !C.on;
      const t = $("#lbCompareToggle");
      t.setAttribute("aria-pressed", String(C.on));
      store.set("compare", C.on);
      updateDivider(false);
    });
    C.on = store.get("compare", true);
    $("#lbCompareToggle").setAttribute("aria-pressed", String(C.on));
  }

  async function lbStep(d) {
    const n = LB.i + d;
    if (n < 0) return;
    if (n >= LB.list.length) {
      if (LB.source === "library" && !S.lib.done) { await loadLibrary(false); LB.list = S.lib.items; }
      if (LB.source === "feed" && !S.feed.done) { await loadFeed(false); LB.list = S.feed.items; }
      if (n >= LB.list.length) return;
    }
    LB.i = n;
    renderLightbox();
  }
  let delTimer = null;
  function resetDelete() {
    clearTimeout(delTimer);
    const b = $("#lbDelete");
    b.classList.remove("confirm");
    b.querySelector("span").textContent = "Delete";
  }
  async function lbDelete() {
    const b = $("#lbDelete");
    if (!b.classList.contains("confirm")) {
      b.classList.add("confirm");
      b.querySelector("span").textContent = "Click again to delete";
      delTimer = setTimeout(resetDelete, 3500);
      return;
    }
    const rec = LB.list[LB.i];
    if (!(await deleteFiles([rec.file]))) return;
    if (LB.list !== S.lib.items && LB.list !== S.feed.items && LB.list !== S.liked.items) {
      const k = LB.list.findIndex((r) => r.file === rec.file);
      if (k >= 0) LB.list.splice(k, 1);
    }
    if (!LB.list.length) return closeLightbox();
    LB.i = Math.min(LB.i, LB.list.length - 1);
    renderLightbox();
  }

  // ---------------------------------------------------------------- multi-select (Ctrl/Cmd+click, Shift+click range)
  const SEL = { files: new Set(), anchor: null };
  function tileClick(e, rec, list, open) {
    const idx = list.indexOf(rec);
    if (e.shiftKey && SEL.anchor && SEL.anchor.list === list) {
      const [a, b] = [SEL.anchor.idx, idx].sort((x, y) => x - y);
      for (let i = a; i <= b; i++) SEL.files.add(list[i].file);
    } else if (e.ctrlKey || e.metaKey || SEL.files.size) {
      // Once something is selected, plain clicks keep adding or removing
      if (SEL.files.has(rec.file)) SEL.files.delete(rec.file); else SEL.files.add(rec.file);
      SEL.anchor = { list, idx };
    } else return open();
    e.preventDefault();
    syncSelection();
  }
  function clearSelection() { SEL.files.clear(); SEL.anchor = null; syncSelection(); }
  function syncSelection() {
    $$("[data-file]").forEach((el) => el.classList.toggle("selected", SEL.files.has(el.dataset.file)));
    const n = SEL.files.size;
    const bar = $("#selBar");
    bar.hidden = n === 0;
    $("#selCount").textContent = `${n} selected`;
    resetSelDelete();
    document.body.classList.toggle("selecting", n > 0);
  }
  const findRec = (file) => [S.feed.items, S.lib.items, S.liked.items].map((l) => l.find((r) => r.file === file)).find(Boolean);
  function downloadFiles(recs) {
    // Plain PNG downloads, one after another (browsers may ask once to allow multiple downloads)
    recs.forEach((rec, i) => setTimeout(() => {
      const a = h("a", { href: rec.url + "?download=1", download: "" });
      document.body.append(a); a.click(); a.remove();
    }, i * 350));
    if (recs.length > 1) toast(`Downloading ${recs.length} images.`, { type: "ok" });
  }
  async function deleteFiles(files) {
    let deleted;
    try {
      deleted = files.length === 1
        ? (await api(`/api/history/${encodeURIComponent(files[0])}`, { method: "DELETE" }), files)
        : (await api("/api/history/delete", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ files }) })).deleted;
    } catch (e) { toast(e.message, { type: "error" }); return false; }
    const gone = new Set(deleted);
    const before = { feed: S.feed.items.length, lib: S.lib.items.length };
    S.feed.items = S.feed.items.filter((r) => !gone.has(r.file));
    S.lib.items = S.lib.items.filter((r) => !gone.has(r.file));
    S.liked.items = S.liked.items.filter((r) => !gone.has(r.file));
    S.feed.total = Math.max(0, S.feed.total - (before.feed - S.feed.items.length));
    S.lib.total = Math.max(0, S.lib.total - (before.lib - S.lib.items.length));
    if (LB.source === "feed") LB.list = S.feed.items;
    if (LB.source === "library") LB.list = S.lib.items;
    if (LB.source === "liked") LB.list = S.liked.items;
    for (const k of Object.keys(S.covers)) if ([...gone].some((f) => S.covers[k].endsWith("/" + f))) delete S.covers[k];
    for (const f of gone) SEL.files.delete(f);
    renderFeed();
    if (S.lib.loaded) renderLibrary(true);
    if (S.liked.loaded) renderLiked();
    syncSelection();
    toast(gone.size === 1 ? "Deleted." : `Deleted ${gone.size} images.`, { type: "ok" });
    return true;
  }
  let selDelTimer = null;
  function resetSelDelete() {
    clearTimeout(selDelTimer);
    const b = $("#selDelete");
    b.classList.remove("confirm");
    b.querySelector("span").textContent = "Delete";
  }
  function wireSelection() {
    $("#selClear").addEventListener("click", clearSelection);
    $("#selDownload").addEventListener("click", () => downloadFiles([...SEL.files].map(findRec).filter(Boolean)));
    $("#selDelete").addEventListener("click", () => {
      const b = $("#selDelete");
      if (!b.classList.contains("confirm")) {
        b.classList.add("confirm");
        b.querySelector("span").textContent = `Delete ${SEL.files.size}? Click again`;
        selDelTimer = setTimeout(resetSelDelete, 3500);
        return;
      }
      deleteFiles([...SEL.files]);
    });
  }

  // ---------------------------------------------------------------- workflows (ComfyUI)
  // Presets and imported ComfyUI graphs, each with a generated form. Runs share the render queue and land in the Library.
  const W = { items: [], loaded: false, sel: null, comfy: null, statusTimer: null, vals: {}, rand: {}, media: {}, showHidden: false, edit: null };
  const MEDIA = new Set(["image", "video"]);
  const wfBy = (slug) => W.items.find((w) => w.slug === slug);
  const wfSel = () => wfBy(W.sel);
  function wfVals(wf) {
    if (!W.vals[wf.slug]) {
      const saved = store.get("wf:" + wf.slug, {});
      const vals = {}, rand = {};
      for (const f of wf.fields) {
        if (MEDIA.has(f.type)) continue;
        vals[f.key] = saved.vals && f.key in saved.vals ? saved.vals[f.key] : f.default;
        if (f.type === "seed") rand[f.key] = saved.rand && f.key in saved.rand ? saved.rand[f.key] : true;
      }
      W.vals[wf.slug] = vals; W.rand[wf.slug] = rand;
    }
    return W.vals[wf.slug];
  }
  const wfRand = (wf) => (wfVals(wf), W.rand[wf.slug]);
  const wfMedia = (wf) => (W.media[wf.slug] ||= {});
  const saveWf = (wf) => store.set("wf:" + wf.slug, { vals: wfVals(wf), rand: wfRand(wf) });
  // Through a Cloudflare tunnel the direct port doesn't exist on this hostname, so ComfyUI gets its own tunnel
  const viaTunnel = () => location.hostname.endsWith(".trycloudflare.com");
  async function loadEditorUrl() {
    let e;
    try { e = await api(`/api/comfy/editor?tunnel=${viaTunnel() ? 1 : 0}`); } catch { return ""; }
    const base = e.url || (viaTunnel() ? e.tunnel : e.port ? `${location.protocol}//${location.hostname}:${e.port}` : "");
    if (!base) return "";
    const u = new URL(base.replace(/\/?$/, "/"));
    if (e.token) u.searchParams.set("token", e.token);
    return u.toString();
  }

  async function loadWorkflows() {
    const res = await api("/api/workflows");
    W.items = res.items;
    W.comfy = res.comfy;
    W.loaded = true;
    if (!wfSel()) W.sel = (wfBy(store.get("wfSel")) || W.items[0] || {}).slug || null;
  }
  async function ensureWorkflows() {
    if (!W.loaded) {
      try { await loadWorkflows(); } catch (e) { toast(e.message, { type: "error" }); }
    }
    renderWorkflows();
    startWfStatus();
  }
  function startWfStatus() { if (!W.statusTimer) W.statusTimer = setInterval(refreshWfStatus, 4000); }
  function stopWfStatus() { clearInterval(W.statusTimer); W.statusTimer = null; }
  async function refreshWfStatus() {
    try { W.comfy = await api("/api/comfy/status"); } catch { return; }
    renderWfStatus();
  }
  async function renderEditorLink() {
    const ed = $("#wfEditor");
    if (!W.editor) W.editor = await loadEditorUrl();
    ed.hidden = !W.editor;
    if (W.editor) ed.href = W.editor;
  }
  // ComfyUI doesn't autostart (it would hold VRAM next to Qwen), so the editor button starts it first.
  // The tab is opened right away, inside the click, so popup blockers allow it.
  async function openEditor(e) {
    if (W.comfy && W.comfy.up) return;
    e.preventDefault();
    const tab = window.open("", "_blank");
    if (tab) { tab.opener = null; tab.document.title = "Starting ComfyUI…"; tab.document.body.textContent = "Starting ComfyUI…"; }
    toast("Starting ComfyUI…");
    try {
      W.comfy = await api("/api/comfy/start", { method: "POST" });
    } catch (err) {
      if (tab) tab.close();
      toast(err.message, { type: "error" });
      return;
    }
    renderWfStatus();
    if (tab) tab.location.href = W.editor; else window.open(W.editor, "_blank", "noopener");
  }
  function renderWorkflows() {
    renderEditorLink();
    renderWfStatus();
    renderWfList();
    renderWfPanel();
  }

  function renderWfStatus() {
    const c = W.comfy || {};
    const gb = (b) => (b / 1e9).toFixed(0);
    const box = $("#wfStatus");
    const qwen = c.qwen_moving ? "Moving Qwen…" : c.qwen_on_gpu ? "Qwen is on the GPU" : "Qwen is parked in RAM";
    const parkBtn = h("button", {
      class: "btn btn-ghost btn-sm", type: "button", disabled: c.qwen_moving || null,
      title: c.qwen_on_gpu ? "Move Qwen to RAM so workflows started in the ComfyUI editor get the whole GPU" : "Load Qwen back now instead of on the next render",
      onclick: async () => {
        try { W.comfy = await api("/api/comfy/park", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: c.qwen_on_gpu ? "park" : "unpark" }) }); }
        catch (e) { toast(e.message, { type: "error" }); }
        renderWfStatus();
        setTimeout(refreshWfStatus, 1500);
      },
    }, icon(c.qwen_on_gpu ? "cpu" : "arrow-counter-clockwise"), c.qwen_on_gpu ? "Free GPU for the editor" : "Load Qwen back");
    const keep = h("button", {
      class: "toggle", type: "button", "aria-pressed": String(!!c.keep_loaded),
      title: "Faster back-to-back videos. Qwen stays parked until you render an image.",
      onclick: async () => {
        try { W.comfy = await api("/api/comfy/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ keep_loaded: !c.keep_loaded }) }); }
        catch (e) { toast(e.message, { type: "error" }); }
        renderWfStatus();
      },
    }, icon("stack"), h("span", {}, "Keep video models loaded"));
    box.replaceChildren(
      h("span", { class: `wf-dot${c.up ? " up" : ""}` }),
      h("span", { class: "wf-stat" }, c.up ? `ComfyUI ${c.version || ""}` : "ComfyUI is off. It starts when you open the editor or run a workflow"),
      c.up && c.vram_total ? h("span", { class: "wf-stat dim" }, `${gb(c.vram_free)} of ${gb(c.vram_total)} GB VRAM free`) : null,
      c.up && (c.running || c.pending) ? h("span", { class: "wf-stat dim" }, `${c.running + c.pending} in ComfyUI queue`) : null,
      h("span", { class: "wf-stat dim" }, qwen),
      h("span", { class: "wf-status-actions" }, keep, parkBtn),
    );
  }

  // Motion transfer (takes a driving video) / image-to-video / still-image workflows
  const wfIcon = (wf) => wf.fields.some((f) => f.type === "video") ? "person-simple-run" : wf.output === "video" ? "film-slate" : "image";
  function renderWfList() {
    const list = $("#wfList");
    if (!W.items.length) {
      list.replaceChildren(h("p", { class: "hint" }, W.loaded ? "No workflows yet. Import one." : "Loading…"));
      return;
    }
    list.replaceChildren(...W.items.map((wf, i) => h("button", {
      class: "wf-item", type: "button", style: `--i:${i}`, "aria-current": String(wf.slug === W.sel),
      onclick: () => { W.sel = wf.slug; store.set("wfSel", wf.slug); W.showHidden = false; renderWfList(); renderWfPanel(); },
    },
    h("span", { class: "wf-item-icon" }, icon(wfIcon(wf))),
    h("span", { class: "wf-item-t" }, h("strong", {}, wf.name), h("small", {}, wf.description || (wf.preset ? "Preset" : "Imported workflow"))),
    wf.preset ? null : h("span", { class: "badge" }, "Yours"))));
  }

  function wfField(wf, f) {
    const vals = wfVals(wf);
    const set = (v) => { vals[f.key] = v; saveWf(wf); };
    const help = f.help ? h("p", { class: "hint" }, f.help) : null;
    const id = `wf_${wf.slug}_${f.key}`;
    if (f.type === "text") {
      const ta = h("textarea", { class: "input wf-text", id, rows: f.key === "prompt" ? 4 : 3, spellcheck: "true", maxlength: "8000" });
      ta.value = vals[f.key] ?? "";
      ta.addEventListener("input", () => set(ta.value));
      return h("div", { class: "field wf-wide" }, h("label", { class: "field-label", for: id }, h("span", {}, f.label)), ta, help);
    }
    if (MEDIA.has(f.type)) return wfMediaField(wf, f, help);
    if (f.type === "bool") {
      const t = h("button", { class: "toggle", type: "button", "aria-pressed": String(!!vals[f.key]),
        onclick: () => { set(!vals[f.key]); t.setAttribute("aria-pressed", String(!!vals[f.key])); } }, h("span", {}, vals[f.key] ? "On" : "Off"));
      t.addEventListener("click", () => { t.lastChild.textContent = vals[f.key] ? "On" : "Off"; });
      return h("div", { class: "field" }, h("div", { class: "field-label" }, h("span", {}, f.label), t), help);
    }
    const input = h("input", { class: "input mono", id, type: "number", inputmode: f.type === "float" ? "decimal" : "numeric",
      min: f.min ?? null, max: f.max ?? null, step: f.step ?? (f.type === "float" ? "any" : 1) });
    input.value = vals[f.key] ?? "";
    input.addEventListener("change", () => {
      const n = f.type === "float" ? parseFloat(input.value) : parseInt(input.value, 10);
      if (Number.isFinite(n)) set(n); else input.value = vals[f.key] ?? "";
      if (f.type === "seed") { wfRand(wf)[f.key] = false; saveWf(wf); renderWfPanel(); }
    });
    if (f.type === "seed") {
      const rand = wfRand(wf);
      const lock = h("button", { class: "toggle", type: "button", "aria-pressed": String(!rand[f.key]),
        onclick: () => { rand[f.key] = !rand[f.key]; saveWf(wf); renderWfPanel(); } },
      icon(rand[f.key] ? "dice-five" : "lock-simple"), h("span", {}, rand[f.key] ? "Random" : "Fixed"));
      input.disabled = rand[f.key];
      if (rand[f.key]) input.value = "";
      input.placeholder = "Random each run";
      return h("div", { class: "field" }, h("div", { class: "field-label" }, h("label", { for: id }, f.label), lock), input, help);
    }
    return h("div", { class: "field" }, h("label", { class: "field-label", for: id }, h("span", {}, f.label)), input, help);
  }

  function wfMediaField(wf, f, help) {
    const media = wfMedia(wf);
    const m = media[f.key];
    const video = f.type === "video";
    const accept = video ? "video/*,.mkv,.mov" : "image/*";
    const picker = h("input", { type: "file", accept, hidden: true });
    picker.addEventListener("change", () => { if (picker.files[0]) setWfFile(wf, f, picker.files[0]); picker.value = ""; });
    let slot;
    if (m) {
      const preview = m.video
        ? h("video", { src: m.url, muted: true, loop: true, autoplay: true, playsinline: true, poster: m.poster || null })
        : h("img", { src: m.url, alt: f.label });
      if (m.video) preview.muted = true;
      slot = h("div", { class: `wf-slot filled${video ? " video" : ""}`, title: m.name || f.label },
        preview,
        h("span", { class: "wf-slot-name" }, m.library ? icon("squares-four") : icon(video ? "video" : "image"), m.name || ""),
        h("button", { class: "remove", type: "button", "aria-label": `Remove ${f.label.toLowerCase()}`, onclick: () => { clearWfMedia(wf, f.key); renderWfPanel(); } }, icon("x")));
    } else {
      slot = h("div", { class: `wf-slot empty${video ? " video" : ""}` },
        h("button", { type: "button", class: "wf-slot-pick", onclick: () => picker.click() }, icon(video ? "video" : "image"), h("span", {}, `Add ${f.label.toLowerCase()}`)),
        h("button", { type: "button", class: "link-sm", onclick: () => openPicker(f.type, (rec) => setWfLibrary(wf, f, rec)) }, icon("squares-four"), "From Library"));
    }
    slot.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); slot.classList.add("over"); hideDrop(); });
    slot.addEventListener("dragleave", () => slot.classList.remove("over"));
    slot.addEventListener("drop", (e) => {
      e.preventDefault(); e.stopPropagation(); slot.classList.remove("over"); hideDrop();
      const file = [...(e.dataTransfer.files || [])].find((x) => x.type.startsWith(video ? "video/" : "image/") || (video && /\.(mkv|mov|webm|mp4|gif)$/i.test(x.name)));
      if (file) setWfFile(wf, f, file); else toast(video ? "That file is not a video." : "That file is not an image.", { type: "error" });
    });
    return h("div", { class: "field wf-media-field" }, h("div", { class: "field-label" }, h("span", {}, f.label)), slot, picker, help);
  }

  function clearWfMedia(wf, key) {
    const m = wfMedia(wf)[key];
    if (m && m.blob) URL.revokeObjectURL(m.url);
    delete wfMedia(wf)[key];
  }
  async function setWfFile(wf, f, file) {
    clearWfMedia(wf, f.key);
    const m = { blob: file, url: URL.createObjectURL(file), name: file.name || (f.type === "video" ? "video.mp4" : "image.png"), video: f.type === "video" };
    wfMedia(wf)[f.key] = m;
    renderWfPanel();
    if (!m.video) {
      try { const bmp = await createImageBitmap(file); fitWfSize(wf, f.key, bmp.width, bmp.height); bmp.close(); } catch { /* size stays */ }
    }
  }
  function setWfLibrary(wf, f, rec) {
    clearWfMedia(wf, f.key);
    wfMedia(wf)[f.key] = { library: rec.file, url: isVideo(rec) ? rec.url : rec.thumb, poster: rec.thumb, name: rec.file, video: isVideo(rec) };
    renderWfPanel();
    if (!isVideo(rec)) fitWfSize(wf, f.key, rec.width, rec.height);
  }
  // Match the output size to the start image's shape, keeping roughly the same pixel count
  function fitWfSize(wf, key, iw, ih) {
    const sz = wf.size;
    if (!sz || sz.from !== key || !iw || !ih) return;
    const vals = wfVals(wf);
    const fw = wf.fields.find((x) => x.key === sz.width), fh = wf.fields.find((x) => x.key === sz.height);
    if (!fw || !fh) return;
    const m = sz.multiple || 16;
    const area = (vals[fw.key] || fw.default) * (vals[fh.key] || fh.default);
    const ar = iw / ih;
    const fit = (v, f) => clamp(Math.round(v / m) * m, f.min ?? m, f.max ?? 4096);
    const w = fit(Math.sqrt(area * ar), fw), hh = fit(Math.sqrt(area / ar), fh);
    if (w === vals[fw.key] && hh === vals[fh.key]) return;
    vals[fw.key] = w; vals[fh.key] = hh;
    saveWf(wf);
    renderWfPanel();
    toast(`Size set to ${w} × ${hh} to match the image.`);
  }

  function renderWfPanel() {
    const panel = $("#wfPanel");
    const wf = wfSel();
    if (!wf) { panel.replaceChildren(); return; }
    const hidden = wf.fields.filter((f) => f.hidden);
    const shown = wf.fields.filter((f) => !f.hidden || W.showHidden);
    const texts = shown.filter((f) => f.type === "text");
    const medias = shown.filter((f) => MEDIA.has(f.type));
    const rest = shown.filter((f) => f.type !== "text" && !MEDIA.has(f.type));
    const del = wf.preset ? null : h("button", {
      class: "icon-btn", type: "button", "aria-label": "Delete workflow", title: "Delete workflow",
      onclick: async (e) => {
        const b = e.currentTarget;
        if (!b.classList.contains("confirm")) { b.classList.add("confirm"); b.title = "Click again to delete"; setTimeout(() => b.classList.remove("confirm"), 3000); return; }
        try { await api(`/api/workflows/${wf.slug}`, { method: "DELETE" }); await loadWorkflows(); renderWorkflows(); toast("Workflow deleted.", { type: "ok" }); }
        catch (err) { toast(err.message, { type: "error" }); }
      },
    }, icon("trash"));
    const run = h("button", { class: "btn btn-primary wf-run", type: "button", onclick: () => runWorkflow(wf) }, icon("play", true), h("span", {}, "Run"));
    panel.replaceChildren(
      h("header", { class: "wf-panel-head" },
        h("div", {}, h("h2", {}, wf.name), wf.description ? h("p", { class: "sub" }, wf.description) : null),
        h("div", { class: "wf-panel-tools" },
          h("button", { class: "icon-btn", type: "button", "aria-label": "Edit fields", title: "Edit fields", onclick: () => openFieldEditor(wf) }, icon("faders")),
          del)),
      ...texts.map((f) => wfField(wf, f)),
      medias.length ? h("div", { class: "wf-media" }, ...medias.map((f) => wfField(wf, f))) : null,
      rest.length ? h("div", { class: "wf-grid" }, ...rest.map((f) => wfField(wf, f))) : null,
      hidden.length ? h("button", { class: "link-sm wf-more", type: "button", onclick: () => { W.showHidden = !W.showHidden; renderWfPanel(); } },
        icon(W.showHidden ? "eye-slash" : "eye"), W.showHidden ? "Hide advanced fields" : `Show ${hidden.length} advanced field${hidden.length === 1 ? "" : "s"}`) : null,
      h("div", { class: "wf-run-row" }, run,
        h("span", { class: "hint" }, wf.output === "video" ? "Qwen moves to RAM while the video renders, and comes back on your next image." : "Runs in ComfyUI. Results land in the Library.")),
      h("div", { class: "wf-runs", id: "wfRuns" }),
    );
    renderWfRuns();
  }

  function renderWfRuns() {
    const box = $("#wfRuns");
    if (!box) return;
    const runs = activeJobs().filter((j) => j.kind === "comfy").sort((a, b) => a.created - b.created);
    box.replaceChildren(...runs.map((job) => {
      const p = job.steps ? job.step / job.steps : 0;
      let eta = "";
      if (job.status === "queued") eta = job.ahead ? `${job.ahead} ahead` : "Up next";
      else if (job.step >= 2 && job.first_step_at && job.step < job.steps) eta = `About ${fmtSecs(((job.server_time - job.first_step_at) / (job.step - 1)) * (job.steps - job.step))} left`;
      else if (job.started) eta = `${fmtSecs(job.server_time - job.started)} elapsed`;
      return h("div", { class: "wf-run-item" },
        h("span", { class: "queue-ring" }),
        h("div", { class: "wf-run-t" },
          h("strong", {}, job.workflow || "Workflow"),
          h("small", {}, job.status === "queued" ? "Queued" : job.phase, job.step ? `  ${job.step}/${job.steps}` : ""),
          h("div", { class: "develop-bar", style: `--p:${p}` }, h("span"))),
        h("span", { class: "wf-run-eta" }, eta),
        h("button", { class: "icon-btn", type: "button", "aria-label": "Cancel", title: "Cancel", onclick: () => cancelJob(job.id) }, icon("x")));
    }));
  }

  async function runWorkflow(wf) {
    const vals = wfVals(wf), media = wfMedia(wf);
    const fd = new FormData();
    const library = {}, values = {};
    for (const f of wf.fields) {
      if (MEDIA.has(f.type)) {
        const m = media[f.key];
        if (m && m.blob) fd.append(`file_${f.key}`, m.blob, m.name);
        else if (m && m.library) library[f.key] = m.library;
        else if (!f.hidden && !f.default) return toast(`Add ${f.label.toLowerCase()} first.`, { type: "error" });
      } else if (vals[f.key] != null && (vals[f.key] !== "" || f.type === "text")) values[f.key] = vals[f.key];
    }
    fd.append("params", JSON.stringify({ values, randomize: wfRand(wf), library }));
    const btn = $(".wf-run");
    if (btn) btn.disabled = true;
    try {
      const job = await api(`/api/workflows/${wf.slug}/run`, { method: "POST", body: fd });
      S.jobs.set(job.id, job);
      renderFeed();
      startPolling();
      renderWfRuns();
      if (wf.output === "video" && wf.fields.some((f) => f.type === "image") && !wf.fields.some((f) => f.type === "video")) store.set("animateWf", wf.slug);
      toast(`${wf.name} queued. It also shows in Create.`, { type: "ok" });
    } catch (e) { toast(e.message, { type: "error", timeout: 8000 }); }
    finally { setTimeout(() => { if (btn) btn.disabled = false; }, 400); }
  }

  // Files dropped or pasted anywhere on the Workflows page go to the first matching slot of the open workflow
  function wfDropAnywhere(files) {
    const wf = wfSel();
    if (!wf) return;
    let used = 0;
    for (const file of files) {
      const kind = file.type.startsWith("video/") || /\.(mkv|mov|webm|mp4)$/i.test(file.name) ? "video" : file.type.startsWith("image/") ? "image" : null;
      const slots = wf.fields.filter((f) => f.type === kind);
      const f = slots.find((x) => !wfMedia(wf)[x.key]) || slots[0];
      if (f) { setWfFile(wf, f, file); used++; }
    }
    if (!used) toast(`${wf.name} has no slot for that file.`, { type: "error" });
  }

  async function animate(rec) {
    closeLightbox();
    try { if (!W.loaded) await loadWorkflows(); } catch (e) { return toast(e.message, { type: "error" }); }
    const i2v = (w) => w && w.output === "video" && w.fields.some((f) => f.type === "image") && !w.fields.some((f) => f.type === "video");
    const wf = [wfBy(store.get("animateWf")), wfBy("wan22-i2v"), ...W.items].find(i2v);
    if (!wf) return toast("No image-to-video workflow is installed.", { type: "error" });
    W.sel = wf.slug;
    store.set("wfSel", wf.slug);
    setWfLibrary(wf, wf.fields.find((f) => f.type === "image"), rec);
    go("workflows");
    toast("Image loaded. Describe the motion, then Run.", { type: "ok" });
    setTimeout(() => { const t = $("#wfPanel textarea"); if (t) { t.focus(); t.select(); } }, 120);
  }

  async function reuseWorkflow(rec) {
    try { if (!W.loaded) await loadWorkflows(); } catch (e) { return toast(e.message, { type: "error" }); }
    const wf = wfBy(rec.workflow);
    if (!wf) return toast("That workflow is no longer installed.", { type: "error" });
    W.sel = wf.slug;
    store.set("wfSel", wf.slug);
    const vals = wfVals(wf), rand = wfRand(wf);
    for (const [k, v] of Object.entries(rec.form || {})) if (wf.fields.some((f) => f.key === k)) vals[k] = v;
    for (const f of wf.fields) if (f.type === "seed" && rec.form && f.key in rec.form) rand[f.key] = false;
    saveWf(wf);
    if (rec.source_image) {
      const f = wf.fields.find((x) => x.type === "image");
      const src = findRec(rec.source_image) || { file: rec.source_image, thumb: `/thumbs/${rec.source_image}`, url: `/outputs/${rec.source_image}` };
      if (f) setWfLibrary(wf, f, src);
    }
    closeLightbox();
    go("workflows");
    toast("Workflow settings restored, seed fixed.", { type: "ok" });
  }

  // Library picker for image and video slots
  async function openPicker(kind, onPick) {
    const grid = $("#wfPickerGrid");
    $("#wfPickerTitle").textContent = kind === "video" ? "Pick a video from the Library" : "Pick an image from the Library";
    grid.replaceChildren(h("p", { class: "hint" }, "Loading…"));
    $("#wfPicker").hidden = false;
    let items;
    try { items = (await api("/api/history?limit=200")).items.filter((r) => (kind === "video") === isVideo(r)); }
    catch (e) { grid.replaceChildren(h("p", { class: "hint" }, e.message)); return; }
    if (!items.length) { grid.replaceChildren(h("p", { class: "hint" }, kind === "video" ? "No videos in the Library yet." : "No images in the Library yet.")); return; }
    grid.replaceChildren(...items.map((r) => h("button", { class: "wf-pick", type: "button", title: r.prompt || r.file,
      onclick: () => { $("#wfPicker").hidden = true; onPick(r); } },
    h("img", { src: r.thumb, alt: "", loading: "lazy" }), isVideo(r) ? videoBadge(r) : null)));
  }

  // Import
  function openImport() {
    $("#wfImportName").value = "";
    $("#wfImportJson").value = "";
    $("#wfImport").hidden = false;
    $("#wfImportName").focus();
  }
  async function readImportFile(file) {
    if (!file) return;
    $("#wfImportJson").value = await file.text();
    if (!$("#wfImportName").value.trim()) $("#wfImportName").value = file.name.replace(/\.json$/i, "").replace(/[_-]+/g, " ").trim();
  }
  async function submitImport() {
    try {
      const wf = await api("/api/workflows", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: $("#wfImportName").value.trim(), graph: $("#wfImportJson").value }) });
      $("#wfImport").hidden = true;
      await loadWorkflows();
      W.sel = wf.slug;
      store.set("wfSel", wf.slug);
      renderWorkflows();
      toast(`Imported with ${wf.fields.length} field${wf.fields.length === 1 ? "" : "s"}. Adjust them in Edit fields if needed.`, { type: "ok", timeout: 6000 });
    } catch (e) { toast(e.message, { type: "error", timeout: 8000 }); }
  }

  // Field editor: rename, hide, retype or remove fields, and expose more node inputs
  const isLink = (v) => Array.isArray(v) && v.length === 2 && typeof v[0] === "string";
  async function openFieldEditor(wf) {
    let graph;
    try { graph = await api(`/api/workflows/${wf.slug}/graph`); } catch (e) { return toast(e.message, { type: "error" }); }
    W.edit = { wf, graph, fields: JSON.parse(JSON.stringify(wf.fields)) };
    $("#wfFieldsName").value = wf.name;
    const ids = Object.keys(graph).sort((a, b) => (parseInt(a, 10) || 0) - (parseInt(b, 10) || 0));
    $("#wfAddNode").replaceChildren(...ids.map((id) => h("option", { value: id }, `${id} · ${(graph[id]._meta || {}).title || graph[id].class_type}`)));
    fillAddInputs();
    renderFieldRows();
    $("#wfFields").hidden = false;
  }
  function fillAddInputs() {
    const node = W.edit.graph[$("#wfAddNode").value];
    const names = node ? Object.keys(node.inputs).filter((k) => !isLink(node.inputs[k])) : [];
    $("#wfAddInput").replaceChildren(...(names.length ? names.map((k) => h("option", { value: k }, k)) : [h("option", { value: "" }, "No editable inputs")]));
  }
  function guessType(node, name) {
    const v = node.inputs[name];
    if (node.class_type === "LoadImage" && name === "image") return "image";
    if (/LoadVideo/.test(node.class_type)) return "video";
    if (typeof v === "boolean") return "bool";
    if (typeof v === "number") return /seed/.test(name) ? "seed" : Number.isInteger(v) ? "int" : "float";
    return "text";
  }
  function renderFieldRows() {
    const { graph, fields } = W.edit;
    const describe = (f) => f.targets.map((t) => `${t[0]} · ${(graph[t[0]] && ((graph[t[0]]._meta || {}).title || graph[t[0]].class_type)) || "missing"} · ${t[1]}${t[2] ? ` (${t[2]})` : ""}`).join("\n");
    $("#wfFieldRows").replaceChildren(...fields.map((f, i) => {
      const label = h("input", { class: "input", value: f.label, "aria-label": "Label", maxlength: "60" });
      label.addEventListener("input", () => { f.label = label.value; });
      const type = h("select", { class: "input", "aria-label": "Type" }, ...["text", "int", "float", "seed", "bool", "image", "video"].map((t) => h("option", { value: t, selected: t === f.type || null }, t)));
      type.addEventListener("change", () => { f.type = type.value; });
      const move = (d) => { const j = i + d; if (j < 0 || j >= fields.length) return; [fields[i], fields[j]] = [fields[j], fields[i]]; renderFieldRows(); };
      return h("div", { class: `wf-frow${f.hidden ? " is-hidden" : ""}` },
        h("div", { class: "wf-frow-main" }, label, type),
        h("div", { class: "wf-frow-target mono", title: describe(f) }, describe(f).split("\n")[0], f.targets.length > 1 ? ` +${f.targets.length - 1}` : ""),
        h("div", { class: "wf-frow-tools" },
          h("button", { class: "icon-btn", type: "button", "aria-label": "Move up", title: "Move up", onclick: () => move(-1) }, icon("arrow-up")),
          h("button", { class: "icon-btn", type: "button", "aria-label": "Move down", title: "Move down", onclick: () => move(1) }, icon("arrow-down")),
          h("button", { class: "icon-btn", type: "button", "aria-pressed": String(!!f.hidden), "aria-label": f.hidden ? "Show in form" : "Hide (advanced)", title: f.hidden ? "Hidden under advanced. Click to show." : "Shown. Click to hide under advanced.",
            onclick: () => { f.hidden = !f.hidden; renderFieldRows(); } }, icon(f.hidden ? "eye-slash" : "eye")),
          h("button", { class: "icon-btn", type: "button", "aria-label": "Remove field", title: "Remove field", onclick: () => { fields.splice(i, 1); renderFieldRows(); } }, icon("trash"))));
    }));
  }
  function addField() {
    const { graph, fields } = W.edit;
    const id = $("#wfAddNode").value, name = $("#wfAddInput").value;
    if (!id || !name) return;
    const node = graph[id];
    const type = guessType(node, name);
    let key = `${name}_${id}`.replace(/\W/g, "_");
    while (fields.some((f) => f.key === key)) key += "_";
    fields.push({ key, label: name.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase()), type, default: MEDIA.has(type) ? "" : node.inputs[name], targets: [[id, name]], hidden: false });
    renderFieldRows();
    $("#wfFieldRows").lastChild?.scrollIntoView({ block: "nearest" });
  }
  async function saveFields() {
    const { wf, fields } = W.edit;
    try {
      const out = await api(`/api/workflows/${wf.slug}`, { method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: $("#wfFieldsName").value.trim(), fields }) });
      $("#wfFields").hidden = true;
      delete W.vals[wf.slug];
      await loadWorkflows();
      W.sel = out.slug;
      renderWorkflows();
      toast("Fields saved.", { type: "ok" });
    } catch (e) { toast(e.message, { type: "error", timeout: 8000 }); }
  }

  function wireWorkflows() {
    $("#wfImportBtn").addEventListener("click", openImport);
    $("#wfEditor").addEventListener("click", openEditor);
    $("#wfImportForm").addEventListener("submit", (e) => { e.preventDefault(); submitImport(); });
    $("#wfImportFile").addEventListener("change", (e) => { readImportFile(e.target.files[0]); e.target.value = ""; });
    const imp = $("#wfImport");
    imp.addEventListener("dragover", (e) => { e.preventDefault(); e.stopPropagation(); });
    imp.addEventListener("drop", (e) => { e.preventDefault(); e.stopPropagation(); hideDrop(); readImportFile((e.dataTransfer.files || [])[0]); });
    $("#wfAddNode").addEventListener("change", fillAddInputs);
    $("#wfAddBtn").addEventListener("click", addField);
    $("#wfFieldsSave").addEventListener("click", saveFields);
    for (const id of ["#wfImport", "#wfFields", "#wfPicker"]) {
      $(id).addEventListener("click", (e) => { if (e.target.id === id.slice(1) || e.target.closest("[data-close]")) $(id).hidden = true; });
    }
    $("#lbAnimate").addEventListener("click", () => animate(LB.list[LB.i]));
    $("#lbVideo").addEventListener("volumechange", (e) => store.set("videoMuted", e.target.muted));
  }

  // ---------------------------------------------------------------- drag, drop, paste
  let dragDepth = 0;
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  function hideDrop() { dragDepth = 0; $("#dropzone").hidden = true; }
  window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); dragDepth++;
    const wf = $("#app").dataset.view === "workflows";
    $("#dropzone strong").textContent = wf ? "Drop into the workflow" : "Drop to use as reference";
    $("#dropzone span").textContent = wf ? "Images and videos fill the open workflow's slots." : "Obscura switches to Edit or Blend for you.";
    $("#dropzone").hidden = false;
  });
  window.addEventListener("dragleave", (e) => { if (!hasFiles(e)) return; dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) hideDrop(); });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); hideDrop();
    if ($("#app").dataset.view === "workflows") return wfDropAnywhere([...e.dataTransfer.files]);
    addImages([...e.dataTransfer.files]);
  });
  window.addEventListener("paste", (e) => {
    if (typing(document.activeElement) && !(e.clipboardData?.files || []).length) return;
    const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith("image/") || f.type.startsWith("video/"));
    if (!files.length) return;
    e.preventDefault();
    if ($("#app").dataset.view === "workflows") return wfDropAnywhere(files);
    addImages(files.filter((f) => f.type.startsWith("image/")));
  });

  // ---------------------------------------------------------------- keyboard
  const typing = (el) => el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!$("#rewrite").hidden) return closeRewrite();
      const modal = ["#wfPicker", "#wfFields", "#wfImport"].find((id) => !$(id).hidden);
      if (modal) return ($(modal).hidden = true);
      if (!$("#shortcuts").hidden) return ($("#shortcuts").hidden = true);
      if (!$("#lightbox").hidden) return Z.s > 1.001 ? zoomReset(true) : closeLightbox();
      if ($("#inspector").classList.contains("open")) return closeInspector();
      if (SEL.files.size) return clearSelection();
      if (typing(document.activeElement)) return document.activeElement.blur();
    }
    if (!$("#rewrite").hidden) return; // the popup handles its own keys
    if (["#wfPicker", "#wfFields", "#wfImport"].some((id) => !$(id).hidden)) return;
    if (!$("#lightbox").hidden) {
      if (typing(document.activeElement)) return; // typing in the enhance prompt, not navigating
      if (isVideo(LB.list[LB.i]) && e.key === " ") { e.preventDefault(); const v = $("#lbVideo"); v.paused ? v.play() : v.pause(); return; }
      if (e.key === "ArrowLeft") { e.preventDefault(); lbStep(-1); }
      if (e.key === "ArrowRight") { e.preventDefault(); lbStep(1); }
      if (e.key === "+" || e.key === "=") { e.preventDefault(); zoomCenter(1.5); }
      if (e.key === "-") { e.preventDefault(); zoomCenter(1 / 1.5); }
      if (e.key === "0") { e.preventDefault(); zoomReset(true); }
      return;
    }
    if (typing(document.activeElement) || e.metaKey || e.ctrlKey || e.altKey) return;
    const view = $("#app").dataset.view;
    if (e.key === "/") { e.preventDefault(); (view === "library" ? $("#librarySearch") : view === "create" ? $("#prompt") : null)?.focus(); }
    if (e.key === "5") go("workflows");
    if (e.key === "6") go("liked");
    if (e.key === "1") go("create");
    if (e.key === "2") go("library");
    if (e.key === "3") go("styles");
    if (e.key === "4") go("prompts");
    if (e.key === "?") $("#shortcuts").hidden = false;
    if (e.key === "\\" && view === "create") { e.preventDefault(); toggleSettings(); }
  });

  // ---------------------------------------------------------------- wiring
  function wire() {
    window.addEventListener("hashchange", route);
    $("#promptNew").addEventListener("click", () => openPromptEditor());
    $("#promptCancel").addEventListener("click", closePromptEditor);
    $("#promptEditor").addEventListener("submit", (e) => { e.preventDefault(); savePrompt(); });
    window.addEventListener("resize", debounce(() => { if ($("#app").dataset.view === "prompts") measurePrompts(); }, 150));
    $("#composer").addEventListener("submit", (e) => { e.preventDefault(); generate({ rewrite: true }); });
    const prompt = $("#prompt");
    prompt.addEventListener("input", autosize);
    prompt.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); generate({ rewrite: true }); }
    });
    $("#filePicker").addEventListener("change", (e) => {
      const f = e.target.files[0];
      if (f) setRef(pickSlot, f);
      e.target.value = "";
    });
    $("#steps").addEventListener("input", (e) => { S.steps = +e.target.value; syncControls(); });
    $("#strength").addEventListener("input", (e) => { S.strength = +e.target.value; syncControls(); });
    $("#seed").addEventListener("change", (e) => {
      const v = parseInt(e.target.value.replace(/\D/g, ""), 10);
      if (Number.isFinite(v)) { S.seed = v % (S.cfg.max_seed + 1); S.seedLocked = true; }
      syncControls();
    });
    $("#seedLock").addEventListener("click", () => { S.seedLocked = !S.seedLocked; syncControls(); });
    $("#seedDice").addEventListener("click", () => { S.seed = Math.floor(Math.random() * S.cfg.max_seed); S.seedLocked = true; syncControls(); });
    $("#styleCurrent").addEventListener("click", () => go("styles"));
    for (const id of ["#customRepo", "#customFile", "#localLoraInput"]) $(id).addEventListener("change", save);
    $("#refreshLoras").addEventListener("click", async () => {
      try {
        const r = await api("/api/local-loras");
        fillLocalLoras(r.items);
        toast(r.items.length ? `${r.items.length} LoRA file${r.items.length === 1 ? "" : "s"} found.` : "No .safetensors files in the folder yet.");
      } catch (e) { toast(e.message, { type: "error" }); }
    });
    $("#settingsToggle").addEventListener("click", openInspector);
    $("#inspectorCollapse").addEventListener("click", () => setCollapsed(true));
    setCollapsed(store.get("inspectorCollapsed", false));
    // Collapsing only applies to the docked sidebar; the phone/tablet sheet ignores it
    matchMedia("(max-width: 1180px)").addEventListener("change", () => { $("#inspector").inert = !narrow() && $("#app").classList.contains("inspector-collapsed"); });
    $("#settingsToggle2").addEventListener("click", openInspector);
    $("#inspectorClose").addEventListener("click", closeInspector);
    $("#scrim").addEventListener("click", closeInspector);
    $("#queuePill").addEventListener("click", () => { go("create"); setTimeout(scrollFeedTop, 50); });
    $("#shortcutsBtn").addEventListener("click", () => { $("#shortcuts").hidden = false; });
    $("#shortcuts").addEventListener("click", (e) => { if (e.target.id === "shortcuts" || e.target.closest("[data-close]")) $("#shortcuts").hidden = true; });

    // Lightbox
    $("#lbClose").addEventListener("click", closeLightbox);
    $("#lbLike").addEventListener("click", () => { const rec = LB.list[LB.i]; if (rec) toggleLike(rec); });
    $("#lbPrev").addEventListener("click", () => lbStep(-1));
    $("#lbNext").addEventListener("click", () => lbStep(1));
    wireZoom();
    wireSelection();
    wireRewrite();
    wireWorkflows();
    $("#lbMore").addEventListener("click", () => {
      const open = $("#lbPrompt").classList.toggle("clamped") === false;
      $("#lbMore").classList.toggle("open", open);
      $("#lbMore span").textContent = open ? "Collapse prompt" : "Show full prompt";
    });
    $("#lbCopy").addEventListener("click", async () => {
      copyPrompt(LB.list[LB.i].prompt);
    });
    $("#lbReuse").addEventListener("click", () => {
      const rec = LB.list[LB.i];
      if (isComfy(rec)) return reuseWorkflow(rec);
      reuse(rec); closeLightbox(); go("create"); toast("Settings restored, seed fixed.", { type: "ok" });
    });
    $("#lbEdit").addEventListener("click", () => editRecord(LB.list[LB.i]));
    $("#lbDelete").addEventListener("click", lbDelete);
    $$("#lbUpscale [data-up]").forEach((b) => b.addEventListener("click", () => upscale(LB.list[LB.i], b.dataset.up)));
    $("#enhStrength").addEventListener("input", renderEnhanceHint);
    $("#lbEnhanceBtn").addEventListener("click", () => enhance(LB.list[LB.i], Number($("#enhStrength").value), $("#enhPrompt").value.trim() || S.cfg.enhance.default_prompt));
    $("#enhAddOriginal").addEventListener("click", () => {
      const rec = LB.list[LB.i], box = $("#enhPrompt");
      if (!rec || !rec.prompt || box.value.includes(rec.prompt)) return;
      box.value = box.value.trim() ? `${rec.prompt}, ${box.value.trim()}` : rec.prompt;
      box.focus();
    });

    // Library search + infinite scroll
    $("#librarySearch").addEventListener("input", debounce((e) => { S.lib.q = e.target.value.trim(); loadLibrary(true); }, 250));
    new IntersectionObserver((ents) => {
      if (ents[0].isIntersecting && S.lib.loaded && !S.lib.done && $("#app").dataset.view === "library") loadLibrary(false);
    }, { root: $("#view-library"), rootMargin: "800px" }).observe($("#librarySentinel"));

    // Layout
    new ResizeObserver(() => { placeThumb(); if (feedColumns() !== FEED.cols) renderFeed(); }).observe($("#feed"));
    const ro = new ResizeObserver(relayoutMasonry);
    ro.observe($("#view-library")); ro.observe($("#view-liked"));
    // The prompt box floats over the grid; pad the grid so its last row can scroll clear of it
    new ResizeObserver(() => $("#feed").style.setProperty("--composer-h", $("#composer").offsetHeight + "px")).observe($("#composer"));
    new IntersectionObserver((ents) => {
      if (ents[0].isIntersecting && S.feed.loaded && !S.feed.done) loadFeed(false);
    }, { root: $("#feed"), rootMargin: "1200px" }).observe($("#feedSentinel"));
    document.fonts && document.fonts.ready.then(placeThumb);
  }
  function fillLocalLoras(items) {
    $("#localLoraList").replaceChildren(...items.map((x) => h("option", { value: x })));
    if (!$("#localLoraInput").value && items.length) $("#localLoraInput").value = items[0];
  }

  // ---------------------------------------------------------------- boot
  async function boot() {
    try {
      S.cfg = await api("/api/config");
    } catch (e) {
      $("#stageEmpty").hidden = false;
      $("#stageEmpty").querySelector("h1").textContent = "Cannot reach the studio";
      $("#stageEmpty").querySelector("p").textContent = "The model server is not responding. It may still be loading.";
      setTimeout(boot, 4000);
      return;
    }
    const c = S.cfg;
    S.status = c.status;
    $("#loraDir").textContent = c.lora_dir;
    fillLocalLoras(c.local_loras);

    const p = store.get("prefs", {});
    S.lora = allStyles().some((s) => s.id === p.lora) ? p.lora : c.none_lora;
    // v2 made 9:16 the default; ratios saved before that were the old 1:1 default
    // v3 made 4MP and 40 steps the defaults; resolution and steps saved before that are dropped
    if (p.v >= 2 && c.ratios.some((r) => r.ratio === p.ratio)) S.ratio = p.ratio;
    if (p.v >= 3 && p.steps >= 4 && p.steps <= 40) S.steps = p.steps;
    if (typeof p.strength === "number") S.strength = p.strength;
    if (Number.isFinite(p.seed)) S.seed = p.seed;
    S.seedLocked = !!p.seedLocked;
    if ([1, 2, 3, 4, 6, 8].includes(p.batch)) S.batch = p.batch;
    $("#batchOpts").replaceChildren(...[1, 2, 3, 4, 6, 8].map((n) => h("button", {
      class: "count", type: "button", role: "radio", "data-n": String(n), "aria-label": `${n} image${n === 1 ? "" : "s"} per click`,
      onclick: () => { S.batch = n; syncControls(); },
    }, String(n))));
    if (p.customRepo) $("#customRepo").value = p.customRepo;
    if (p.customFile) $("#customFile").value = p.customFile;
    if (p.localLora) $("#localLoraInput").value = p.localLora;

    buildModes();
    buildRatios();
    buildResolution();
    if (p.v >= 3 && c.resolutions.includes(p.res)) S.res = p.res;
    renderModels();
    if (S.status.switch.state === "loading") pollStatus();
    $("#ideaChips").replaceChildren(...IDEAS.map(([text, lora], i) => h("button", {
      class: "idea-chip", type: "button", style: `--i:${i}`,
      onclick: () => { $("#prompt").value = text; autosize(); if (loraSpec(lora)) selectStyle(lora, { quiet: true }); $("#prompt").focus(); },
    }, icon("sparkle", true), h("span", {}, text))));
    // Blend is the default mode every time the app opens
    setMode(BLEND, { quiet: true });
    syncControls();
    wire();
    route();
    autosize();

    // The create grid, plus real covers for styles from recent renders
    loadFeed(true);
    try {
      const hist = await api("/api/history?limit=200");
      for (const r of hist.items) if (r.lora && !S.covers[r.lora]) S.covers[r.lora] = r.thumb;
      renderStyleCurrent();
      if ($("#app").dataset.view === "styles") renderStyles();
    } catch { /* history is optional */ }
  }
  boot();
})();
