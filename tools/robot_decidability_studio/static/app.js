"use strict";

const state = {
  manifest: null,
  setups: [],
  annotations: new Map(),
  setupIndex: 0,
  viewIndex: 0,
  draft: {},
  filter: "pending",
  search: "",
  annotator: localStorage.getItem("robot_decidability_annotator") || "",
  saving: false,
};

const $ = (id) => document.getElementById(id);
const keyFor = (setup, view) => `${setup.setup_id}::${view.view_id}::${setup.claim_id}`;
const activeSetup = () => state.setups[state.setupIndex];
const activeView = () => activeSetup()?.views[state.viewIndex];
const activeRecord = () => state.annotations.get(keyFor(activeSetup(), activeView()));
const clone = (value) => JSON.parse(JSON.stringify(value));

function taskType(record) {
  if (record?.workflow_status === "review" || record?.payload?.flagged) return "review";
  if (record?.workflow_status === "complete") return "complete";
  if (record?.payload?.decidability_locked && record.payload.claim_decidable === "no") return "occlusion";
  return "new";
}

function isPending(record) {
  return !record || record.workflow_status !== "complete";
}

function matchesFilter(record) {
  const type = taskType(record);
  if (state.filter === "all") return true;
  if (state.filter === "pending") return isPending(record);
  return type === state.filter;
}

function setupMatchesSearch(setup) {
  const haystack = `${setup.setup_id} ${setup.claim_id} ${setup.claim_text} ${setup.target_step} ${setup.assembly_family}`.toLowerCase();
  return haystack.includes(state.search);
}

function recordCounts() {
  const records = [...state.annotations.values()];
  return {
    total: records.length,
    complete: records.filter((record) => record.workflow_status === "complete").length,
    occlusion: records.filter((record) => taskType(record) === "occlusion").length,
    newTasks: records.filter((record) => taskType(record) === "new").length,
    review: records.filter((record) => taskType(record) === "review").length,
  };
}

async function bootstrap() {
  const response = await fetch("/api/bootstrap");
  if (!response.ok) throw new Error("Could not load the annotation project.");
  const data = await response.json();
  state.manifest = data.manifest;
  state.setups = data.manifest.setups;
  for (const record of data.annotations) state.annotations.set(record.entity_key, record);
  $("annotatorInput").value = state.annotator;
  selectFirstPending();
  bindEvents();
  renderAll();
}

function selectFirstPending() {
  for (let setupIndex = 0; setupIndex < state.setups.length; setupIndex += 1) {
    const setup = state.setups[setupIndex];
    for (let viewIndex = 0; viewIndex < setup.views.length; viewIndex += 1) {
      const record = state.annotations.get(keyFor(setup, setup.views[viewIndex]));
      if (isPending(record)) {
        state.setupIndex = setupIndex;
        state.viewIndex = viewIndex;
        loadDraft();
        return;
      }
    }
  }
  loadDraft();
}

function loadDraft() {
  const record = activeRecord();
  state.draft = clone(record?.payload || {
    claim_decidable: "",
    explicit_occlusion: "",
    legacy_human_utility: "",
    decidability_locked: false,
    decidability_source: "new_annotation",
    flagged: false,
    notes: "",
  });
}

function bindEvents() {
  $("annotatorInput").addEventListener("change", (event) => {
    state.annotator = event.target.value.trim();
    localStorage.setItem("robot_decidability_annotator", state.annotator);
  });
  $("searchInput").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    renderQueue();
  });
  $("taskFilter").addEventListener("change", (event) => {
    state.filter = event.target.value;
    renderQueue();
  });
  $("nextPendingButton").addEventListener("click", () => navigatePending(1));
  $("previousSetupButton").addEventListener("click", () => moveSetup(-1));
  $("nextSetupButton").addEventListener("click", () => moveSetup(1));
  $("decidableControl").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-value]");
    if (button) setDecidable(button.dataset.value);
  });
  $("occlusionControl").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-value]");
    if (button) setOcclusion(button.dataset.value);
  });
  $("flagCheckbox").addEventListener("change", (event) => {
    state.draft.flagged = event.target.checked;
    renderValidation();
  });
  $("notesInput").addEventListener("input", (event) => { state.draft.notes = event.target.value; });
  $("saveButton").addEventListener("click", () => saveCurrent("draft", false));
  $("completeButton").addEventListener("click", () => completeCurrent(false));
  $("exportButton").addEventListener("click", () => { window.location.href = "/api/export"; });
  $("helpButton").addEventListener("click", () => $("helpDialog").showModal());
  $("closeHelpButton").addEventListener("click", () => $("helpDialog").close());
  $("activeImage").addEventListener("load", () => $("imageLoading").classList.add("hidden"));
  document.addEventListener("keydown", handleKey);
}

function renderAll() {
  renderProgress();
  renderQueue();
  renderContext();
  renderViewTabs();
  renderImage();
  renderLabelPanel();
}

function renderProgress() {
  const counts = recordCounts();
  const percent = counts.total ? Math.round(100 * counts.complete / counts.total) : 0;
  $("progressText").textContent = `${counts.complete}/${counts.total} complete | ${counts.occlusion} occlusion-only | ${counts.newTasks} new`;
  $("progressPercent").textContent = `${percent}%`;
  $("progressBar").style.width = `${percent}%`;
}

function renderQueue() {
  const queue = $("setupQueue");
  const visible = [];
  state.setups.forEach((setup, setupIndex) => {
    if (!setupMatchesSearch(setup)) return;
    const matching = setup.views.filter((view) => matchesFilter(state.annotations.get(keyFor(setup, view))));
    if (!matching.length) return;
    const records = setup.views.map((view) => state.annotations.get(keyFor(setup, view)));
    const pending = records.filter(isPending).length;
    const type = matching.some((view) => taskType(state.annotations.get(keyFor(setup, view))) === "review") ? "review"
      : matching.some((view) => taskType(state.annotations.get(keyFor(setup, view))) === "occlusion") ? "occlusion"
      : matching.some((view) => taskType(state.annotations.get(keyFor(setup, view))) === "new") ? "new" : "complete";
    visible.push({setup, setupIndex, pending, type});
  });
  $("queueCount").textContent = `${visible.length} setup${visible.length === 1 ? "" : "s"}`;
  queue.innerHTML = visible.map(({setup, setupIndex, pending, type}) => `
    <button class="setup-item ${setupIndex === state.setupIndex ? "active" : ""}" data-setup-index="${setupIndex}">
      <strong>${escapeHtml(setup.setup_id)}</strong><span class="counts">${6 - pending}/6</span>
      <span class="claim">${escapeHtml(setup.claim_text)}</span>
      <span class="task-kind ${type}">${type === "occlusion" ? "occlusion only" : type === "new" ? "new decision" : type}</span>
    </button>`).join("");
  queue.querySelectorAll("[data-setup-index]").forEach((button) => {
    button.addEventListener("click", () => selectSetup(Number(button.dataset.setupIndex)));
  });
}

function renderContext() {
  const setup = activeSetup();
  if (!setup) return;
  $("contextMeta").textContent = `${setup.setup_id} | ${setup.target_step} | ${setup.claim_id}`;
  $("claimText").textContent = setup.claim_text;
  const ref = state.manifest.project.family_reference[setup.assembly_family];
  $("familyText").textContent = ref
    ? `Family ${setup.assembly_family}: ${ref.housing_cover} housing/cover, ${ref.small_gear} small gear, ${ref.big_gear} big gear.`
    : `Family ${setup.assembly_family}`;
}

function renderViewTabs() {
  const setup = activeSetup();
  $("viewTabs").innerHTML = setup.views.map((view, index) => {
    const record = state.annotations.get(keyFor(setup, view));
    const type = taskType(record);
    const css = type === "occlusion" ? "pending-occlusion" : type;
    return `<button class="view-tab ${css} ${index === state.viewIndex ? "active" : ""}" data-view-index="${index}">${view.view_id}</button>`;
  }).join("");
  $("viewTabs").querySelectorAll("[data-view-index]").forEach((button) => {
    button.addEventListener("click", () => selectView(Number(button.dataset.viewIndex)));
  });
}

function renderImage() {
  const setup = activeSetup();
  const view = activeView();
  if (!view) return;
  $("imageLoading").classList.remove("hidden");
  $("activeImage").src = `/media?path=${encodeURIComponent(view.image)}`;
  $("imageCaption").textContent = `${setup.setup_id} | ${view.view_id} ${view.view_name} | ${view.elevation} ${view.azimuth}`;
}

function renderLabelPanel() {
  const view = activeView();
  const record = activeRecord();
  if (!view || !record) return;
  $("viewMeta").textContent = `${view.view_id} | revision ${record.revision}`;
  const badge = $("workflowBadge");
  badge.textContent = record.workflow_status;
  badge.className = `status-badge ${record.workflow_status}`;
  const old = state.draft.legacy_human_utility;
  $("legacySection").classList.toggle("hidden", old === "");
  if (old !== "") {
    const oldText = old === "2" ? "2 clear: decidable and already complete."
      : old === "1" ? "1 helpful: not decidable; only occlusion remains."
      : "0 not useful: not decidable; only occlusion remains.";
    $("legacyText").textContent = oldText;
    const provenance = state.draft.legacy_provenance || {};
    $("legacySource").textContent = `Source: existing human annotation${provenance.annotator ? ` (${provenance.annotator})` : ""}`;
  }
  const locked = Boolean(state.draft.decidability_locked);
  $("lockedMessage").classList.toggle("hidden", !locked);
  $("decidableControl").querySelectorAll("button").forEach((button) => {
    button.disabled = locked;
    button.classList.toggle("selected", button.dataset.value === state.draft.claim_decidable);
  });
  const showOcclusion = state.draft.claim_decidable === "no";
  $("occlusionSection").classList.toggle("hidden", !showOcclusion);
  $("occlusionControl").querySelectorAll("button").forEach((button) => {
    button.classList.toggle("selected", button.dataset.value === state.draft.explicit_occlusion);
  });
  $("flagCheckbox").checked = Boolean(state.draft.flagged);
  $("notesInput").value = state.draft.notes || "";
  renderValidation();
}

function renderValidation() {
  const message = $("validationMessage");
  if (state.draft.flagged) {
    message.textContent = "This view will enter the reviewer queue.";
    message.className = "validation-message";
  } else if (!state.draft.claim_decidable) {
    message.textContent = "Answer Question 1. Existing labels are reused automatically.";
    message.className = "validation-message";
  } else if (state.draft.claim_decidable === "no" && !state.draft.explicit_occlusion) {
    message.textContent = "Only the explicit occlusion check remains.";
    message.className = "validation-message";
  } else {
    message.textContent = "Annotation is complete and internally consistent.";
    message.className = "validation-message ok";
  }
}

function selectSetup(index) {
  state.setupIndex = index;
  const setup = activeSetup();
  let next = setup.views.findIndex((view) => matchesFilter(state.annotations.get(keyFor(setup, view))));
  if (next < 0) next = setup.views.findIndex((view) => isPending(state.annotations.get(keyFor(setup, view))));
  state.viewIndex = next < 0 ? 0 : next;
  loadDraft();
  renderAll();
}

function selectView(index) {
  state.viewIndex = index;
  loadDraft();
  renderViewTabs();
  renderImage();
  renderLabelPanel();
}

function moveSetup(delta) {
  const next = Math.max(0, Math.min(state.setups.length - 1, state.setupIndex + delta));
  selectSetup(next);
}

function setDecidable(value) {
  if (state.draft.decidability_locked || state.saving) return;
  state.draft.claim_decidable = value;
  if (value === "yes") {
    state.draft.explicit_occlusion = "";
    renderLabelPanel();
    completeCurrent(true);
  } else {
    state.draft.explicit_occlusion = "";
    renderLabelPanel();
  }
}

function setOcclusion(value) {
  if (state.draft.claim_decidable !== "no" || state.saving) return;
  state.draft.explicit_occlusion = value;
  renderLabelPanel();
  completeCurrent(true);
}

function validForCompletion() {
  if (!state.draft.claim_decidable) return false;
  return state.draft.claim_decidable !== "no" || ["yes", "no"].includes(state.draft.explicit_occlusion);
}

async function completeCurrent(fromQuickAction) {
  if (!validForCompletion()) {
    showToast("Complete the required binary question first.", true);
    return;
  }
  const status = state.draft.flagged ? "review" : "complete";
  const saved = await saveCurrent(status, false);
  if (saved && (fromQuickAction ? $("autoAdvanceCheckbox").checked : true)) navigatePending(1);
}

async function saveCurrent(workflowStatus, navigate) {
  if (state.saving) return false;
  if (!state.annotator) {
    $("annotatorInput").focus();
    showToast("Enter an annotator ID before saving.", true);
    return false;
  }
  state.saving = true;
  const setup = activeSetup();
  const view = activeView();
  const record = activeRecord();
  try {
    const response = await fetch("/api/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        setup_id: setup.setup_id,
        view_id: view.view_id,
        claim_id: setup.claim_id,
        payload: state.draft,
        workflow_status: workflowStatus,
        annotator: state.annotator,
        expected_revision: record.revision,
      }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Save failed.");
    state.annotations.set(data.entity_key, data);
    state.draft = clone(data.payload);
    renderAll();
    if (navigate) navigatePending(1);
    return true;
  } catch (error) {
    showToast(error.message, true);
    return false;
  } finally {
    state.saving = false;
  }
}

function navigatePending(direction) {
  const flat = [];
  state.setups.forEach((setup, setupIndex) => setup.views.forEach((view, viewIndex) => {
    if (isPending(state.annotations.get(keyFor(setup, view)))) flat.push({setupIndex, viewIndex});
  }));
  if (!flat.length) {
    showToast("All annotation tasks are complete.");
    return;
  }
  let current = flat.findIndex((item) => item.setupIndex === state.setupIndex && item.viewIndex === state.viewIndex);
  if (current < 0) current = direction > 0 ? -1 : 0;
  const target = flat[(current + direction + flat.length) % flat.length];
  state.setupIndex = target.setupIndex;
  state.viewIndex = target.viewIndex;
  loadDraft();
  renderAll();
}

function handleKey(event) {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (key === "1") setDecidable("yes");
  else if (key === "0") setDecidable("no");
  else if (key === "o") setOcclusion("yes");
  else if (key === "v") setOcclusion("no");
  else if (key === "j") navigatePending(1);
  else if (key === "k") navigatePending(-1);
}

function showToast(message, error = false) {
  const toast = $("toast");
  toast.textContent = message;
  toast.className = `toast${error ? " error" : ""}`;
  setTimeout(() => toast.classList.add("hidden"), 2600);
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

bootstrap().catch((error) => showToast(error.message, true));
