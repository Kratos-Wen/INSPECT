"use strict";

const app = {
  manifest: null,
  setups: [],
  annotations: new Map(),
  selectedSetup: 0,
  selectedView: 0,
  workflow: "view",
  filter: "all",
  search: "",
  annotator: localStorage.getItem("robot_annotation_annotator") || "",
  blindMode: localStorage.getItem("robot_annotation_blind") !== "false",
  setupDraft: {},
  viewDraft: {},
  setupRevision: 0,
  viewRevision: 0,
  setupStatus: "draft",
  viewStatus: "draft",
  dirtySetup: false,
  dirtyView: false,
  saveTimer: null,
  geometryTool: "pan",
  geometryHistory: [],
  draftBox: null,
};

const el = (id) => document.getElementById(id);
const deepClone = (value) => JSON.parse(JSON.stringify(value));
const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

function annotationKey(setup, view = null) {
  if (!view) return setup.setup_id;
  return `${setup.setup_id}::${view.view_id}::${setup.claim_id || ""}`;
}

function activeSetup() {
  return app.setups[app.selectedSetup] || null;
}

function activeView() {
  const setup = activeSetup();
  return setup?.views?.[app.selectedView] || null;
}

function ontologyList(name) {
  return app.manifest?.ontology?.[name] || [];
}

function itemId(item) {
  return typeof item === "string" ? item : item.id;
}

function itemLabel(item) {
  return typeof item === "string" ? item.replaceAll("_", " ") : (item.label || item.id);
}

function defaultSetupPayload(setup) {
  return {
    assembly_family: "",
    active_claim_id: setup?.claim_id || "",
    physical_claim_truth: "",
    error_type: "",
    target_part_identity: "",
    housing_identity: "",
    cover_identity: "",
    gear_orientation: "",
    insertion_state: "",
    alignment_state: "",
    cover_seating_state: "",
    counterfactual_type: "",
    notes: "",
  };
}

function defaultViewPayload(setup) {
  const roleStates = {};
  for (const role of ontologyList("evidence_roles")) roleStates[itemId(role)] = "unrated";
  return {
    oracle_utility: "",
    observable_decision: "",
    role_states: roleStates,
    object_visibility: {},
    relations: {},
    occlusion_level: "",
    identity_ambiguous: false,
    decisive_counterfactual: "",
    short_rationale: "",
    boxes: [],
    keypoints: [],
    required_roles_snapshot: deepClone(setup?.required_roles || []),
  };
}

async function bootstrap() {
  try {
    const response = await fetch("/api/bootstrap", {cache: "no-store"});
    if (!response.ok) throw new Error(`Server returned ${response.status}`);
    const data = await response.json();
    app.manifest = data.manifest;
    app.setups = data.manifest.setups || [];
    for (const annotation of data.annotations || []) app.annotations.set(annotation.entity_key, annotation);
    el("annotatorInput").value = app.annotator;
    el("blindModeToggle").checked = app.blindMode;
    bindEvents();
    populateGeometryLabels();
    loadSelection();
    renderAll();
  } catch (error) {
    showToast(`Could not load project: ${error.message}`, "error", 8000);
  }
}

function bindEvents() {
  el("annotatorInput").addEventListener("change", (event) => {
    app.annotator = event.target.value.trim();
    localStorage.setItem("robot_annotation_annotator", app.annotator);
    if (app.annotator) scheduleAutosave();
  });
  el("searchInput").addEventListener("input", (event) => {
    app.search = event.target.value.trim().toLowerCase();
    renderQueue();
  });
  el("statusFilter").addEventListener("change", (event) => {
    app.filter = event.target.value;
    renderQueue();
  });
  el("nextIncompleteButton").addEventListener("click", () => moveToNextIncomplete());
  el("prevSetupButton").addEventListener("click", () => moveSetup(-1));
  el("nextSetupButton").addEventListener("click", () => moveSetup(1));
  el("viewTab").addEventListener("click", () => setWorkflow("view"));
  el("setupTab").addEventListener("click", () => setWorkflow("setup"));
  el("blindModeToggle").addEventListener("change", (event) => {
    app.blindMode = event.target.checked;
    localStorage.setItem("robot_annotation_blind", String(app.blindMode));
    renderContext();
  });
  el("shortcutButton").addEventListener("click", () => el("shortcutDialog").showModal());
  el("qualityButton").addEventListener("click", openQualityDialog);
  document.querySelectorAll("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog").close());
  });
  document.querySelectorAll(".quick-action").forEach((button) => {
    button.addEventListener("click", () => applyQuickTemplate(button.dataset.template));
  });
  el("utilityControl").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-value]");
    if (!button) return;
    setUtility(button.dataset.value);
  });
  el("decisionControl").addEventListener("click", (event) => {
    const button = event.target.closest("button[data-value]");
    if (!button) return;
    setDecision(button.dataset.value);
  });
  el("markMissingButton").addEventListener("click", markRemainingRequiredMissing);
  el("occlusionSelect").addEventListener("change", (event) => updateViewField("occlusion_level", event.target.value));
  el("identityAmbiguousCheckbox").addEventListener("change", (event) => updateViewField("identity_ambiguous", event.target.checked));
  el("counterfactualInput").addEventListener("input", (event) => updateViewField("decisive_counterfactual", event.target.value));
  el("rationaleInput").addEventListener("input", (event) => updateViewField("short_rationale", event.target.value));
  el("saveDraftButton").addEventListener("click", () => saveCurrent("draft"));
  el("completeNextButton").addEventListener("click", completeAndNext);
  el("applyPrefillButton").addEventListener("click", applySetupPrefill);
  document.querySelectorAll(".tool-button").forEach((button) => {
    button.addEventListener("click", () => setGeometryTool(button.dataset.tool));
  });
  el("overlayToggle").addEventListener("change", drawGeometry);
  el("undoGeometryButton").addEventListener("click", undoGeometry);
  el("activeImage").addEventListener("load", () => {
    el("imageEmptyState").classList.add("hidden");
    resizeCanvas();
  });
  window.addEventListener("resize", resizeCanvas);
  const canvas = el("annotationCanvas");
  canvas.addEventListener("pointerdown", geometryPointerDown);
  canvas.addEventListener("pointermove", geometryPointerMove);
  canvas.addEventListener("pointerup", geometryPointerUp);
  canvas.addEventListener("pointercancel", geometryPointerUp);
  document.addEventListener("keydown", handleKeyboard);
}

function loadSelection() {
  const setup = activeSetup();
  const view = activeView();
  if (!setup) {
    app.setupDraft = {};
    app.viewDraft = {};
    return;
  }
  const setupAnnotation = app.annotations.get(annotationKey(setup));
  app.setupDraft = setupAnnotation ? deepClone(setupAnnotation.payload) : defaultSetupPayload(setup);
  app.setupRevision = setupAnnotation?.revision || 0;
  app.setupStatus = setupAnnotation?.workflow_status || "draft";
  const viewAnnotation = view ? app.annotations.get(annotationKey(setup, view)) : null;
  app.viewDraft = viewAnnotation ? deepClone(viewAnnotation.payload) : defaultViewPayload(setup);
  app.viewDraft.role_states ||= {};
  for (const role of ontologyList("evidence_roles")) app.viewDraft.role_states[itemId(role)] ||= "unrated";
  app.viewDraft.object_visibility ||= {};
  app.viewDraft.relations ||= {};
  app.viewDraft.boxes ||= [];
  app.viewDraft.keypoints ||= [];
  app.viewRevision = viewAnnotation?.revision || 0;
  app.viewStatus = viewAnnotation?.workflow_status || "draft";
  app.dirtySetup = false;
  app.dirtyView = false;
  app.geometryHistory = [];
  app.draftBox = null;
}

function renderAll() {
  renderProject();
  renderContext();
  renderQueue();
  renderViewStrip();
  renderActiveImage();
  renderInspector();
  renderSetupForm();
  renderProgress();
  renderQualitySummary();
  setWorkflow(app.workflow, false);
}

function renderProject() {
  const project = app.manifest?.project || {};
  el("projectSubtitle").textContent = project.name || "Fixed-lattice robot inspection";
}

function renderContext() {
  const setup = activeSetup();
  if (!setup) return;
  el("setupLabel").textContent = `Setup ${setup.setup_id}`;
  el("splitBadge").textContent = setup.split || "unassigned";
  el("claimText").textContent = setup.claim_text || "Claim text not provided";
  let detail = setup.claim_id || "No claim ID";
  if (!app.blindMode && app.workflow === "view") {
    const truth = app.setupDraft.physical_claim_truth || "unrated";
    const error = app.setupDraft.error_type || "unrated";
    detail += ` · physical truth: ${truth} · ${error}`;
  }
  el("claimId").textContent = detail;
}

function setupQueueStatus(setup) {
  let complete = 0;
  let started = 0;
  let review = 0;
  for (const view of setup.views) {
    const annotation = app.annotations.get(annotationKey(setup, view));
    if (annotation) started += 1;
    if (annotation?.workflow_status === "complete") complete += 1;
    if (annotation?.workflow_status === "review") review += 1;
  }
  if (review) return {name: "review", complete, started};
  if (complete === setup.views.length) return {name: "complete", complete, started};
  if (started) return {name: "in_progress", complete, started};
  return {name: "unstarted", complete, started};
}

function filteredSetupIndexes() {
  const indexes = [];
  app.setups.forEach((setup, index) => {
    const status = setupQueueStatus(setup);
    const haystack = `${setup.setup_id} ${setup.claim_id || ""} ${setup.claim_text || ""}`.toLowerCase();
    if (app.search && !haystack.includes(app.search)) return;
    if (app.filter !== "all" && status.name !== app.filter) return;
    indexes.push(index);
  });
  return indexes;
}

function renderQueue() {
  const container = el("setupQueue");
  const indexes = filteredSetupIndexes();
  el("queueResultCount").textContent = `${indexes.length} setup${indexes.length === 1 ? "" : "s"}`;
  container.innerHTML = indexes.map((index) => {
    const setup = app.setups[index];
    const status = setupQueueStatus(setup);
    const bars = setup.views.map((view) => {
      const annotation = app.annotations.get(annotationKey(setup, view));
      const className = annotation?.workflow_status === "complete" ? "complete" : annotation?.workflow_status === "review" ? "review" : "";
      return `<span class="${className}"></span>`;
    }).join("");
    return `
      <button class="queue-item ${index === app.selectedSetup ? "active" : ""}" data-setup-index="${index}">
        <span class="queue-item-top"><strong>${escapeHtml(setup.setup_id)}</strong><span>${status.complete}/${setup.views.length}</span></span>
        <span class="queue-item-claim">${escapeHtml(setup.claim_text || setup.claim_id || "No claim")}</span>
        <span class="queue-item-bottom"><span class="mini-progress">${bars}</span><span>${status.name.replace("_", " ")}</span></span>
      </button>`;
  }).join("") || `<div class="queue-summary">No setups match the current filter.</div>`;
  container.querySelectorAll("[data-setup-index]").forEach((button) => {
    button.addEventListener("click", () => selectSetup(Number(button.dataset.setupIndex)));
  });
}

function renderViewStrip() {
  const setup = activeSetup();
  const strip = el("viewStrip");
  if (!setup) {
    strip.innerHTML = "";
    return;
  }
  strip.style.gridTemplateColumns = `repeat(${setup.views.length}, minmax(90px, 1fr))`;
  strip.innerHTML = setup.views.map((view, index) => {
    const annotation = app.annotations.get(annotationKey(setup, view));
    const utility = annotation?.payload?.oracle_utility;
    const status = annotation?.workflow_status;
    const stateClass = status === "review" ? "review" : utility !== undefined && utility !== "" ? `u${utility}` : "";
    const stateText = status === "review" ? "!" : utility !== undefined && utility !== "" ? utility : "-";
    return `
      <button class="view-thumb ${index === app.selectedView ? "active" : ""}" data-view-index="${index}" title="Open ${escapeHtml(view.view_id)}">
        <img src="/media?path=${encodeURIComponent(view.image)}" alt="${escapeHtml(view.view_id)}">
        <span class="view-thumb-label">${escapeHtml(view.view_id)}</span>
        <span class="view-thumb-state ${stateClass}">${stateText}</span>
      </button>`;
  }).join("");
  strip.querySelectorAll("[data-view-index]").forEach((button) => {
    button.addEventListener("click", () => selectView(Number(button.dataset.viewIndex)));
  });
}

function renderActiveImage() {
  const setup = activeSetup();
  const view = activeView();
  const image = el("activeImage");
  if (!setup || !view) {
    image.removeAttribute("src");
    el("imageEmptyState").classList.remove("hidden");
    return;
  }
  image.src = `/media?path=${encodeURIComponent(view.image)}`;
  el("imageStatus").textContent = `${setup.setup_id} · ${view.view_id} · ${view.image}`;
  el("activeViewLabel").textContent = view.view_id;
  requestAnimationFrame(resizeCanvas);
}

function renderInspector() {
  const payload = app.viewDraft;
  document.querySelectorAll("#utilityControl button").forEach((button) => {
    button.classList.toggle("active", String(payload.oracle_utility) === button.dataset.value);
  });
  document.querySelectorAll("#decisionControl button").forEach((button) => {
    button.classList.toggle("active", payload.observable_decision === button.dataset.value);
  });
  el("occlusionSelect").value = payload.occlusion_level || "";
  el("identityAmbiguousCheckbox").checked = Boolean(payload.identity_ambiguous);
  el("counterfactualInput").value = payload.decisive_counterfactual || "";
  el("rationaleInput").value = payload.short_rationale || "";
  renderRoles();
  renderObjects();
  renderRelations();
  renderValidation();
  const badge = el("viewWorkflowBadge");
  badge.textContent = app.viewStatus;
  badge.className = `workflow-badge ${app.viewStatus}`;
  drawGeometry();
}

function renderRoles() {
  const setup = activeSetup();
  const required = new Set(setup?.required_roles || []);
  el("roleGrid").innerHTML = ontologyList("evidence_roles").map((role) => {
    const id = itemId(role);
    const state = app.viewDraft.role_states?.[id] || "unrated";
    return `<button class="role-chip ${state} ${required.has(id) ? "required" : ""}" data-role-id="${escapeHtml(id)}"><span>${escapeHtml(itemLabel(role))}</span><small>${state}</small></button>`;
  }).join("");
  el("roleGrid").querySelectorAll("[data-role-id]").forEach((button) => {
    button.addEventListener("click", () => cycleRole(button.dataset.roleId));
  });
}

function renderObjects() {
  el("objectVisibilityGrid").innerHTML = ontologyList("object_classes").map((objectClass) => {
    const id = itemId(objectClass);
    const value = app.viewDraft.object_visibility?.[id] || "";
    return `<label class="compact-row"><span>${escapeHtml(itemLabel(objectClass))}</span><select data-object-id="${escapeHtml(id)}"><option value="" ${value === "" ? "selected" : ""}>Not rated</option><option value="full" ${value === "full" ? "selected" : ""}>Full</option><option value="partial" ${value === "partial" ? "selected" : ""}>Partial</option><option value="absent" ${value === "absent" ? "selected" : ""}>Absent</option></select></label>`;
  }).join("");
  el("objectVisibilityGrid").querySelectorAll("[data-object-id]").forEach((select) => {
    select.addEventListener("change", () => {
      app.viewDraft.object_visibility[select.dataset.objectId] = select.value;
      markViewDirty();
    });
  });
}

function renderRelations() {
  el("relationGrid").innerHTML = ontologyList("relation_types").map((relation) => {
    const id = itemId(relation);
    const value = app.viewDraft.relations?.[id] || "unrated";
    return `<button class="relation-chip ${value}" data-relation-id="${escapeHtml(id)}">${escapeHtml(itemLabel(relation))}: ${value}</button>`;
  }).join("");
  el("relationGrid").querySelectorAll("[data-relation-id]").forEach((button) => {
    button.addEventListener("click", () => cycleRelation(button.dataset.relationId));
  });
}

function fieldSelect(name, label, options, span = "") {
  const value = app.setupDraft[name] || "";
  const optionsHtml = [{value: "", label: "Not rated"}, ...options].map((option) =>
    `<option value="${escapeHtml(option.value)}" ${value === option.value ? "selected" : ""}>${escapeHtml(option.label)}</option>`
  ).join("");
  return `<label class="field-label ${span}">${escapeHtml(label)}<select data-setup-field="${escapeHtml(name)}">${optionsHtml}</select></label>`;
}

function fieldInput(name, label, placeholder = "", span = "") {
  return `<label class="field-label ${span}">${escapeHtml(label)}<input data-setup-field="${escapeHtml(name)}" value="${escapeHtml(app.setupDraft[name] || "")}" placeholder="${escapeHtml(placeholder)}"></label>`;
}

function renderSetupForm() {
  if (!activeSetup()) return;
  const errorOptions = ontologyList("error_types").map((value) => ({value: itemId(value), label: itemLabel(value)}));
  el("setupForm").innerHTML = [
    fieldSelect("assembly_family", "Assembly family", [{value: "A", label: "A"}, {value: "B", label: "B"}, {value: "other", label: "Other"}]),
    fieldInput("active_claim_id", "Active claim ID"),
    fieldSelect("physical_claim_truth", "Physical claim truth", [{value: "true", label: "True"}, {value: "false", label: "False"}]),
    fieldSelect("error_type", "Physical condition", errorOptions),
    fieldInput("target_part_identity", "Target part identity", "Canonical part name"),
    fieldInput("housing_identity", "Housing identity", "Canonical housing name"),
    fieldInput("cover_identity", "Cover identity", "Canonical cover name"),
    fieldSelect("gear_orientation", "Gear orientation", [{value: "correct", label: "Correct"}, {value: "wrong", label: "Wrong"}, {value: "not_applicable", label: "Not applicable"}]),
    fieldSelect("insertion_state", "Insertion state", [{value: "absent", label: "Absent"}, {value: "not_inserted", label: "Not inserted"}, {value: "partial", label: "Partial"}, {value: "full", label: "Full"}, {value: "not_applicable", label: "Not applicable"}]),
    fieldSelect("alignment_state", "Alignment state", [{value: "aligned", label: "Aligned"}, {value: "misaligned", label: "Misaligned"}, {value: "not_applicable", label: "Not applicable"}]),
    fieldSelect("cover_seating_state", "Cover seating", [{value: "unseated", label: "Unseated"}, {value: "partial", label: "Partial / gap"}, {value: "seated", label: "Seated"}, {value: "not_applicable", label: "Not applicable"}]),
    fieldInput("counterfactual_type", "Actual counterfactual / failure", "e.g. wrong orientation", "span-two"),
    `<label class="field-label span-three">Setup notes<textarea data-setup-field="notes" rows="3" placeholder="Record only setup construction facts or uncertainty.">${escapeHtml(app.setupDraft.notes || "")}</textarea></label>`,
    `<div class="span-three inspector-footer"><button id="saveSetupDraftButton" class="command-button secondary">Save draft</button><button id="completeSetupButton" class="command-button primary">Complete setup truth</button></div>`,
  ].join("");
  el("setupForm").querySelectorAll("[data-setup-field]").forEach((control) => {
    control.addEventListener("input", () => {
      app.setupDraft[control.dataset.setupField] = control.value;
      markSetupDirty();
    });
  });
  el("saveSetupDraftButton").addEventListener("click", () => saveSetup("draft"));
  el("completeSetupButton").addEventListener("click", () => saveSetup("complete"));
}

function renderProgress() {
  let total = 0;
  let complete = 0;
  for (const setup of app.setups) {
    for (const view of setup.views) {
      total += 1;
      if (app.annotations.get(annotationKey(setup, view))?.workflow_status === "complete") complete += 1;
    }
  }
  const percent = total ? Math.round((complete / total) * 100) : 0;
  el("progressText").textContent = `${complete} / ${total} views complete`;
  el("progressPercent").textContent = `${percent}%`;
  el("progressFill").style.width = `${percent}%`;
}

function validateSetup(payload = app.setupDraft) {
  const issues = [];
  if (!payload.active_claim_id) issues.push({level: "error", message: "Active claim ID is required."});
  if (!payload.physical_claim_truth) issues.push({level: "error", message: "Physical claim truth is required."});
  if (!payload.error_type) issues.push({level: "error", message: "Physical condition is required."});
  if (payload.physical_claim_truth === "true" && payload.error_type && payload.error_type !== "correct") {
    issues.push({level: "warning", message: "A true claim is paired with a non-correct physical condition."});
  }
  if (payload.physical_claim_truth === "false" && payload.error_type === "correct") {
    issues.push({level: "warning", message: "A false claim is paired with the correct condition."});
  }
  return issues;
}

function validateView(payload = app.viewDraft, setup = activeSetup()) {
  const issues = [];
  const utility = payload.oracle_utility === "" || payload.oracle_utility === undefined ? null : Number(payload.oracle_utility);
  const decision = payload.observable_decision || "";
  if (utility === null || Number.isNaN(utility)) issues.push({level: "error", message: "Oracle utility is required."});
  if (!decision) issues.push({level: "error", message: "Observable decision is required."});
  if (utility !== null && utility < 2 && decision && decision !== "insufficient") {
    issues.push({level: "error", message: "Utility 0/1 must remain insufficient."});
  }
  if (utility === 2 && !["supported", "contradicted"].includes(decision)) {
    issues.push({level: "error", message: "Utility 2 requires supported or contradicted."});
  }
  const roleStates = payload.role_states || {};
  const visible = Object.values(roleStates).filter((value) => value === "visible").length;
  const missing = Object.values(roleStates).filter((value) => value === "missing").length;
  if (utility === 2 && visible === 0) issues.push({level: "error", message: "A decidable view needs at least one visible evidence role."});
  if (utility !== null && utility < 2 && setup?.required_roles?.length && missing === 0) {
    issues.push({level: "warning", message: "Rate at least one missing required evidence role."});
  }
  const setupAnnotation = setup ? app.annotations.get(annotationKey(setup)) : null;
  const truth = setupAnnotation?.payload?.physical_claim_truth;
  if (truth === "true" && decision === "contradicted") {
    issues.push({level: "warning", message: "View decision conflicts with physical claim truth."});
  }
  if (truth === "false" && decision === "supported") {
    issues.push({level: "warning", message: "View decision conflicts with physical claim truth."});
  }
  for (const box of payload.boxes || []) {
    if (![box.x, box.y, box.w, box.h].every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) {
      issues.push({level: "error", message: "A bounding box has invalid normalized coordinates."});
      break;
    }
  }
  return issues;
}

function renderValidation() {
  const issues = validateView();
  const panel = el("validationPanel");
  const highest = issues.some((issue) => issue.level === "error") ? "error" : issues.some((issue) => issue.level === "warning") ? "warning" : "ok";
  panel.className = `validation-panel ${highest === "ok" ? "" : highest}`;
  el("validationList").innerHTML = issues.length
    ? issues.map((issue) => `<li><strong>${escapeHtml(issue.level)}</strong>: ${escapeHtml(issue.message)}</li>`).join("")
    : "<li>Current annotation is internally consistent.</li>";
}

function collectQualityIssues() {
  const issues = [];
  for (let setupIndex = 0; setupIndex < app.setups.length; setupIndex += 1) {
    const setup = app.setups[setupIndex];
    const setupAnnotation = app.annotations.get(annotationKey(setup));
    if (setupAnnotation?.workflow_status === "complete") {
      for (const issue of validateSetup(setupAnnotation.payload)) {
        issues.push({setupIndex, viewIndex: null, ...issue});
      }
    }
    setup.views.forEach((view, viewIndex) => {
      const annotation = app.annotations.get(annotationKey(setup, view));
      if (!annotation || annotation.workflow_status === "draft") return;
      const savedSetupDraft = app.setupDraft;
      const viewIssues = validateView(annotation.payload, setup);
      app.setupDraft = savedSetupDraft;
      for (const issue of viewIssues) issues.push({setupIndex, viewIndex, ...issue});
    });
  }
  return issues;
}

function renderQualitySummary() {
  const issues = collectQualityIssues();
  el("qualityCount").textContent = `${issues.length} issue${issues.length === 1 ? "" : "s"}`;
}

function openQualityDialog() {
  const issues = collectQualityIssues();
  el("qualityIssueList").innerHTML = issues.length ? issues.map((issue, index) => {
    const setup = app.setups[issue.setupIndex];
    const view = issue.viewIndex === null ? null : setup.views[issue.viewIndex];
    return `<button class="quality-item" data-quality-index="${index}"><strong>${escapeHtml(issue.level)}</strong><span>${escapeHtml(issue.message)}</span><span>${escapeHtml(setup.setup_id)}${view ? ` / ${escapeHtml(view.view_id)}` : ""}</span></button>`;
  }).join("") : `<div class="queue-summary">No saved annotation issues.</div>`;
  el("qualityIssueList").querySelectorAll("[data-quality-index]").forEach((button) => {
    button.addEventListener("click", () => {
      const issue = issues[Number(button.dataset.qualityIndex)];
      selectSetup(issue.setupIndex, issue.viewIndex ?? 0);
      setWorkflow(issue.viewIndex === null ? "setup" : "view");
      el("qualityDialog").close();
    });
  });
  el("qualityDialog").showModal();
}

function setWorkflow(workflow, rerender = true) {
  app.workflow = workflow;
  el("viewWorkflow").classList.toggle("hidden", workflow !== "view");
  el("inspectorPanel").classList.toggle("hidden", workflow !== "view");
  el("setupWorkflow").classList.toggle("hidden", workflow !== "setup");
  el("viewTab").classList.toggle("active", workflow === "view");
  el("setupTab").classList.toggle("active", workflow === "setup");
  el("viewTab").setAttribute("aria-selected", String(workflow === "view"));
  el("setupTab").setAttribute("aria-selected", String(workflow === "setup"));
  if (rerender) renderContext();
}

async function selectSetup(index, viewIndex = 0) {
  await flushAutosave();
  app.selectedSetup = Math.max(0, Math.min(index, app.setups.length - 1));
  app.selectedView = Math.max(0, Math.min(viewIndex, (activeSetup()?.views?.length || 1) - 1));
  loadSelection();
  renderAll();
}

async function selectView(index) {
  await flushAutosave();
  const count = activeSetup()?.views?.length || 0;
  if (!count) return;
  app.selectedView = (index + count) % count;
  loadSelection();
  renderContext();
  renderQueue();
  renderViewStrip();
  renderActiveImage();
  renderInspector();
}

function moveSetup(delta) {
  if (!app.setups.length) return;
  selectSetup((app.selectedSetup + delta + app.setups.length) % app.setups.length);
}

function moveView(delta) {
  const count = activeSetup()?.views?.length || 0;
  if (count) selectView((app.selectedView + delta + count) % count);
}

function moveToNextIncomplete() {
  const total = app.setups.reduce((sum, setup) => sum + setup.views.length, 0);
  let setupIndex = app.selectedSetup;
  let viewIndex = app.selectedView;
  for (let step = 0; step < total; step += 1) {
    viewIndex += 1;
    if (viewIndex >= app.setups[setupIndex].views.length) {
      setupIndex = (setupIndex + 1) % app.setups.length;
      viewIndex = 0;
    }
    const setup = app.setups[setupIndex];
    const annotation = app.annotations.get(annotationKey(setup, setup.views[viewIndex]));
    if (annotation?.workflow_status !== "complete") {
      selectSetup(setupIndex, viewIndex);
      return;
    }
  }
  showToast("All views are complete.", "success");
}

function updateViewField(field, value) {
  app.viewDraft[field] = value;
  markViewDirty();
}

function setUtility(value) {
  app.viewDraft.oracle_utility = String(value);
  if (Number(value) < 2) app.viewDraft.observable_decision = "insufficient";
  renderInspector();
  markViewDirty();
}

function setDecision(value) {
  app.viewDraft.observable_decision = value;
  if (value === "insufficient" && Number(app.viewDraft.oracle_utility) === 2) app.viewDraft.oracle_utility = "1";
  if (["supported", "contradicted"].includes(value)) app.viewDraft.oracle_utility = "2";
  renderInspector();
  markViewDirty();
}

function applyQuickTemplate(template) {
  if (template === "none") {
    app.viewDraft.oracle_utility = "0";
    app.viewDraft.observable_decision = "insufficient";
    for (const role of activeSetup()?.required_roles || []) app.viewDraft.role_states[role] = "missing";
  } else if (template === "partial") {
    app.viewDraft.oracle_utility = "1";
    app.viewDraft.observable_decision = "insufficient";
  } else if (template === "supported") {
    app.viewDraft.oracle_utility = "2";
    app.viewDraft.observable_decision = "supported";
  } else if (template === "contradicted") {
    app.viewDraft.oracle_utility = "2";
    app.viewDraft.observable_decision = "contradicted";
  }
  renderInspector();
  markViewDirty();
}

function cycleRole(roleId) {
  const order = ["unrated", "visible", "missing"];
  const current = app.viewDraft.role_states[roleId] || "unrated";
  app.viewDraft.role_states[roleId] = order[(order.indexOf(current) + 1) % order.length];
  renderRoles();
  renderValidation();
  markViewDirty();
}

function markRemainingRequiredMissing() {
  for (const role of activeSetup()?.required_roles || []) {
    if ((app.viewDraft.role_states[role] || "unrated") === "unrated") app.viewDraft.role_states[role] = "missing";
  }
  renderRoles();
  renderValidation();
  markViewDirty();
}

function cycleRelation(relationId) {
  const order = ["unrated", "true", "false", "unclear"];
  const current = app.viewDraft.relations[relationId] || "unrated";
  app.viewDraft.relations[relationId] = order[(order.indexOf(current) + 1) % order.length];
  renderRelations();
  markViewDirty();
}

function applySetupPrefill() {
  const prefill = activeSetup()?.setup_prefill || {};
  if (!Object.keys(prefill).length) {
    showToast("This setup has no metadata prefill.");
    return;
  }
  for (const [key, value] of Object.entries(prefill)) {
    if (key in app.setupDraft && value !== "") app.setupDraft[key] = value;
  }
  renderSetupForm();
  markSetupDirty();
  showToast("Metadata prefill applied. Review before completing.", "success");
}

function markViewDirty() {
  app.dirtyView = true;
  app.viewStatus = app.viewStatus === "complete" ? "draft" : app.viewStatus;
  setSaveState("Unsaved view changes", "saving");
  renderValidation();
  scheduleAutosave();
}

function markSetupDirty() {
  app.dirtySetup = true;
  app.setupStatus = app.setupStatus === "complete" ? "draft" : app.setupStatus;
  setSaveState("Unsaved setup changes", "saving");
  scheduleAutosave();
}

function scheduleAutosave() {
  clearTimeout(app.saveTimer);
  if (!app.annotator || (!app.dirtyView && !app.dirtySetup)) return;
  app.saveTimer = setTimeout(() => {
    if (app.workflow === "setup" && app.dirtySetup) saveSetup("draft", true);
    else if (app.dirtyView) saveCurrent("draft", true);
  }, 700);
}

async function flushAutosave() {
  clearTimeout(app.saveTimer);
  if (!app.annotator) return;
  if (app.dirtySetup) await saveSetup("draft", true);
  if (app.dirtyView) await saveCurrent("draft", true);
}

function ensureAnnotator() {
  if (app.annotator) return true;
  el("annotatorInput").focus();
  showToast("Enter an annotator ID before saving.", "error");
  return false;
}

async function saveCurrent(status = "draft", quiet = false) {
  const setup = activeSetup();
  const view = activeView();
  if (!setup || !view || !ensureAnnotator()) return false;
  const issues = validateView();
  if (status === "complete" && issues.some((issue) => issue.level === "error")) {
    showToast("Resolve the required quality checks before completing.", "error");
    renderValidation();
    return false;
  }
  setSaveState("Saving view...", "saving");
  try {
    const response = await fetch("/api/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        entity_type: "view",
        setup_id: setup.setup_id,
        view_id: view.view_id,
        claim_id: setup.claim_id || "",
        payload: app.viewDraft,
        workflow_status: status,
        annotator: app.annotator,
        expected_revision: app.viewRevision,
      }),
    });
    const result = await response.json();
    if (response.status === 409) throw new Error("Revision conflict: reload before overwriting another annotation.");
    if (!response.ok) throw new Error(result.error || `Save failed (${response.status})`);
    const annotation = result.annotation;
    app.annotations.set(annotation.entity_key, annotation);
    app.viewRevision = annotation.revision;
    app.viewStatus = annotation.workflow_status;
    app.dirtyView = false;
    setSaveState(`Saved ${new Date().toLocaleTimeString()}`, "saved");
    renderQueue();
    renderViewStrip();
    renderProgress();
    renderQualitySummary();
    renderInspector();
    if (!quiet) showToast(status === "complete" ? "View completed." : "Draft saved.", "success");
    return true;
  } catch (error) {
    setSaveState(error.message, "error");
    if (!quiet) showToast(error.message, "error", 7000);
    return false;
  }
}

async function saveSetup(status = "draft", quiet = false) {
  const setup = activeSetup();
  if (!setup || !ensureAnnotator()) return false;
  const issues = validateSetup();
  if (status === "complete" && issues.some((issue) => issue.level === "error")) {
    showToast(issues.map((issue) => issue.message).join(" "), "error", 7000);
    return false;
  }
  setSaveState("Saving setup...", "saving");
  try {
    const response = await fetch("/api/save", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        entity_type: "setup",
        setup_id: setup.setup_id,
        payload: app.setupDraft,
        workflow_status: status,
        annotator: app.annotator,
        expected_revision: app.setupRevision,
      }),
    });
    const result = await response.json();
    if (response.status === 409) throw new Error("Revision conflict: reload before overwriting another annotation.");
    if (!response.ok) throw new Error(result.error || `Save failed (${response.status})`);
    const annotation = result.annotation;
    app.annotations.set(annotation.entity_key, annotation);
    app.setupRevision = annotation.revision;
    app.setupStatus = annotation.workflow_status;
    app.dirtySetup = false;
    setSaveState(`Saved ${new Date().toLocaleTimeString()}`, "saved");
    renderContext();
    renderQualitySummary();
    if (!quiet) showToast(status === "complete" ? "Setup truth completed." : "Setup draft saved.", "success");
    return true;
  } catch (error) {
    setSaveState(error.message, "error");
    if (!quiet) showToast(error.message, "error", 7000);
    return false;
  }
}

async function completeAndNext() {
  const saved = await saveCurrent("complete");
  if (saved) moveToNextIncomplete();
}

function setSaveState(message, className = "") {
  el("saveState").textContent = message;
  el("saveState").className = `save-state ${className}`;
}

function populateGeometryLabels() {
  const select = el("geometryLabel");
  const objectOptions = ontologyList("object_classes").map((item) => `<option value="box:${escapeHtml(itemId(item))}">Box · ${escapeHtml(itemLabel(item))}</option>`).join("");
  const pointOptions = ontologyList("keypoint_types").map((item) => `<option value="point:${escapeHtml(itemId(item))}">Point · ${escapeHtml(itemLabel(item))}</option>`).join("");
  select.innerHTML = `<optgroup label="Bounding boxes">${objectOptions}</optgroup><optgroup label="Keypoints">${pointOptions}</optgroup>`;
}

function setGeometryTool(tool) {
  app.geometryTool = tool;
  el("imageStage").dataset.tool = tool;
  document.querySelectorAll(".tool-button").forEach((button) => button.classList.toggle("active", button.dataset.tool === tool));
  const select = el("geometryLabel");
  if (tool === "bbox" && !select.value.startsWith("box:")) {
    const first = [...select.options].find((option) => option.value.startsWith("box:"));
    if (first) select.value = first.value;
  }
  if (tool === "point" && !select.value.startsWith("point:")) {
    const first = [...select.options].find((option) => option.value.startsWith("point:"));
    if (first) select.value = first.value;
  }
}

function imageBounds() {
  const stageRect = el("imageStage").getBoundingClientRect();
  const imageRect = el("activeImage").getBoundingClientRect();
  return {
    x: imageRect.left - stageRect.left,
    y: imageRect.top - stageRect.top,
    w: imageRect.width,
    h: imageRect.height,
  };
}

function normalizedPoint(event) {
  const rect = el("annotationCanvas").getBoundingClientRect();
  const bounds = imageBounds();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const nx = (x - bounds.x) / bounds.w;
  const ny = (y - bounds.y) / bounds.h;
  if (nx < 0 || ny < 0 || nx > 1 || ny > 1) return null;
  return {x: nx, y: ny};
}

function pushGeometryHistory() {
  app.geometryHistory.push({boxes: deepClone(app.viewDraft.boxes || []), keypoints: deepClone(app.viewDraft.keypoints || [])});
  if (app.geometryHistory.length > 30) app.geometryHistory.shift();
}

function geometryPointerDown(event) {
  if (!["bbox", "point", "erase"].includes(app.geometryTool)) return;
  const point = normalizedPoint(event);
  if (!point) return;
  event.currentTarget.setPointerCapture(event.pointerId);
  if (app.geometryTool === "bbox") {
    pushGeometryHistory();
    app.draftBox = {start: point, end: point};
  } else if (app.geometryTool === "point") {
    pushGeometryHistory();
    const selected = el("geometryLabel").value;
    const label = selected.startsWith("point:") ? selected.slice(6) : (ontologyList("keypoint_types")[0] ? itemId(ontologyList("keypoint_types")[0]) : "point");
    app.viewDraft.keypoints.push({label, x: point.x, y: point.y});
    markViewDirty();
    drawGeometry();
  } else {
    eraseNearest(point);
  }
}

function geometryPointerMove(event) {
  if (!app.draftBox) return;
  const point = normalizedPoint(event);
  if (!point) return;
  app.draftBox.end = point;
  drawGeometry();
}

function geometryPointerUp(event) {
  if (!app.draftBox) return;
  const point = normalizedPoint(event) || app.draftBox.end;
  const x = Math.min(app.draftBox.start.x, point.x);
  const y = Math.min(app.draftBox.start.y, point.y);
  const w = Math.abs(point.x - app.draftBox.start.x);
  const h = Math.abs(point.y - app.draftBox.start.y);
  if (w > 0.005 && h > 0.005) {
    const selected = el("geometryLabel").value;
    const label = selected.startsWith("box:") ? selected.slice(4) : (ontologyList("object_classes")[0] ? itemId(ontologyList("object_classes")[0]) : "object");
    app.viewDraft.boxes.push({label, x, y, w, h});
    markViewDirty();
  }
  app.draftBox = null;
  drawGeometry();
}

function eraseNearest(point) {
  const boxes = app.viewDraft.boxes || [];
  const points = app.viewDraft.keypoints || [];
  let best = {type: null, index: -1, distance: Infinity};
  boxes.forEach((box, index) => {
    const inside = point.x >= box.x && point.x <= box.x + box.w && point.y >= box.y && point.y <= box.y + box.h;
    const distance = inside ? 0 : Math.hypot(point.x - (box.x + box.w / 2), point.y - (box.y + box.h / 2));
    if (distance < best.distance) best = {type: "box", index, distance};
  });
  points.forEach((keypoint, index) => {
    const distance = Math.hypot(point.x - keypoint.x, point.y - keypoint.y);
    if (distance < best.distance) best = {type: "point", index, distance};
  });
  if (best.index < 0 || best.distance > 0.08) return;
  pushGeometryHistory();
  if (best.type === "box") boxes.splice(best.index, 1);
  else points.splice(best.index, 1);
  markViewDirty();
  drawGeometry();
}

function undoGeometry() {
  const previous = app.geometryHistory.pop();
  if (!previous) return;
  app.viewDraft.boxes = previous.boxes;
  app.viewDraft.keypoints = previous.keypoints;
  markViewDirty();
  drawGeometry();
}

function resizeCanvas() {
  const canvas = el("annotationCanvas");
  const stage = el("imageStage");
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(stage.clientWidth * ratio));
  canvas.height = Math.max(1, Math.round(stage.clientHeight * ratio));
  canvas.style.width = `${stage.clientWidth}px`;
  canvas.style.height = `${stage.clientHeight}px`;
  drawGeometry();
}

function drawGeometry() {
  const canvas = el("annotationCanvas");
  const context = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, canvas.width / ratio, canvas.height / ratio);
  if (!el("overlayToggle").checked || !activeView()) return;
  const bounds = imageBounds();
  const sourceBoxes = activeView()?.source_boxes || [];
  context.font = "12px Segoe UI, Arial";
  context.lineWidth = 2;
  for (const box of sourceBoxes) drawBox(context, bounds, box, "#aab5c1", true);
  for (const box of app.viewDraft.boxes || []) drawBox(context, bounds, box, "#45a6ff", false);
  for (const point of app.viewDraft.keypoints || []) drawPoint(context, bounds, point, "#ffd166");
  if (app.draftBox) {
    const box = {
      label: "new box",
      x: Math.min(app.draftBox.start.x, app.draftBox.end.x),
      y: Math.min(app.draftBox.start.y, app.draftBox.end.y),
      w: Math.abs(app.draftBox.end.x - app.draftBox.start.x),
      h: Math.abs(app.draftBox.end.y - app.draftBox.start.y),
    };
    drawBox(context, bounds, box, "#ffffff", false);
  }
}

function drawBox(context, bounds, box, color, dashed) {
  const x = bounds.x + box.x * bounds.w;
  const y = bounds.y + box.y * bounds.h;
  const w = box.w * bounds.w;
  const h = box.h * bounds.h;
  context.strokeStyle = color;
  context.fillStyle = color;
  context.setLineDash(dashed ? [5, 4] : []);
  context.strokeRect(x, y, w, h);
  context.setLineDash([]);
  const label = String(box.label || "object");
  const width = context.measureText(label).width + 8;
  context.fillRect(x, Math.max(0, y - 19), width, 18);
  context.fillStyle = "#111820";
  context.fillText(label, x + 4, Math.max(13, y - 6));
}

function drawPoint(context, bounds, point, color) {
  const x = bounds.x + point.x * bounds.w;
  const y = bounds.y + point.y * bounds.h;
  context.fillStyle = color;
  context.strokeStyle = "#111820";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(x, y, 6, 0, Math.PI * 2);
  context.fill();
  context.stroke();
  context.fillStyle = "#111820";
  context.fillRect(x + 8, y - 11, context.measureText(point.label || "point").width + 8, 18);
  context.fillStyle = "#fff";
  context.fillText(point.label || "point", x + 12, y + 2);
}

function handleKeyboard(event) {
  const target = event.target;
  const editing = ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    app.workflow === "setup" ? saveSetup("draft") : saveCurrent("draft");
    return;
  }
  if (editing || event.ctrlKey || event.metaKey || event.altKey) return;
  if (event.key === "[") moveView(-1);
  else if (event.key === "]") moveView(1);
  else if (event.key.toLowerCase() === "j") moveSetup(1);
  else if (event.key.toLowerCase() === "k") moveSetup(-1);
  else if (event.key === "0") applyQuickTemplate("none");
  else if (event.key === "1") applyQuickTemplate("partial");
  else if (event.key === "2") setUtility("2");
  else if (event.key.toLowerCase() === "s") setDecision("supported");
  else if (event.key.toLowerCase() === "c") setDecision("contradicted");
  else if (event.key.toLowerCase() === "i") setDecision("insufficient");
  else if (event.key.toLowerCase() === "b") setGeometryTool("bbox");
  else if (event.key.toLowerCase() === "p") setGeometryTool("point");
  else if (event.key.toLowerCase() === "e") setGeometryTool("erase");
  else if (event.key === "Enter") completeAndNext();
}

function showToast(message, type = "", duration = 3200) {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  el("toastRegion").appendChild(toast);
  setTimeout(() => toast.remove(), duration);
}

bootstrap();
