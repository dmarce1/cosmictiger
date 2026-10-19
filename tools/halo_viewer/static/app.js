"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const filterNames = ["halo_id", "kind", "mvir_min", "mvir_max", "rvir_min", "rvir_max", "vcircMax_min", "vcircMax_max", "lambda_min", "lambda_max", "x_min", "x_max", "y_min", "y_max", "z_min", "z_max"];
  const numericFilters = filterNames.filter((name) => name !== "halo_id" && name !== "kind");
  const pageSize = 50;
  const formatter = new Intl.NumberFormat("en-US");
  const state = {catalogs: [], catalog: null, filters: {}, sort: "mvir", order: "desc", page: 0, view: "xy", selected: null, detail: null, total: 0, tableItems: [], points: [], bounds: null, plotted: [], tableController: null, pointsController: null, detailController: null, tableSequence: 0, pointsSequence: 0, detailSequence: 0};
  const canvas = $("spatial-canvas");
  const context = canvas.getContext("2d");
  let plotSize = {width: 0, height: 0};
  const propertyGroups = [
    ["Position & mass", ["x", "y", "z", "zValue", "mvir", "m200", "m500", "m2500"]],
    ["Radii & structure", ["rvir", "rmax", "rvmax", "rs", "rkly", "sigX", "deltaX", "lambda", "lambdaB", "toW"]],
    ["Velocities & dynamics", ["vcircMax", "sigV", "sigVb", "deltaV", "vxc", "vyc", "vzc", "vxb", "vyb", "vzb", "e", "jx", "jy", "jz"]],
    ["Shape", ["boa", "coa", "ax", "ay", "az", "boa500", "coa500", "ax500", "ay500", "az500"]],
    ["Identifiers & provenance", ["halo_id", "host_id", "fof_id", "local_halo_id", "rank", "ordinal", "sys", "pid"]]
  ];
  const labels = {x:"X",y:"Y",z:"Z",zValue:"Redshift",mvir:"Virial mass",m200:"M200c",m500:"M500c",m2500:"M2500c",rvir:"Virial radius",rmax:"Maximum radius",rvmax:"Radius at Vmax",rs:"NFW scale radius",rkly:"Klypin scale radius",sigX:"Position dispersion",deltaX:"Position offset",lambda:"Spin λ",lambdaB:"Bullock spin λB",toW:"T / |U|",vcircMax:"Maximum circular velocity",sigV:"Velocity dispersion",sigVb:"Bulk velocity dispersion",deltaV:"Velocity offset",vxc:"Core velocity X",vyc:"Core velocity Y",vzc:"Core velocity Z",vxb:"Bulk velocity X",vyb:"Bulk velocity Y",vzb:"Bulk velocity Z",e:"Energy",jx:"Angular momentum X",jy:"Angular momentum Y",jz:"Angular momentum Z",boa:"Axis ratio b/a",coa:"Axis ratio c/a",ax:"Shape axis X",ay:"Shape axis Y",az:"Shape axis Z",boa500:"Axis ratio b/a (500c)",coa500:"Axis ratio c/a (500c)",ax500:"Shape axis X (500c)",ay500:"Shape axis Y (500c)",az500:"Shape axis Z (500c)",halo_id:"Persistent halo ID",host_id:"Host halo ID",fof_id:"FOF ID",local_halo_id:"Local halo index",rank:"Source rank",ordinal:"Record ordinal",sys:"Inclusive system flag",pid:"Local host index"};
  const units = {};
  for (const name of ["x", "y", "z"]) units[name] = "comoving Mpc/h";
  for (const name of ["mvir", "m200", "m500", "m2500"]) units[name] = "M☉/h";
  for (const name of ["rvir", "rmax", "rvmax", "rs", "rkly", "sigX", "deltaX", "ax", "ay", "az", "ax500", "ay500", "az500"]) units[name] = "comoving kpc/h";
  for (const name of ["vcircMax", "sigV", "sigVb", "deltaV", "vxc", "vyc", "vzc", "vxb", "vyb", "vzb"]) units[name] = "km/s";
  for (const name of ["lambda", "lambdaB", "toW", "boa", "coa", "boa500", "coa500", "zValue"]) units[name] = "dimensionless";
  for (const name of ["jx", "jy", "jz"]) units[name] = "(M☉/h)(Mpc/h)(km/s) · physical";
  units.e = "(M☉/h)(km/s)²";

  function node(tag, className, text) {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
  }
  function displayNumber(value, digits = 3) {
    if (value === null || value === undefined || value === "") return "—";
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    if (number === 0) return "0";
    if (Math.abs(number) >= 100000 || Math.abs(number) < 0.001) return number.toExponential(digits - 1);
    return new Intl.NumberFormat("en-US", {maximumSignificantDigits: digits}).format(number);
  }
  function count(value) { return formatter.format(value || 0); }
  function typeBadge(halo) { return node("span", "type-label" + (halo.host_id !== null && halo.host_id !== undefined ? " subhalo" : ""), halo.host_id !== null && halo.host_id !== undefined ? "Subhalo" : "Distinct"); }
  function matchesFilters(halo) {
    if (state.filters.halo_id && String(halo.halo_id) !== state.filters.halo_id) return false;
    if (state.filters.kind === "distinct" && halo.host_id != null) return false;
    if (state.filters.kind === "subhalo" && halo.host_id == null) return false;
    for (const name of numericFilters) {
      if (state.filters[name] === undefined) continue;
      const field = name.slice(0, -4); const limit = Number(state.filters[name]); const value = Number(halo[field]);
      if (!Number.isFinite(value) || (name.endsWith("_min") ? value < limit : value > limit)) return false;
    }
    return true;
  }
  function activeQuery() {
    const query = new URLSearchParams({catalog: String(state.catalog.id)});
    for (const name of filterNames) if (state.filters[name] !== undefined && state.filters[name] !== "" && !(name === "kind" && state.filters[name] === "all")) query.set(name, state.filters[name]);
    return query;
  }
  function writeUrl(replace = false) {
    if (!state.catalog) return;
    const query = activeQuery();
    query.set("view", state.view);
    if (state.sort !== "mvir") query.set("sort", state.sort);
    if (state.order !== "desc") query.set("order", state.order);
    if (state.page) query.set("page", String(state.page + 1));
    if (state.selected) query.set("selected", state.selected);
    const url = `${location.pathname}?${query}`;
    if (url !== `${location.pathname}${location.search}`) history[replace ? "replaceState" : "pushState"]({}, "", url);
  }
  function readUrl() {
    const query = new URLSearchParams(location.search);
    state.catalog = state.catalogs.find((catalog) => String(catalog.id) === query.get("catalog")) || state.catalogs[0];
    state.filters = {};
    for (const name of filterNames) {
      const value = query.get(name);
      if (value !== null && value !== "") state.filters[name] = value;
    }
    if (!["all", "distinct", "subhalo"].includes(state.filters.kind)) delete state.filters.kind;
    state.sort = ["halo_id", "mvir", "rvir", "vcircMax"].includes(query.get("sort")) ? query.get("sort") : "mvir";
    state.order = query.get("order") === "asc" ? "asc" : "desc";
    state.page = Math.max(0, Math.floor(Number(query.get("page")) || 1) - 1);
    state.view = ["xy", "xz", "yz"].includes(query.get("view")) ? query.get("view") : "xy";
    state.selected = /^\d+$/.test(query.get("selected") || "") ? query.get("selected") : null;
    state.detail = null;
    $("catalog-select").value = String(state.catalog.id);
    for (const name of filterNames) $("filter-form").elements[name].value = state.filters[name] || (name === "kind" ? "all" : "");
    if (numericFilters.some((name) => /^[xyz]_/.test(name) && state.filters[name])) $("filter-form").querySelector(".region-filter").open = true;
    updateFilterCount(); updateCatalog(); updateProjection();
  }
  function updateFilterCount() {
    $("filter-count").textContent = Object.entries(state.filters).filter(([name, value]) => value !== "" && !(name === "kind" && value === "all")).length;
  }
  function updateCatalog() {
    const catalog = state.catalog;
    $("catalog-title").textContent = catalog.name;
    const snapshot = catalog.snapshot === null || catalog.snapshot === undefined ? "Snapshot" : `Snapshot ${catalog.snapshot}`;
    $("catalog-subtitle").textContent = `${snapshot} · z = ${displayNumber(catalog.redshift, 5)} · a = ${displayNumber(catalog.scale_factor, 5)}`;
    const boxLabel = catalog.box_length_mpc != null ? `\nBox ${displayNumber(catalog.box_length_mpc, 5)} comoving Mpc` : "";
    const hubbleLabel = catalog.hubble != null ? `h = ${displayNumber(catalog.hubble, 5)}` : "h unspecified";
    $("catalog-context").textContent = `z ${displayNumber(catalog.redshift, 5)}  /  a ${displayNumber(catalog.scale_factor, 5)}${boxLabel}\n${hubbleLabel}`;
    $("catalog-context").style.whiteSpace = "pre-line";
    $("catalog-total").textContent = count(catalog.halo_count);
    $("catalog-hosts").textContent = count(catalog.host_count);
    $("catalog-subhalos").textContent = count(catalog.subhalo_count);
    $("catalog-title").title = catalog.source_path || "";
  }
  function showError(message) { $("app-error").textContent = message; $("app-error").hidden = !message; }
  async function fetchJson(url, signal) {
    const response = await fetch(url, {signal, headers: {Accept: "application/json"}});
    let result;
    try { result = await response.json(); } catch (_) { throw new Error(`The server returned an unreadable response (${response.status}).`); }
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status}).`);
    return result;
  }
  function updateExport() {
    const query = activeQuery(); query.set("sort", state.sort); query.set("order", state.order);
    $("export-link").href = `/api/export?${query}`;
    $("export-link").title = "Download all matching halos as CSV, including halos beyond this page or the displayed spatial sample.";
    $("metadata-link").href = `/api/export_metadata?${query}`;
    $("metadata-link").title = "Download source hashes, units, active filters, and sort order for this complete selection.";
  }
  async function loadCatalogs() {
    showError("");
    try {
      const result = await fetchJson("/api/catalogs");
      state.catalogs = result.catalogs;
      $("empty-workspace").hidden = !!result.catalogs.length;
      $("app-layout").hidden = !result.catalogs.length;
      if (!result.catalogs.length) return;
      $("catalog-select").replaceChildren(...result.catalogs.map((catalog) => {
        const option = node("option", "", catalog.name); option.value = catalog.id; return option;
      }));
      readUrl(); writeUrl(true); loadSelection(); loadData();
    } catch (error) { showError(`Unable to load catalogs: ${error.message}`); }
  }
  function loadData() {
    loadTable(); loadPoints();
  }
  async function loadTable() {
    updateExport();
    state.tableController?.abort();
    state.tableController = new AbortController();
    const sequence = ++state.tableSequence;
    const query = activeQuery();
    query.set("sort", state.sort); query.set("order", state.order); query.set("limit", pageSize); query.set("offset", state.page * pageSize);
    $("table-status").className = "table-status";
    $("table-status").textContent = "Loading catalog selection…";
    $("halo-table").setAttribute("aria-busy", "true");
    $("previous-page").disabled = true; $("next-page").disabled = true;
    try {
      const result = await fetchJson(`/api/halos?${query}`, state.tableController.signal);
      if (sequence !== state.tableSequence) return;
      state.total = result.total;
      if (result.total && state.page * pageSize >= result.total) {
        state.page = Math.floor((result.total - 1) / pageSize); writeUrl(true); loadTable(); return;
      }
      state.tableItems = result.items;
      $("table-status").textContent = "";
      renderTable();
    } catch (error) {
      if (error.name === "AbortError" || sequence !== state.tableSequence) return;
      state.tableItems = []; state.total = 0; renderTable();
      $("table-status").className = "table-status error";
      $("table-status").textContent = `Unable to load halos: ${error.message}`;
    } finally { if (sequence === state.tableSequence) $("halo-table").setAttribute("aria-busy", "false"); }
  }
  function renderTable() {
    const body = $("halo-table-body");
    body.replaceChildren();
    $("match-count").textContent = count(state.total);
    if (!state.tableItems.length) {
      const row = node("tr"); const cell = node("td", "empty-cell", "No halos match these filters. Try a wider mass range or spatial region."); cell.colSpan = 5; row.append(cell); body.append(row);
    }
    for (const halo of state.tableItems) {
      const row = node("tr", String(halo.halo_id) === state.selected ? "selected" : "");
      row.tabIndex = 0; row.dataset.haloId = String(halo.halo_id);
      row.setAttribute("aria-label", `Halo ${halo.halo_id}, ${halo.host_id == null ? "distinct halo" : "subhalo"}, virial mass ${displayNumber(halo.mvir)} solar masses per h. Press Enter to inspect.`);
      row.setAttribute("aria-selected", String(String(halo.halo_id) === state.selected));
      row.append(node("td", "", String(halo.halo_id)));
      const type = node("td"); type.append(typeBadge(halo)); row.append(type);
      for (const name of ["mvir", "rvir", "vcircMax"]) row.append(node("td", "", displayNumber(halo[name], 4)));
      row.addEventListener("click", () => selectHalo(String(halo.halo_id)));
      row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectHalo(String(halo.halo_id)); } });
      body.append(row);
    }
    const start = state.total ? state.page * pageSize + 1 : 0;
    const end = Math.min(state.total, (state.page + 1) * pageSize);
    $("page-info").textContent = `${count(start)}–${count(end)} of ${count(state.total)} matching`;
    $("page-number").textContent = `${state.total ? state.page + 1 : 0} / ${Math.max(1, Math.ceil(state.total / pageSize))}`;
    $("previous-page").disabled = state.page === 0;
    $("next-page").disabled = end >= state.total;
    for (const button of document.querySelectorAll("[data-sort]")) {
      const active = button.dataset.sort === state.sort;
      button.closest("th").setAttribute("aria-sort", active ? (state.order === "asc" ? "ascending" : "descending") : "none");
      button.querySelector("span").textContent = active ? (state.order === "asc" ? "↑" : "↓") : "↕";
    }
  }
  async function loadPoints() {
    state.pointsController?.abort(); state.pointsController = new AbortController();
    const sequence = ++state.pointsSequence;
    const query = activeQuery(); query.set("limit", "20000");
    state.points = []; state.bounds = null; drawPlot();
    $("point-count").textContent = "Loading spatial distribution…"; $("plot-empty").hidden = true;
    try {
      const result = await fetchJson(`/api/points?${query}`, state.pointsController.signal);
      if (sequence !== state.pointsSequence) return;
      state.points = result.items; state.bounds = result.bounds;
      $("point-count").textContent = result.sampled ? `${count(result.items.length)} sampled points · ${count(result.total)} matching halos` : `${count(result.items.length)} points · ${count(result.total)} matching halos`;
      $("plot-empty").hidden = result.total !== 0;
      drawPlot();
    } catch (error) {
      if (error.name === "AbortError" || sequence !== state.pointsSequence) return;
      $("point-count").textContent = `Spatial view unavailable: ${error.message}`;
    }
  }
  function selectHalo(id) {
    state.selected = id; state.detail = null; writeUrl();
    for (const row of $("halo-table-body").querySelectorAll("[data-halo-id]")) {
      const selected = row.dataset.haloId === id;
      row.classList.toggle("selected", selected); row.setAttribute("aria-selected", String(selected));
    }
    drawPlot(); loadSelection();
  }
  async function loadSelection() {
    state.detailController?.abort(); state.detailController = new AbortController();
    const sequence = ++state.detailSequence;
    $("selection-dot").hidden = !state.selected;
    $("detail-empty").hidden = !!state.selected; $("halo-detail").hidden = !state.selected;
    if (!state.selected) { $("halo-detail").replaceChildren(); return; }
    $("halo-detail").replaceChildren(node("div", "detail-loading", `Loading halo ${state.selected}…`));
    try {
      const result = await fetchJson(`/api/halos/${encodeURIComponent(state.selected)}?catalog=${encodeURIComponent(state.catalog.id)}`, state.detailController.signal);
      if (sequence !== state.detailSequence) return;
      state.detail = result; renderDetail(); drawPlot();
    } catch (error) {
      if (error.name === "AbortError" || sequence !== state.detailSequence) return;
      $("halo-detail").replaceChildren(node("p", "detail-error", `Unable to inspect halo: ${error.message}`));
    }
  }
  function relationshipButton(halo, withMass = false) {
    const button = node("button", "relationship-link", String(halo.halo_id)); button.type = "button";
    button.append(node("span", "", "↗"));
    if (withMass) button.append(node("small", "", `${displayNumber(halo.mvir, 4)} M☉/h`));
    button.addEventListener("click", () => selectHalo(String(halo.halo_id)));
    return button;
  }
  function renderDetail() {
    const {halo, host, ancestors = [], subhalos = [], subhalo_count = 0} = state.detail;
    const container = $("halo-detail"); container.replaceChildren();
    const summary = node("div", "halo-summary");
    summary.append(node("p", "eyebrow", "SELECTED HALO"), node("h3", "", String(halo.halo_id)), typeBadge(halo));
    if (!matchesFilters(halo)) summary.append(node("p", "outside-selection", "Outside active filters. Host and subhalo links remain available."));
    const properties = node("dl", "summary-properties");
    for (const [label, key, unit] of [["Virial mass", "mvir", "M☉/h"], ["Virial radius", "rvir", "comoving kpc/h"], ["Vmax", "vcircMax", "km/s"], ["Spin λ", "lambda", "dimensionless"]]) {
      const block = node("div"); const value = node("dd", "", displayNumber(halo[key], 4)); value.append(node("small", "", unit)); block.append(node("dt", "", label), value); properties.append(block);
    }
    summary.append(properties); container.append(summary);
    const hostSection = node("section", "detail-section"); hostSection.append(node("h4", "", "Containing host"));
    if (host) { hostSection.append(relationshipButton(host), node("p", "relationship-meta", `${displayNumber(host.mvir, 4)} M☉/h · direct host`)); }
    else hostSection.append(node("p", "no-relationship", "Distinct halo · no containing host"));
    if (ancestors.length > 1) {
      const chain = node("div", "ancestor-list");
      chain.append(node("span", "", "ANCESTORS"));
      ancestors.forEach((ancestor, index) => {
        if (index) chain.append(node("span", "", "→"));
        const button = node("button", "", String(ancestor.halo_id)); button.type = "button"; button.addEventListener("click", () => selectHalo(String(ancestor.halo_id))); chain.append(button);
      }); hostSection.append(chain);
    }
    container.append(hostSection);
    const subs = node("section", "detail-section"); subs.append(node("h4", "", `Direct subhalos · ${count(subhalo_count)}`));
    if (!subhalos.length) subs.append(node("p", "no-relationship", "No direct subhalos in this catalog."));
    else {
      const list = node("div", "subhalo-list"); for (const subhalo of subhalos) list.append(relationshipButton(subhalo, true)); subs.append(list);
      if (subhalos.length < subhalo_count) subs.append(node("p", "relationship-meta", `Showing ${count(subhalos.length)} of ${count(subhalo_count)} direct subhalos.`));
    }
    container.append(subs);
    for (const [group, names] of propertyGroups) {
      const details = node("details", "property-group"); details.append(node("summary", "", group));
      const list = node("dl", "property-list");
      for (const name of names) {
        if (!(name in halo)) continue;
        const row = node("div", "property-row"); const label = node("dt", "", labels[name] || name);
        label.title = name; if (units[name]) label.append(node("small", "", units[name]));
        const identifier = ["halo_id", "host_id", "fof_id", "local_halo_id", "rank", "ordinal", "sys", "pid"].includes(name);
        const value = halo[name] === null ? "—" : identifier ? String(halo[name]) : displayNumber(halo[name], 8);
        row.append(label, node("dd", "", value)); list.append(row);
      }
      details.append(list); container.append(details);
    }
  }
  function updateProjection() {
    for (const button of document.querySelectorAll("[data-view]")) button.setAttribute("aria-pressed", String(button.dataset.view === state.view));
    canvas.setAttribute("aria-label", `${state.view.toUpperCase()} halo spatial projection in comoving Mpc per h, with equal physical scale on both axes. Select a halo using the catalog table for keyboard access.`);
    canvas.title = "Both axes use the same physical scale. The view expands to show the complete matching region without stretching its geometry.";
    document.querySelector(".coordinate-label").textContent = "EQUAL SCALE · Mpc/h";
  }
  function axisBounds(axis) {
    const values = state.bounds?.[axis];
    let low = Array.isArray(values) ? Number(values[0]) : 0;
    let high = Array.isArray(values) ? Number(values[1]) : 1;
    if (!Number.isFinite(low) || !Number.isFinite(high)) { low = 0; high = 1; }
    if (high <= low) { const padding = Math.max(Math.abs(low) * 0.03, 0.5); low -= padding; high += padding; }
    else { const padding = (high - low) * 0.04; low -= padding; high += padding; }
    return [low, high];
  }
  function massColor(fraction, alpha = 0.65) {
    const t = Math.min(1, Math.max(0, fraction));
    return `rgba(${Math.round(72 + 162 * t)},${Math.round(177 - 17 * t)},${Math.round(187 - 83 * t)},${alpha})`;
  }
  function drawPlot() {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const ratio = window.devicePixelRatio || 1;
    if (plotSize.width !== rect.width || plotSize.height !== rect.height || canvas.width !== Math.round(rect.width * ratio)) {
      canvas.width = Math.round(rect.width * ratio); canvas.height = Math.round(rect.height * ratio); plotSize = {width:rect.width,height:rect.height};
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, rect.width, rect.height);
    const margin = {left: 57, right: 18, top: 20, bottom: 47};
    const width = rect.width - margin.left - margin.right; const height = rect.height - margin.top - margin.bottom;
    if (width <= 0 || height <= 0) return;
    const [horizontal, vertical] = state.view.split("");
    let [xmin, xmax] = axisBounds(horizontal); let [ymin, ymax] = axisBounds(vertical);
    // Expand around the matching region's center so a physical distance has
    // exactly the same pixel length on either axis. Reserve marker padding
    // before fitting so boundary points and their selection rings stay visible.
    const centerX = (xmin + xmax) / 2; const centerY = (ymin + ymax) / 2;
    const unitsPerPixel = Math.max((xmax - xmin) / Math.max(1, width - 18), (ymax - ymin) / Math.max(1, height - 18));
    xmin = centerX - unitsPerPixel * width / 2; xmax = centerX + unitsPerPixel * width / 2;
    ymin = centerY - unitsPerPixel * height / 2; ymax = centerY + unitsPerPixel * height / 2;
    context.font = '9px "SFMono-Regular",Consolas,monospace'; context.lineWidth = 1;
    for (let tick = 0; tick <= 4; ++tick) {
      const x = margin.left + width * tick / 4; const y = margin.top + height * tick / 4;
      context.strokeStyle = "#20313d"; context.beginPath(); context.moveTo(x, margin.top); context.lineTo(x, margin.top + height); context.stroke();
      context.beginPath(); context.moveTo(margin.left, y); context.lineTo(margin.left + width, y); context.stroke();
      context.fillStyle = "#59788b"; context.textAlign = "center"; context.fillText(displayNumber(xmin + (xmax - xmin) * tick / 4, 3), x, margin.top + height + 18);
      context.textAlign = "right"; context.fillText(displayNumber(ymax - (ymax - ymin) * tick / 4, 3), margin.left - 10, y + 3);
    }
    context.fillStyle = "#86a9b7"; context.textAlign = "center"; context.font = '9px "SFMono-Regular",Consolas,monospace';
    context.fillText(`${horizontal.toUpperCase()} [cMpc/h]`, margin.left + width / 2, rect.height - 9);
    context.save(); context.translate(14, margin.top + height / 2); context.rotate(-Math.PI / 2); context.fillText(`${vertical.toUpperCase()} [cMpc/h]`, 0, 0); context.restore();
    const scaleX = (value) => margin.left + (value - xmin) / (xmax - xmin) * width;
    const scaleY = (value) => margin.top + height - (value - ymin) / (ymax - ymin) * height;
    let minMass = Infinity; let maxMass = -Infinity;
    for (const halo of state.points) { const mass = Math.log10(Number(halo.mvir)); if (Number.isFinite(mass)) { minMass = Math.min(minMass, mass); maxMass = Math.max(maxMass, mass); } }
    if (!Number.isFinite(minMass)) { minMass = 0; maxMass = 1; }
    state.plotted = [];
    context.save(); context.beginPath(); context.rect(margin.left - 2, margin.top - 2, width + 4, height + 4); context.clip();
    const size = state.points.length > 10000 ? 1.45 : state.points.length > 2000 ? 1.9 : 2.5;
    let selectedPoint = null;
    for (const halo of state.points) {
      const x = scaleX(Number(halo[horizontal])); const y = scaleY(Number(halo[vertical]));
      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      const mass = Math.log10(Number(halo.mvir)); const fraction = maxMass > minMass ? (mass - minMass) / (maxMass - minMass) : 0.5;
      context.fillStyle = massColor(Number.isFinite(fraction) ? fraction : 0.5, state.points.length > 10000 ? 0.45 : 0.73);
      context.beginPath(); context.arc(x, y, size + Math.max(0, fraction) * 0.8, 0, Math.PI * 2); context.fill();
      state.plotted.push({x, y, halo});
      if (String(halo.halo_id) === state.selected) selectedPoint = {x, y};
    }
    const detailHalo = state.detail?.halo;
    if (detailHalo && !selectedPoint && matchesFilters(detailHalo)) {
      const x = scaleX(Number(detailHalo[horizontal])); const y = scaleY(Number(detailHalo[vertical]));
      if (Number.isFinite(x) && Number.isFinite(y) && x >= margin.left && x <= margin.left + width && y >= margin.top && y <= margin.top + height) selectedPoint = {x, y};
    }
    if (selectedPoint) {
      context.strokeStyle = "#eebb83"; context.fillStyle = "#fff0d5"; context.lineWidth = 1.4;
      context.beginPath(); context.arc(selectedPoint.x, selectedPoint.y, 7, 0, Math.PI * 2); context.stroke(); context.beginPath(); context.arc(selectedPoint.x, selectedPoint.y, 2.4, 0, Math.PI * 2); context.fill();
      context.strokeStyle = "#eebb8330"; context.lineWidth = 1; context.beginPath(); context.moveTo(selectedPoint.x, margin.top); context.lineTo(selectedPoint.x, margin.top + height); context.moveTo(margin.left, selectedPoint.y); context.lineTo(margin.left + width, selectedPoint.y); context.stroke();
    }
    context.restore();
  }
  function nearestPoint(event) {
    const rect = canvas.getBoundingClientRect(); const x = event.clientX - rect.left; const y = event.clientY - rect.top;
    let result = null; let distance = 100;
    for (const point of state.plotted) { const candidate = (point.x - x) ** 2 + (point.y - y) ** 2; if (candidate < distance) { result = point; distance = candidate; } }
    return result;
  }
  canvas.addEventListener("pointermove", (event) => {
    const point = nearestPoint(event); const tooltip = $("plot-tooltip");
    tooltip.hidden = !point; canvas.style.cursor = point ? "pointer" : "crosshair";
    if (!point) return;
    tooltip.textContent = `Halo ${point.halo.halo_id}\nMvir  ${displayNumber(point.halo.mvir, 4)} M☉/h`;
    tooltip.style.left = `${Math.max(4, Math.min(point.x + 14, plotSize.width - 235))}px`;
    tooltip.style.top = `${Math.max(4, point.y - 62)}px`;
  });
  canvas.addEventListener("pointerleave", () => { $("plot-tooltip").hidden = true; });
  canvas.addEventListener("click", (event) => { const point = nearestPoint(event); if (point) selectHalo(String(point.halo.halo_id)); });
  new ResizeObserver(drawPlot).observe(canvas);
  window.addEventListener("resize", drawPlot);
  $("catalog-select").addEventListener("change", () => {
    state.catalog = state.catalogs.find((catalog) => String(catalog.id) === $("catalog-select").value); state.page = 0; state.selected = null; state.detail = null;
    updateCatalog(); writeUrl(); loadSelection(); loadData();
  });
  $("filter-form").addEventListener("submit", (event) => {
    event.preventDefault(); const data = new FormData(event.currentTarget); const filters = {}; let error = "";
    for (const name of filterNames) { const value = String(data.get(name) || "").trim(); if (value && !(name === "kind" && value === "all")) filters[name] = value; }
    if (filters.halo_id && !/^\d+$/.test(filters.halo_id)) error = "Halo ID must contain decimal digits only.";
    for (const name of numericFilters) if (filters[name] !== undefined && !Number.isFinite(Number(filters[name]))) error = "Filter values must be finite numbers.";
    for (const field of ["mvir", "rvir", "vcircMax", "lambda", "x", "y", "z"]) if (filters[`${field}_min`] !== undefined && filters[`${field}_max`] !== undefined && Number(filters[`${field}_min`]) > Number(filters[`${field}_max`])) error = "Each minimum must be less than or equal to its maximum.";
    $("filter-error").textContent = error; $("filter-error").hidden = !error; if (error) return;
    state.filters = filters; state.page = 0; updateFilterCount(); writeUrl(); if (state.detail) renderDetail(); loadData();
  });
  $("filter-form").addEventListener("reset", () => {
    setTimeout(() => { state.filters = {}; state.page = 0; $("filter-error").hidden = true; updateFilterCount(); writeUrl(); if (state.detail) renderDetail(); loadData(); }, 0);
  });
  for (const button of document.querySelectorAll("[data-view]")) button.addEventListener("click", () => { state.view = button.dataset.view; updateProjection(); writeUrl(); drawPlot(); });
  for (const button of document.querySelectorAll("[data-sort]")) button.addEventListener("click", () => {
    if (state.sort === button.dataset.sort) state.order = state.order === "desc" ? "asc" : "desc";
    else { state.sort = button.dataset.sort; state.order = state.sort === "halo_id" ? "asc" : "desc"; }
    state.page = 0; writeUrl(); loadTable();
  });
  $("previous-page").addEventListener("click", () => { if (state.page > 0) { --state.page; writeUrl(); loadTable(); } });
  $("next-page").addEventListener("click", () => { if ((state.page + 1) * pageSize < state.total) { ++state.page; writeUrl(); loadTable(); } });
  $("reload-catalogs").addEventListener("click", loadCatalogs);
  window.addEventListener("popstate", () => { if (state.catalogs.length) { readUrl(); loadSelection(); loadData(); } });
  loadCatalogs();
})();
