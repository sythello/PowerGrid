(() => {
  "use strict";

  const PHASES = [
    ["auction", "竞拍电厂"],
    ["buy_resources", "购买资源"],
    ["build_houses", "建设网络"],
    ["bureaucracy", "供电结算"],
  ];
  const PHASE_LABELS = Object.fromEntries(PHASES);
  const RESOURCE_ORDER = ["coal", "oil", "garbage", "uranium"];
  const RESOURCE_LABELS = {
    coal: "煤",
    oil: "石油",
    garbage: "垃圾",
    uranium: "铀",
  };
  const RESOURCE_COLORS = {
    coal: "#704526",
    oil: "#202728",
    garbage: "#e3b326",
    uranium: "#b84535",
  };
  const PLAYER_COLORS = {
    black: "#253033",
    purple: "#804d91",
    green: "#2f7a53",
    blue: "#3677a8",
    yellow: "#c7901e",
    red: "#b64b3b",
  };
  const PRICE_COLUMNS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16];

  const ui = {
    meta: null,
    snapshot: null,
    requestKey: "",
    auctionPlant: null,
    bidValue: 0,
    resourceAmount: 0,
    buildCities: [],
    buildQuote: null,
    runs: {},
    zoom: 1,
    aiPaused: false,
    aiTimer: null,
    aiWorking: false,
    toastTimer: null,
    plantPreviewTarget: null,
    globalParametersInvoker: null,
    seatDrafts: [],
    layoutEditor: {
      active: false,
      original: null,
      draft: null,
      changed: new Set(),
      dragging: null,
      resumeAi: false,
      saving: false,
    },
  };

  const refs = {};

  document.addEventListener("DOMContentLoaded", initialize);

  async function initialize() {
    Object.assign(refs, {
      launcher: document.querySelector("#launcher"),
      gameShell: document.querySelector("#game-shell"),
      newGameForm: document.querySelector("#new-game-form"),
      mapSelect: document.querySelector("#map-select"),
      playerCount: document.querySelector("#player-count"),
      seedInput: document.querySelector("#seed-input"),
      seatList: document.querySelector("#seat-list"),
      startGame: document.querySelector("#start-game"),
      launcherStatus: document.querySelector("#launcher-status"),
      newGameButton: document.querySelector("#new-game-button"),
      aiToggle: document.querySelector("#ai-toggle"),
      phaseTrack: document.querySelector("#phase-track"),
      roundNumber: document.querySelector("#round-number"),
      stepNumber: document.querySelector("#step-number"),
      playerList: document.querySelector("#player-list"),
      mapKicker: document.querySelector("#map-kicker"),
      mapTitle: document.querySelector("#map-title"),
      activeCallout: document.querySelector("#active-callout"),
      boardViewport: document.querySelector("#board-viewport"),
      boardCanvas: document.querySelector("#board-canvas"),
      layoutEditorButton: document.querySelector("#layout-editor-button"),
      zoomOutput: document.querySelector("#zoom-output"),
      actionConsole: document.querySelector("#action-console"),
      deckCount: document.querySelector("#deck-count"),
      plantMarket: document.querySelector("#plant-market"),
      resourceMarket: document.querySelector("#resource-market"),
      eventList: document.querySelector("#event-list"),
      toast: document.querySelector("#toast"),
      plantPreviewTooltip: document.querySelector("#plant-preview-tooltip"),
      plantPreviewImage: document.querySelector("#plant-preview-image"),
      plantPreviewLabel: document.querySelector("#plant-preview-label"),
      globalParametersDialog: document.querySelector("#global-parameters-dialog"),
      globalParametersContent: document.querySelector("#global-parameters-content"),
      winnerDialog: document.querySelector("#winner-dialog"),
      winnerContent: document.querySelector("#winner-content"),
    });

    refs.newGameForm.addEventListener("submit", startGame);
    refs.playerCount.addEventListener("change", () => {
      captureSeatDrafts();
      renderSeatRows(Number(refs.playerCount.value));
    });
    refs.mapSelect.addEventListener("change", () => {
      captureSeatDrafts();
      renderSeatRows(Number(refs.playerCount.value));
    });
    refs.newGameButton.addEventListener("click", () => showLauncher(true));
    refs.aiToggle.addEventListener("click", toggleAi);
    document.addEventListener("click", handleClick);
    document.addEventListener("change", handleChange);
    document.addEventListener("input", handleInput);
    document.addEventListener("keydown", handleKeydown);
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("pointermove", handlePointerMove);
    document.addEventListener("pointerup", handlePointerUp);
    document.addEventListener("pointercancel", handlePointerUp);
    document.addEventListener("pointerover", handlePlantPreviewOver);
    document.addEventListener("pointerout", handlePlantPreviewOut);
    document.addEventListener("focusin", handlePlantPreviewFocusIn);
    document.addEventListener("focusout", handlePlantPreviewFocusOut);
    document.addEventListener("scroll", hidePlantPreview, true);
    window.addEventListener("resize", hidePlantPreview);
    refs.plantPreviewImage.addEventListener("load", () => {
      if (ui.plantPreviewTarget) positionPlantPreview(ui.plantPreviewTarget);
    });
    refs.plantPreviewImage.addEventListener("error", hidePlantPreview);

    try {
      const [meta, state] = await Promise.all([api("/api/meta"), api("/api/state")]);
      ui.meta = meta;
      setupLauncher();
      refs.launcherStatus.textContent = "配置完成，可以启动。";
      if (state.has_game) {
        setSnapshot(state);
        showGame();
      }
    } catch (error) {
      refs.launcherStatus.textContent = `无法连接游戏服务：${error.message}`;
      showToast(error.message);
    }
  }

  function setupLauncher() {
    refs.mapSelect.innerHTML = ui.meta.maps
      .map((map) => `<option value="${escapeHtml(map.id)}">${escapeHtml(map.name)}</option>`)
      .join("");
    refs.mapSelect.value = ui.meta.defaults.map_id;
    refs.playerCount.value = String(ui.meta.defaults.player_count);
    refs.seedInput.value = String(ui.meta.defaults.seed);
    ui.seatDrafts = Array.from({ length: 6 }, (_, index) => ({
      name: `玩家 ${index + 1}`,
      controller: ui.meta.defaults.controllers[index] || ui.meta.defaults.ai_controller,
    }));
    renderSeatRows(ui.meta.defaults.player_count);
  }

  function renderSeatRows(count) {
    while (ui.seatDrafts.length < count) {
      const index = ui.seatDrafts.length;
      ui.seatDrafts.push({ name: `玩家 ${index + 1}`, controller: ui.meta.defaults.ai_controller });
    }
    refs.seatList.innerHTML = ui.seatDrafts
      .slice(0, count)
      .map((seat, index) => {
        if (!controllerIsAvailable(seat.controller, count)) {
          seat.controller = firstAvailableAiController(count);
        }
        const options = ui.meta.controllers
          .map((controller) => {
            const available = controllerIsAvailable(controller.id, count);
            const support = available || controller.id === "human" ? "" : "（仅德国地图 3 人局）";
            return `<option value="${escapeHtml(controller.id)}" ${controller.id === seat.controller ? "selected" : ""} ${available ? "" : "disabled"}>${escapeHtml(controller.name + support)}</option>`;
          })
          .join("");
        return `
          <div class="seat-row" data-seat="${index}">
            <label>
              <span>席位 ${String(index + 1).padStart(2, "0")}</span>
              <input class="seat-name" maxlength="32" value="${escapeHtml(seat.name)}" aria-label="席位 ${index + 1} 名称" />
            </label>
            <label>
              <span>控制方式</span>
              <select class="seat-controller" aria-label="席位 ${index + 1} 控制方式">${options}</select>
            </label>
          </div>`;
      })
      .join("");
  }

  function controllerIsAvailable(controllerId, playerCount) {
    const controller = ui.meta.controllers.find((candidate) => candidate.id === controllerId);
    if (!controller) return false;
    const mapSupported = !controller.supported_maps || controller.supported_maps.includes(refs.mapSelect.value);
    const countSupported = !controller.supported_player_counts || controller.supported_player_counts.includes(playerCount);
    return mapSupported && countSupported;
  }

  function firstAvailableAiController(playerCount) {
    return ui.meta.controllers.find(
      (controller) => controller.id !== "human" && controllerIsAvailable(controller.id, playerCount),
    )?.id || "human";
  }

  function captureSeatDrafts() {
    refs.seatList.querySelectorAll(".seat-row").forEach((row) => {
      const index = Number(row.dataset.seat);
      ui.seatDrafts[index] = {
        name: row.querySelector(".seat-name").value.trim() || `玩家 ${index + 1}`,
        controller: row.querySelector(".seat-controller").value,
      };
    });
  }

  async function startGame(event) {
    event.preventDefault();
    captureSeatDrafts();
    const count = Number(refs.playerCount.value);
    const players = ui.seatDrafts.slice(0, count);
    if (!players.some((player) => player.controller === "human")) {
      refs.launcherStatus.textContent = "建议至少保留一位本地玩家；当前会自动进行 AI 对局。";
    } else {
      refs.launcherStatus.textContent = "正在初始化电网…";
    }
    refs.startGame.disabled = true;
    try {
      const result = await api("/api/game", {
        method: "POST",
        body: {
          map_id: refs.mapSelect.value,
          seed: Number(refs.seedInput.value || 7),
          players,
        },
      });
      ui.aiPaused = false;
      ui.zoom = result.layout?.fit === "portrait" ? 0.82 : 1;
      resetLayoutEditorState();
      setSnapshot(result, true);
      showGame();
    } catch (error) {
      refs.launcherStatus.textContent = error.message;
      showToast(error.message);
    } finally {
      refs.startGame.disabled = false;
    }
  }

  function showLauncher(confirmFirst = false) {
    if (ui.layoutEditor.active) {
      const message = ui.layoutEditor.changed.size
        ? "尚有未保存的城市坐标，放弃调整并返回设置页？"
        : "退出城市坐标调整并返回设置页？";
      if (!window.confirm(message)) return;
      finishLayoutEditor();
      confirmFirst = false;
    }
    if (confirmFirst && ui.snapshot && !window.confirm("返回设置页？当前本地对局会保留到下一局开始。")) {
      return;
    }
    clearTimeout(ui.aiTimer);
    ui.aiPaused = true;
    refs.winnerDialog.hidden = true;
    closeGlobalParameters(false);
    refs.gameShell.hidden = true;
    refs.launcher.hidden = false;
  }

  function showGame() {
    refs.launcher.hidden = true;
    refs.gameShell.hidden = false;
    renderAll();
  }

  function setSnapshot(snapshot, forceReset = false) {
    const previousKey = ui.requestKey;
    ui.snapshot = snapshot;
    ui.requestKey = makeRequestKey(snapshot);
    if (forceReset || previousKey !== ui.requestKey) {
      ui.auctionPlant = null;
      ui.bidValue = 0;
      ui.resourceAmount = 0;
      ui.buildCities = [];
      ui.buildQuote = null;
      ui.runs = {};
    }
    if (!snapshot.needs_ai_advance) {
      ui.aiPaused = false;
    }
  }

  function makeRequestKey(snapshot) {
    const state = snapshot.state;
    const request = snapshot.request;
    if (!request) return `${state.round_number}:${state.step}:${state.phase}:none`;
    const auction = state.auction_state || {};
    return [
      state.round_number,
      state.step,
      state.phase,
      request.player_id,
      request.decision_type,
      request.metadata?.resource || "",
      auction.active_plant_price || "",
      auction.current_bid || "",
    ].join(":");
  }

  function renderAll() {
    if (!ui.snapshot?.has_game) return;
    renderHeader();
    renderPlayers();
    renderBoard();
    renderPlantMarket();
    renderResourceMarket();
    renderGlobalParameters();
    renderActionConsole();
    renderEvents();
    renderWinner();
    scheduleAi();
  }

  function renderHeader() {
    const { state, request } = ui.snapshot;
    const phaseIndex = PHASES.findIndex(([id]) => id === state.phase);
    refs.phaseTrack.innerHTML = PHASES.map(([id, label], index) => {
      const status = index === phaseIndex ? "active" : index < phaseIndex ? "complete" : "";
      return `<div class="phase-item ${status}"><i></i><span>${label}</span></div>`;
    }).join("");
    refs.roundNumber.textContent = state.round_number;
    refs.stepNumber.textContent = state.step;
    const active = request ? playerById(request.player_id) : null;
    refs.activeCallout.innerHTML = ui.layoutEditor.active
      ? `<b>布局编辑模式</b> · 拖动城市圆圈校准位置`
      : active
        ? `当前操作：<b>${escapeHtml(active.name)}</b> · ${escapeHtml(PHASE_LABELS[state.phase] || state.phase)}`
        : "正在推进回合状态…";
    refs.aiToggle.hidden = ui.layoutEditor.active || !ui.snapshot.needs_ai_advance;
    refs.aiToggle.textContent = ui.aiPaused ? "继续 AI" : "暂停 AI";
    refs.layoutEditorButton.disabled = ui.layoutEditor.active;
    refs.layoutEditorButton.textContent = ui.layoutEditor.active ? "正在调整坐标" : "调整城市坐标";
    refs.layoutEditorButton.classList.toggle("active", ui.layoutEditor.active);
    refs.gameShell.classList.toggle("layout-editing", ui.layoutEditor.active);
  }

  function renderPlayers() {
    hidePlantPreview();
    const { state, request } = ui.snapshot;
    const players = state.player_order.map((playerId) => playerById(playerId)).filter(Boolean);
    refs.playerList.innerHTML = players
      .map((player, index) => {
        const totals = resourceTotals(player.resource_storage);
        const pips = RESOURCE_ORDER.map(
          (resource) =>
            `<span class="resource-pip" style="--resource-color:${RESOURCE_COLORS[resource]}" title="${RESOURCE_LABELS[resource]}">${totals[resource]}</span>`,
        ).join("");
        const plants = player.power_plants.length
          ? player.power_plants
              .filter((plant) => !plant.is_step_3_placeholder)
              .sort((a, b) => a.price - b.price)
              .map((plant) => {
                const label = `#${plant.price} 发电厂 · 可供 ${plant.output_cities} 城`;
                return `<span
                  class="plant-number owned-plant-number"
                  data-owned-plant-preview="/assets/plants/${plant.price}.png"
                  data-plant-preview-label="${escapeHtml(label)}"
                  tabindex="0"
                  aria-describedby="plant-preview-tooltip"
                  aria-label="${escapeHtml(label)}，悬浮或聚焦查看卡牌"
                >#${plant.price}</span>`;
              })
              .join("")
          : `<span class="plant-number">暂无电厂</span>`;
        const active = request?.player_id === player.player_id ? "active" : "";
        return `
          <article class="player-card ${active}" style="--player-color:${playerColor(player.color)}">
            <div class="player-card-head">
              <strong>${escapeHtml(player.name)}</strong>
              <span>${String(index + 1).padStart(2, "0")} · ${player.controller === "human" ? "LOCAL" : "AI"}</span>
            </div>
            <div class="player-stats">
              <div><b>${player.elektro}</b><span>ELEKTRO</span></div>
              <div><b>${player.network_city_ids.length}</b><span>城市</span></div>
              <div><b>${player.houses_in_supply}</b><span>余屋</span></div>
            </div>
            <div class="resource-pips">${pips}</div>
            <div class="owned-plants">${plants}</div>
          </article>`;
      })
      .join("");
  }

  function handlePlantPreviewOver(event) {
    const target = event.target.closest?.("[data-owned-plant-preview]");
    if (!target || target === ui.plantPreviewTarget) return;
    showPlantPreview(target);
  }

  function handlePlantPreviewOut(event) {
    const target = event.target.closest?.("[data-owned-plant-preview]");
    if (!target || target.contains(event.relatedTarget) || document.activeElement === target) return;
    hidePlantPreview();
  }

  function handlePlantPreviewFocusIn(event) {
    const target = event.target.closest?.("[data-owned-plant-preview]");
    if (target) showPlantPreview(target);
  }

  function handlePlantPreviewFocusOut(event) {
    const target = event.target.closest?.("[data-owned-plant-preview]");
    if (!target || target.matches(":hover")) return;
    hidePlantPreview();
  }

  function showPlantPreview(target) {
    ui.plantPreviewTarget = target;
    refs.plantPreviewLabel.textContent = target.dataset.plantPreviewLabel || "发电厂";
    refs.plantPreviewTooltip.style.left = "-9999px";
    refs.plantPreviewTooltip.style.top = "-9999px";
    refs.plantPreviewTooltip.hidden = false;
    refs.plantPreviewImage.src = target.dataset.ownedPlantPreview;
    positionPlantPreview(target);
  }

  function positionPlantPreview(target) {
    if (target !== ui.plantPreviewTarget || !target.isConnected || refs.plantPreviewTooltip.hidden) return;
    const anchor = target.getBoundingClientRect();
    const preview = refs.plantPreviewTooltip.getBoundingClientRect();
    const gap = 12;
    const edge = 8;
    let left = anchor.right + gap;
    if (left + preview.width > window.innerWidth - edge) left = anchor.left - preview.width - gap;
    left = clamp(left, edge, Math.max(edge, window.innerWidth - preview.width - edge));
    const top = clamp(
      anchor.top + (anchor.height - preview.height) / 2,
      edge,
      Math.max(edge, window.innerHeight - preview.height - edge),
    );
    refs.plantPreviewTooltip.style.left = `${Math.round(left)}px`;
    refs.plantPreviewTooltip.style.top = `${Math.round(top)}px`;
  }

  function hidePlantPreview() {
    ui.plantPreviewTarget = null;
    if (refs.plantPreviewTooltip) refs.plantPreviewTooltip.hidden = true;
  }

  function renderBoard() {
    const { state, layout, request } = ui.snapshot;
    const map = state.game_map;
    const size = layout.size;
    const width = size.width;
    const height = size.height;
    const editorActive = ui.layoutEditor.active;
    const positions = editorActive ? ui.layoutEditor.draft : layout.cities || {};
    const selectedRegions = new Set(state.selected_regions || []);
    const buildActions = (request?.legal_actions || []).filter((action) => action.action_type === "build_city");
    const clickable = new Set(editorActive ? [] : buildActions.map((action) => action.payload.city_id));
    const selected = new Set(ui.buildCities);

    refs.mapKicker.textContent = `${map.id.toUpperCase()} · NETWORK MAP`;
    refs.mapTitle.textContent = `${map.name} 电网`;
    refs.zoomOutput.textContent = `${Math.round(ui.zoom * 100)}%`;
    refs.boardCanvas.style.width = `${Math.round(ui.zoom * 100)}%`;

    let base = "";
    if (layout.asset) {
      base = `<image href="${escapeHtml(layout.asset)}" x="0" y="0" width="${width}" height="${height}" />`;
    } else {
      const lines = map.connections
        .map((connection) => {
          const start = normalizedPoint(positions[connection.city_1], width, height);
          const end = normalizedPoint(positions[connection.city_2], width, height);
          if (!start || !end) return "";
          const mx = (start.x + end.x) / 2;
          const my = (start.y + end.y) / 2;
          return `
            <line class="map-connection" x1="${start.x}" y1="${start.y}" x2="${end.x}" y2="${end.y}" />
            <circle class="map-cost" cx="${mx}" cy="${my}" r="18" />
            <text class="map-cost-label" x="${mx}" y="${my}">${connection.cost}</text>`;
        })
        .join("");
      const schematicCities = map.cities
        .map((city) => {
          const point = normalizedPoint(positions[city.id], width, height);
          if (!point) return "";
          return `
            <circle class="schematic-city" cx="${point.x}" cy="${point.y}" r="32" />
            <text class="schematic-city-label" x="${point.x}" y="${point.y - 48}">${escapeHtml(city.name)}</text>`;
        })
        .join("");
      base = `<rect width="${width}" height="${height}" fill="#ded5c3" />${lines}${schematicCities}`;
    }

    const occupancy = {};
    for (const player of state.players) {
      for (const cityId of player.network_city_ids) {
        (occupancy[cityId] ||= []).push(player);
      }
    }
    const nodeRadius = layout.fit === "portrait" ? 44 : 35;
    const cityOverlay = map.cities
      .map((city) => {
        const point = normalizedPoint(positions[city.id], width, height);
        if (!point) return "";
        const isInactive = selectedRegions.size && !selectedRegions.has(city.region);
        const classes = [
          "city-hit",
          isInactive ? "inactive" : "",
          clickable.has(city.id) ? "clickable" : "",
          selected.has(city.id) ? "selected" : "",
        ].filter(Boolean).join(" ");
        const houses = (occupancy[city.id] || [])
          .slice(0, 3)
          .map((player, index) => {
            const slot = houseSlotPoint(positions[city.id], point, index, layout, width, height, nodeRadius);
            return renderHouse(slot.x, slot.y, playerColor(player.color), nodeRadius);
          })
          .join("");
        const action = buildActions.find((candidate) => candidate.payload.city_id === city.id);
        const priceText = action ? ` · 当前最低 ${action.payload.total_cost} Elektro` : "";
        return `
          <g class="${classes}" data-city="${escapeHtml(city.id)}" ${clickable.has(city.id) ? 'role="button" tabindex="0"' : ""}>
            <title>${escapeHtml(city.name)}${escapeHtml(priceText)}</title>
            <circle class="city-target" cx="${point.x}" cy="${point.y}" r="${nodeRadius}" />
            ${houses}
          </g>`;
      })
      .join("");
    const editorOverlay = editorActive
      ? map.cities
          .map((city) => {
            const position = positions[city.id];
            const point = normalizedPoint(position, width, height);
            if (!point) return "";
            const coordinate = `${Number(position.x).toFixed(5)}, ${Number(position.y).toFixed(5)}`;
            return `
              <g
                class="city-coordinate"
                data-layout-city="${escapeHtml(city.id)}"
                transform="translate(${point.x} ${point.y})"
                role="button"
                tabindex="0"
                aria-label="拖动 ${escapeHtml(city.name)}，当前坐标 ${coordinate}"
              >
                <title>${escapeHtml(city.name)} · ${coordinate}</title>
                <circle class="city-coordinate-hit" r="${nodeRadius + 10}" />
                <circle class="city-coordinate-ring" r="${nodeRadius}" />
                <path class="city-coordinate-cross" d="M -10 0 H 10 M 0 -10 V 10" />
                <text class="city-coordinate-name" y="${-nodeRadius - 12}">${escapeHtml(city.name)}</text>
                <text class="city-coordinate-value" y="${nodeRadius + 24}">${coordinate}</text>
              </g>`;
          })
          .join("")
      : "";
    const mappedCount = map.cities.filter((city) => normalizedPoint(positions[city.id], width, height)).length;
    const accessibilityTitle = `${map.name} 电网地图，共 ${map.cities.length} 座城市，已定位 ${mappedCount} 座。`;
    refs.boardCanvas.innerHTML = `
      <svg class="grid-map" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(accessibilityTitle)}">
        ${base}
        ${cityOverlay}
        ${editorOverlay}
      </svg>`;
  }

  function houseSlotPoint(cityPosition, center, index, layout, width, height, radius) {
    const calibrated = normalizedPoint(cityPosition?.house_slots?.[index], width, height);
    if (calibrated) return calibrated;
    const fallbackOffsets = [
      { x: 0, y: -radius * 0.52 },
      { x: -radius * 0.48, y: radius * 0.36 },
      { x: radius * 0.48, y: radius * 0.36 },
    ];
    const offset = layout.house_slot_offsets?.[index] || fallbackOffsets[index] || { x: 0, y: 0 };
    return { x: center.x + Number(offset.x), y: center.y + Number(offset.y) };
  }

  function renderHouse(cx, cy, color, radius) {
    const s = radius * 0.25;
    const roofBase = cy - s * 0.05;
    const roof = `${cx - s},${roofBase} ${cx},${cy - s * 0.85} ${cx + s},${roofBase}`;
    return `
      <g class="house-token" style="--house-color:${color}" pointer-events="none">
        <polygon points="${roof}" />
        <rect x="${cx - s * 0.7}" y="${roofBase}" width="${s * 1.4}" height="${s * 0.9}" rx="2" />
      </g>`;
  }

  function renderPlantMarket() {
    const { state, request } = ui.snapshot;
    const startPrices = new Set(
      (request?.legal_actions || [])
        .filter((action) => action.action_type === "auction_start")
        .map((action) => Number(action.payload.plant_price)),
    );
    const activePrice = state.auction_state?.active_plant_price;
    const discountPrice = state.auction_state?.discount_token_plant_price;
    const backLabels = {
      plug: "PLUG · 插头",
      socket: "SOCKET · 插座",
    };
    const deckBack = backLabels[state.deck_top_back] ? state.deck_top_back : "empty";
    refs.deckCount.innerHTML = `
      <span>牌堆 ${state.deck_count}</span>
      <span class="deck-back-badge ${deckBack}">牌顶 ${backLabels[state.deck_top_back] || "空"}</span>`;
    const current = state.current_market.map((plant) => renderPlantCard(plant, {
      clickable: startPrices.has(plant.price),
      selected: ui.auctionPlant === plant.price,
      active: activePrice === plant.price,
      discount: discountPrice === plant.price,
    })).join("");
    const future = state.future_market.map((plant) => renderPlantCard(plant, { future: true })).join("");
    refs.plantMarket.innerHTML = `
      <p class="market-label">CURRENT MARKET · 当前市场</p>
      <div class="plant-row">${current || '<span class="market-label">市场为空</span>'}</div>
      ${future ? `<p class="market-label">FUTURE MARKET · 未来市场</p><div class="plant-row">${future}</div>` : ""}`;
  }

  function renderPlantCard(plant, options = {}) {
    const classes = [
      "plant-card",
      options.clickable ? "clickable" : "",
      options.selected ? "selected" : "",
      options.active ? "active" : "",
      options.future ? "future" : "",
    ].filter(Boolean).join(" ");
    const source = plant.is_step_3_placeholder ? "/assets/plants/step3.png" : `/assets/plants/${plant.price}.png`;
    const label = plant.is_step_3_placeholder
      ? "阶段三"
      : `发电厂 ${plant.price}，消耗 ${plant.resource_cost || 0}，供应 ${plant.output_cities} 城`;
    return `
      <button type="button" class="${classes}" ${options.clickable ? `data-plant-price="${plant.price}"` : "disabled"} aria-label="${escapeHtml(label)}">
        <img src="${source}" alt="" onerror="this.hidden=true;this.nextElementSibling.hidden=false" />
        <span class="plant-fallback" hidden>${plant.is_step_3_placeholder ? "Ⅲ" : plant.price}</span>
        ${options.discount ? '<span class="discount-badge" title="优惠电厂">1</span>' : ""}
      </button>`;
  }

  function renderGlobalParameters() {
    const parameters = ui.snapshot?.global_parameters;
    if (!parameters) {
      refs.globalParametersContent.innerHTML = '<p class="reference-empty">当前没有可显示的对局参数。</p>';
      return;
    }
    const paymentEntries = Object.entries(parameters.payment_schedule || {})
      .map(([cities, payout]) => [Number(cities), Number(payout)])
      .sort(([left], [right]) => left - right);
    const payoutHeader = paymentEntries.map(([cities]) => `<th scope="col">${cities}</th>`).join("");
    const payoutValues = paymentEntries.map(([, payout]) => `<td>${payout}</td>`).join("");
    const refill = parameters.resource_refill || {};
    const refillRows = [1, 2, 3].map((step) => {
      const amounts = refill[`step_${step}`] || {};
      const current = step === Number(parameters.current_step) ? "current" : "";
      return `<tr class="${current}">
        <th scope="row">STEP ${step}${current ? '<span class="current-step-dot">当前</span>' : ""}</th>
        ${RESOURCE_ORDER.map((resource) => `<td>${Number(amounts[resource] || 0)}</td>`).join("")}
      </tr>`;
    }).join("");
    refs.globalParametersContent.innerHTML = `
      <div class="step-status">
        <span>当前游戏阶段</span>
        <strong>STEP ${Number(parameters.current_step)}</strong>
        <small>${Number(parameters.player_count)} 人局资源补充规则</small>
      </div>
      <div class="rule-thresholds" aria-label="城市数量阈值">
        <div>
          <span>进入 STEP 2</span>
          <strong>${Number(parameters.step_2_cities)}</strong>
          <small>城市</small>
        </div>
        <div>
          <span>触发游戏结束</span>
          <strong>${Number(parameters.end_game_cities)}</strong>
          <small>城市</small>
        </div>
      </div>
      <section class="reference-section">
        <div class="reference-section-title"><h3>供电收入</h3><span>单位：Elektro</span></div>
        <div class="reference-table-scroll">
          <table class="parameter-table payout-table">
            <tbody>
              <tr><th scope="row">供电城市</th>${payoutHeader}</tr>
              <tr><th scope="row">收入</th>${payoutValues}</tr>
            </tbody>
          </table>
        </div>
      </section>
      <section class="reference-section">
        <div class="reference-section-title"><h3>资源补充</h3><span>每轮供电结算后</span></div>
        <div class="reference-table-scroll">
          <table class="parameter-table refill-table">
            <thead><tr><th scope="col">阶段</th>${RESOURCE_ORDER.map((resource) => `<th scope="col">${RESOURCE_LABELS[resource]}</th>`).join("")}</tr></thead>
            <tbody>${refillRows}</tbody>
          </table>
        </div>
      </section>`;
  }

  function showGlobalParameters(invoker) {
    ui.globalParametersInvoker = invoker || document.activeElement;
    renderGlobalParameters();
    refs.globalParametersDialog.hidden = false;
    refs.globalParametersDialog.querySelector(".reference-close")?.focus();
  }

  function closeGlobalParameters(restoreFocus = true) {
    if (!refs.globalParametersDialog || refs.globalParametersDialog.hidden) return;
    refs.globalParametersDialog.hidden = true;
    if (restoreFocus && ui.globalParametersInvoker?.isConnected) ui.globalParametersInvoker.focus();
    ui.globalParametersInvoker = null;
  }

  function renderResourceMarket() {
    const { state, request } = ui.snapshot;
    const activeResource = request?.decision_type === "buy_resources" ? request.metadata?.resource : null;
    const headings = PRICE_COLUMNS.map((price) => `<th scope="col">${price}</th>`).join("");
    const rows = RESOURCE_ORDER.map((resource) => {
      const bands = state.resource_market.market[resource] || {};
      const cells = PRICE_COLUMNS.map((price) => {
        if (!(String(price) in bands)) return "<td></td>";
        const amount = Number(bands[String(price)] || 0);
        const tokens = Array.from({ length: amount }, () => `<i class="resource-token"></i>`).join("");
        const classes = ["resource-cell", amount ? "" : "empty", resource === activeResource ? "buyable" : ""]
          .filter(Boolean).join(" ");
        return `<td class="${classes}" style="--resource-color:${RESOURCE_COLORS[resource]}" title="${RESOURCE_LABELS[resource]}：${amount} 个，单价 ${price}"><span class="resource-tokens">${tokens}</span></td>`;
      }).join("");
      return `<tr><th class="resource-name" scope="row">${RESOURCE_LABELS[resource]}</th>${cells}</tr>`;
    }).join("");
    const supply = RESOURCE_ORDER.map(
      (resource) => `<span>${RESOURCE_LABELS[resource]}备库 <b>${state.resource_market.supply[resource]}</b></span>`,
    ).join("");
    refs.resourceMarket.innerHTML = `
      <table class="resource-table" aria-label="资源市场价格与库存">
        <thead><tr><th></th>${headings}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <div class="resource-supply">${supply}</div>`;
  }

  function renderActionConsole() {
    const { state, request, winner } = ui.snapshot;
    if (ui.layoutEditor.active) {
      renderLayoutEditorAction();
      return;
    }
    if (winner) {
      refs.actionConsole.innerHTML = consoleLayout("对局结束", "最终结算", "电网已完成最终供电核算。", "", "");
      return;
    }
    if (!request) {
      refs.actionConsole.innerHTML = consoleLayout(
        "自动调度",
        "推进回合",
        "规则引擎正在切换阶段。",
        '<div class="loading-line"></div>',
        "",
      );
      return;
    }
    const player = playerById(request.player_id);
    if (player?.controller !== "human") {
      refs.actionConsole.innerHTML = consoleLayout(
        "AI CONTROL",
        `${escapeHtml(player.name)} 正在决策`,
        `当前阶段：${PHASE_LABELS[state.phase] || state.phase}`,
        '<div class="loading-line"></div>',
        `<button class="action-button secondary" type="button" data-action="toggle-ai">${ui.aiPaused ? "继续 AI" : "暂停 AI"}</button>`,
      );
      return;
    }
    if (state.pending_decision || request.decision_type.startsWith("discard_")) {
      renderPendingDecision(request, player);
      return;
    }
    if (request.decision_type === "auction_start") {
      renderAuctionStart(request, player);
      return;
    }
    if (request.decision_type === "auction_bid") {
      renderAuctionBid(request, player);
      return;
    }
    if (request.decision_type === "buy_resources") {
      renderResourceAction(request, player);
      return;
    }
    if (request.decision_type === "build_houses") {
      renderBuildAction(request, player);
      return;
    }
    if (request.decision_type === "bureaucracy") {
      renderBureaucracyAction(request, player);
      return;
    }
    refs.actionConsole.innerHTML = consoleLayout("等待输入", escapeHtml(player.name), escapeHtml(request.prompt), "", "");
  }

  function consoleLayout(kicker, title, description, body, actions) {
    return `
      <div class="console-grid">
        <div class="console-intro">
          <p class="eyebrow">${kicker}</p>
          <h2>${title}</h2>
          <p>${description}</p>
        </div>
        <div class="console-body">${body}</div>
        <div class="console-actions">${actions}</div>
      </div>`;
  }

  function renderLayoutEditorAction() {
    const changedIds = [...ui.layoutEditor.changed];
    const changedNames = changedIds.slice(0, 5).map(cityName);
    const moreCount = Math.max(0, changedIds.length - changedNames.length);
    const summary = changedIds.length
      ? `已调整 <strong>${changedIds.length}</strong> 座城市：${changedNames.map(escapeHtml).join("、")}${moreCount ? ` 等 ${changedIds.length} 座` : ""}`
      : "尚未移动城市；所有圆圈均可拖动。";
    const body = `
      <div class="layout-editor-summary">
        <p>${summary}</p>
        <small>圆圈中心即保存位置；坐标会限制在地图边界内。</small>
      </div>`;
    const actions = `
      <button class="action-button secondary" type="button" data-action="cancel-layout-editor" ${ui.layoutEditor.saving ? "disabled" : ""}>取消</button>
      <button class="action-button" type="button" data-action="save-layout-editor" ${changedIds.length && !ui.layoutEditor.saving ? "" : "disabled"}>${ui.layoutEditor.saving ? "正在保存…" : "保存坐标"}</button>`;
    refs.actionConsole.innerHTML = consoleLayout(
      "DEVELOPER · MAP LAYOUT",
      `调整 ${escapeHtml(ui.snapshot.state.game_map.name)} 城市坐标`,
      "拖动地图上的城市圆圈；此模式不会推进游戏。",
      body,
      actions,
    );
  }

  function renderAuctionStart(request, player) {
    const starts = request.legal_actions.filter((action) => action.action_type === "auction_start");
    const selected = starts.find((action) => Number(action.payload.plant_price) === ui.auctionPlant);
    if (selected && !ui.bidValue) ui.bidValue = Number(selected.payload.min_bid);
    const passAllowed = request.legal_actions.some((action) => action.action_type === "auction_pass");
    const body = selected
      ? `<div class="choice-summary">已选择 <strong>#${selected.payload.plant_price} 发电厂</strong><br />合法出价 ${selected.payload.min_bid}–${selected.payload.max_bid} Elektro</div>`
      : '<div class="choice-summary"><strong>请点击当前市场中的发电厂。</strong><br />发光边框表示可以发起竞拍。</div>';
    const controls = selected
      ? numberControl("bid", ui.bidValue, selected.payload.min_bid, selected.payload.max_bid)
      : "";
    const actions = `${controls}<button class="action-button" type="button" data-action="start-auction" ${selected ? "" : "disabled"}>发起竞拍</button>${passAllowed ? '<button class="action-button secondary" type="button" data-action="auction-pass">跳过本轮</button>' : ""}`;
    refs.actionConsole.innerHTML = consoleLayout("PHASE 01 · AUCTION", `${escapeHtml(player.name)} 选择电厂`, "市场下排为当前可竞拍电厂。", body, actions);
  }

  function renderAuctionBid(request, player) {
    const bidAction = request.legal_actions.find((action) => action.action_type === "auction_bid");
    const auction = ui.snapshot.state.auction_state;
    if (bidAction && (ui.bidValue < bidAction.payload.min_bid || ui.bidValue > bidAction.payload.max_bid)) {
      ui.bidValue = Number(bidAction.payload.min_bid);
    }
    const body = `<div class="choice-summary">竞拍电厂 <strong>#${auction.active_plant_price}</strong> · 当前最高价 <strong>${auction.current_bid}</strong><br />最高出价者：${escapeHtml(playerName(auction.highest_bidder_id))}</div>`;
    const controls = bidAction
      ? numberControl("bid", ui.bidValue, bidAction.payload.min_bid, bidAction.payload.max_bid)
      : "";
    const actions = `${controls}<button class="action-button" type="button" data-action="submit-bid" ${bidAction ? "" : "disabled"}>提交出价</button><button class="action-button secondary" type="button" data-action="auction-pass">退出竞拍</button>`;
    refs.actionConsole.innerHTML = consoleLayout("PHASE 01 · ACTIVE BID", `${escapeHtml(player.name)} 出价`, "加价或退出当前竞拍。", body, actions);
  }

  function renderResourceAction(request, player) {
    const action = request.legal_actions.find((candidate) => candidate.action_type === "buy_resource");
    const resource = action?.payload.resource || request.metadata.resource;
    const max = Number(action?.payload.max_affordable_units || 0);
    ui.resourceAmount = clamp(ui.resourceAmount, 0, max);
    const cost = quoteResourceCost(resource, ui.resourceAmount);
    const body = `<div class="choice-summary">购买 <strong>${RESOURCE_LABELS[resource]}</strong> · 最多 ${max} 个<br />本次费用：<strong>${cost} Elektro</strong> · 现金 ${player.elektro}</div>`;
    const actions = `${numberControl("resource", ui.resourceAmount, 0, max)}<button class="action-button" type="button" data-action="buy-resource">${ui.resourceAmount ? "确认购买" : "跳过此资源"}</button>`;
    refs.actionConsole.innerHTML = consoleLayout("PHASE 02 · RESOURCES", `${escapeHtml(player.name)} 采购燃料`, "资源会从最低价格格开始依次购买。", body, actions);
  }

  function renderBuildAction(request, player) {
    const legal = new Set(
      request.legal_actions
        .filter((action) => action.action_type === "build_city")
        .map((action) => action.payload.city_id),
    );
    ui.buildCities = ui.buildCities.filter((cityId) => legal.has(cityId));
    const chips = ui.buildCities
      .map((cityId) => `<span class="city-chip">${escapeHtml(cityName(cityId))}</span>`)
      .join("");
    const quote = ui.buildQuote?.valid
      ? `<strong>${ui.buildQuote.cost} Elektro</strong> · ${escapeHtml(ui.buildQuote.message)}`
      : ui.buildCities.length
        ? escapeHtml(ui.buildQuote?.message || "正在核算建设费用…")
        : "点击地图上发光的城市建立建设方案。";
    const body = `<div class="quote-box">${quote}${chips ? `<div class="city-chips">${chips}</div>` : ""}</div>`;
    const actions = `<button class="action-button secondary" type="button" data-action="clear-build" ${ui.buildCities.length ? "" : "disabled"}>清空</button><button class="action-button" type="button" data-action="commit-build" ${ui.buildQuote?.valid ? "" : "disabled"}>确认建设</button><button class="action-button secondary" type="button" data-action="finish-building">结束建设</button>`;
    refs.actionConsole.innerHTML = consoleLayout("PHASE 03 · EXPANSION", `${escapeHtml(player.name)} 建设网络`, `现金 ${player.elektro} · 可用房屋 ${player.houses_in_supply}`, body, actions);
  }

  function renderBureaucracyAction(request, player) {
    initializeRuns(player);
    const totals = resourceTotals(player.resource_storage);
    const cards = player.power_plants
      .filter((plant) => !plant.is_step_3_placeholder)
      .sort((a, b) => a.price - b.price)
      .map((plant) => {
        const run = ui.runs[plant.price];
        const fuel = plant.is_ecological
          ? "无需燃料"
          : plant.is_hybrid
            ? `混合燃料 × ${plant.resource_cost}`
            : `${RESOURCE_LABELS[plant.resource_types[0]]} × ${plant.resource_cost}`;
        const hybrid = plant.is_hybrid
          ? `<label class="hybrid-mix">煤
              <input type="number" min="0" max="${plant.resource_cost}" value="${run.coal}" data-hybrid-price="${plant.price}" ${run.selected ? "" : "disabled"} />
              · 油 ${plant.resource_cost - run.coal}
            </label>`
          : "";
        return `
          <label class="run-option ${run.selected ? "selected" : ""}">
            <input type="checkbox" data-run-price="${plant.price}" ${run.selected ? "checked" : ""} />
            <span><b>#${plant.price} → ${plant.output_cities} 城</b><span>${fuel}</span></span>
            ${hybrid}
          </label>`;
      })
      .join("");
    const body = `<div class="run-grid">${cards || "暂无可运行电厂"}</div>`;
    const stock = RESOURCE_ORDER.map((resource) => `${RESOURCE_LABELS[resource]} ${totals[resource]}`).join(" · ");
    const actions = `<button class="action-button" type="button" data-action="run-plants">提交运行方案</button><button class="action-button secondary" type="button" data-action="skip-bureaucracy">不发电</button>`;
    refs.actionConsole.innerHTML = consoleLayout(
      "PHASE 04 · BUREAUCRACY",
      `${escapeHtml(player.name)} 供电`,
      `库存：${stock} · 零燃料电厂优先，其余按编号从大到小补足城市需求`,
      body,
      actions,
    );
  }

  function renderPendingDecision(request, player) {
    const buttons = request.legal_actions.map((action, index) => {
      let label = action.action_type;
      if (action.action_type === "discard_power_plant") label = `弃置 #${action.payload.price} 发电厂`;
      if (action.action_type === "discard_hybrid_resources") label = `弃置煤 ${action.payload.coal || 0} · 油 ${action.payload.oil || 0}`;
      return `<button class="action-button secondary" type="button" data-action="pending-option" data-index="${index}">${escapeHtml(label)}</button>`;
    }).join("");
    refs.actionConsole.innerHTML = consoleLayout("REQUIRED DECISION", `${escapeHtml(player.name)} 必须处理`, escapeHtml(request.prompt), '<div class="choice-summary">完成此决定后才能继续游戏。</div>', buttons);
  }

  function numberControl(kind, value, min, max) {
    return `
      <div class="number-control" data-kind="${kind}" data-min="${min}" data-max="${max}">
        <button type="button" data-action="number-minus" aria-label="减一">−</button>
        <input id="${kind}-amount" data-number-input="${kind}" type="number" min="${min}" max="${max}" value="${value}" aria-label="数量或出价" />
        <button type="button" data-action="number-plus" aria-label="加一">＋</button>
      </div>`;
  }

  function renderEvents() {
    const events = ui.snapshot.events.slice(-24).reverse();
    refs.eventList.innerHTML = events.length
      ? events.map((event) => `
          <li class="${event.level === "error" ? "error" : ""}">
            <time>R${event.round_number ?? "–"} · S${event.step ?? "–"}</time>${escapeHtml(event.message)}
          </li>`).join("")
      : "<li>对局开始后，关键事件会显示在这里。</li>";
  }

  function renderWinner() {
    const winner = ui.snapshot.winner;
    refs.winnerDialog.hidden = !winner;
    if (!winner) return;
    const rows = ui.snapshot.state.players
      .slice()
      .sort((a, b) => {
        const power = (winner.powered_cities[b.player_id] || 0) - (winner.powered_cities[a.player_id] || 0);
        return power || (winner.money[b.player_id] || 0) - (winner.money[a.player_id] || 0);
      })
      .map((player) => `
        <div class="winner-row ${winner.winner_ids.includes(player.player_id) ? "winner" : ""}">
          <strong>${escapeHtml(player.name)}</strong>
          <small>供电 ${winner.powered_cities[player.player_id] || 0}</small>
          <small>现金 ${winner.money[player.player_id] || 0}</small>
          <small>城市 ${winner.connected_cities[player.player_id] || 0}</small>
        </div>`).join("");
    refs.winnerContent.innerHTML = `<div class="winner-result">${rows}</div>`;
  }

  async function handleClick(event) {
    const control = event.target.closest("[data-action]");
    if (event.target === refs.globalParametersDialog) {
      closeGlobalParameters();
      return;
    }
    if (control?.dataset.action === "show-global-parameters") {
      showGlobalParameters(control);
      return;
    }
    if (control?.dataset.action === "close-global-parameters") {
      closeGlobalParameters();
      return;
    }
    if (ui.layoutEditor.active) {
      if (!control || control.disabled) return;
      const editorAction = control.dataset.action;
      try {
        if (editorAction === "zoom-in") setZoom(ui.zoom + 0.12);
        else if (editorAction === "zoom-out") setZoom(ui.zoom - 0.12);
        else if (editorAction === "cancel-layout-editor") cancelLayoutEditor();
        else if (editorAction === "save-layout-editor") await saveLayoutEditor();
      } catch (error) {
        showToast(error.message);
      }
      return;
    }

    const plantButton = event.target.closest("[data-plant-price]");
    if (plantButton) {
      const price = Number(plantButton.dataset.plantPrice);
      const action = ui.snapshot.request?.legal_actions.find(
        (candidate) => candidate.action_type === "auction_start" && Number(candidate.payload.plant_price) === price,
      );
      if (action) {
        ui.auctionPlant = price;
        ui.bidValue = Number(action.payload.min_bid);
        renderPlantMarket();
        renderActionConsole();
      }
      return;
    }

    const cityTarget = event.target.closest("[data-city]");
    if (cityTarget?.classList.contains("clickable")) {
      await toggleBuildCity(cityTarget.dataset.city);
      return;
    }

    if (!control || control.disabled) return;
    const actionName = control.dataset.action;
    try {
      if (actionName === "zoom-in") setZoom(ui.zoom + 0.12);
      else if (actionName === "zoom-out") setZoom(ui.zoom - 0.12);
      else if (actionName === "start-layout-editor") startLayoutEditor();
      else if (actionName === "toggle-ai") toggleAi();
      else if (actionName === "return-launcher") showLauncher(false);
      else if (actionName === "number-minus" || actionName === "number-plus") adjustNumber(control, actionName.endsWith("plus") ? 1 : -1);
      else if (actionName === "start-auction") await submitCurrentIntent("auction_start", { plant_price: ui.auctionPlant, bid: ui.bidValue });
      else if (actionName === "submit-bid") await submitCurrentIntent("auction_bid", { bid: ui.bidValue });
      else if (actionName === "auction-pass") await submitCurrentIntent("auction_pass", {});
      else if (actionName === "buy-resource") {
        const resource = ui.snapshot.request.metadata.resource;
        await submitCurrentIntent("buy_resource", { resource, amount: ui.resourceAmount });
      } else if (actionName === "clear-build") {
        ui.buildCities = [];
        ui.buildQuote = null;
        renderBoard();
        renderActionConsole();
      } else if (actionName === "commit-build") {
        const cities = [...ui.buildCities];
        const cost = Number(ui.buildQuote?.cost || 0);
        await submitCurrentIntent("commit_build", { city_ids: cities }, { forceReset: true });
        if (ui.snapshot.request?.decision_type === "build_houses") {
          showToast(`已建设 ${cities.length} 座城市，支出 ${cost} Elektro；可以继续选择城市，或结束建设。`);
        }
      } else if (actionName === "finish-building") await submitCurrentIntent("finish_building", {});
      else if (actionName === "run-plants") await submitRunPlans();
      else if (actionName === "skip-bureaucracy") await submitCurrentIntent("skip_bureaucracy", {});
      else if (actionName === "pending-option") await submitPendingOption(Number(control.dataset.index));
    } catch (error) {
      showToast(error.message);
    }
  }

  function startLayoutEditor() {
    if (ui.aiWorking) {
      showToast("AI 正在提交动作，请稍后再进入坐标调整。");
      return;
    }
    const mapCities = ui.snapshot.state.game_map.cities;
    const positions = ui.snapshot.layout.cities || {};
    const missing = mapCities.filter((city) => !normalizedPoint(positions[city.id], 1, 1));
    if (missing.length) {
      showToast(`有 ${missing.length} 座城市尚无初始坐标，暂时无法进入调整模式。`);
      return;
    }
    ui.layoutEditor = {
      active: true,
      original: cloneCityPositions(positions, mapCities),
      draft: cloneCityPositions(positions, mapCities),
      changed: new Set(),
      dragging: null,
      resumeAi: !ui.aiPaused,
      saving: false,
    };
    clearTimeout(ui.aiTimer);
    ui.aiPaused = true;
    renderHeader();
    renderBoard();
    renderActionConsole();
    showToast("坐标调整已开启：拖动任意城市圆圈。");
  }

  function cancelLayoutEditor() {
    const changedCount = ui.layoutEditor.changed.size;
    finishLayoutEditor();
    showToast(changedCount ? "已取消，坐标修改未保存。" : "已退出坐标调整。");
  }

  async function saveLayoutEditor() {
    const changedCount = ui.layoutEditor.changed.size;
    if (!changedCount || ui.layoutEditor.saving) return;
    const map = ui.snapshot.state.game_map;
    const confirmed = window.confirm(
      `确认保存 ${map.name} 的 ${changedCount} 项坐标修改？\n\n坐标将写回 Web 地图布局配置文件。`,
    );
    if (!confirmed) return;
    ui.layoutEditor.saving = true;
    renderActionConsole();
    try {
      const result = await api("/api/layout/cities", {
        method: "POST",
        body: {
          map_id: map.id,
          confirmed: true,
          cities: Object.fromEntries(
            map.cities.map((city) => {
              const point = ui.layoutEditor.draft[city.id];
              return [city.id, { x: point.x, y: point.y }];
            }),
          ),
        },
      });
      ui.snapshot.layout = result.layout;
      finishLayoutEditor();
      showToast(`已保存 ${changedCount} 项坐标修改到 ${result.config_file}。`);
    } catch (error) {
      ui.layoutEditor.saving = false;
      renderActionConsole();
      throw error;
    }
  }

  function finishLayoutEditor() {
    const resumeAi = ui.layoutEditor.resumeAi;
    resetLayoutEditorState();
    ui.aiPaused = !resumeAi;
    if (!refs.gameShell.hidden) renderAll();
  }

  function resetLayoutEditorState() {
    ui.layoutEditor = {
      active: false,
      original: null,
      draft: null,
      changed: new Set(),
      dragging: null,
      resumeAi: false,
      saving: false,
    };
  }

  function cloneCityPositions(positions, mapCities) {
    return Object.fromEntries(
      mapCities.map((city) => {
        const position = positions[city.id];
        return [
          city.id,
          {
            ...position,
            x: Number(position.x),
            y: Number(position.y),
            house_slots: position.house_slots?.map((slot) => ({ ...slot })),
          },
        ];
      }),
    );
  }

  function handlePointerDown(event) {
    if (!ui.layoutEditor.active || ui.layoutEditor.saving || event.button !== 0) return;
    const marker = event.target.closest("[data-layout-city]");
    if (!marker) return;
    const svg = marker.closest("svg");
    const pointer = pointerToMapPoint(event, svg);
    const cityId = marker.dataset.layoutCity;
    const size = ui.snapshot.layout.size;
    const center = normalizedPoint(ui.layoutEditor.draft[cityId], size.width, size.height);
    if (!pointer || !center) return;
    event.preventDefault();
    marker.setPointerCapture?.(event.pointerId);
    marker.classList.add("dragging");
    ui.layoutEditor.dragging = {
      cityId,
      marker,
      pointerId: event.pointerId,
      offsetX: center.x - pointer.x,
      offsetY: center.y - pointer.y,
    };
  }

  function handlePointerMove(event) {
    const drag = ui.layoutEditor.dragging;
    if (!ui.layoutEditor.active || !drag || drag.pointerId !== event.pointerId) return;
    const svg = drag.marker.closest("svg");
    const pointer = pointerToMapPoint(event, svg);
    if (!pointer) return;
    event.preventDefault();
    updateDraftPosition(drag.cityId, pointer.x + drag.offsetX, pointer.y + drag.offsetY);
    syncCoordinateMarker(drag.marker, drag.cityId);
  }

  function handlePointerUp(event) {
    const drag = ui.layoutEditor.dragging;
    if (!drag || drag.pointerId !== event.pointerId) return;
    drag.marker.releasePointerCapture?.(event.pointerId);
    drag.marker.classList.remove("dragging");
    ui.layoutEditor.dragging = null;
    renderBoard();
    renderActionConsole();
  }

  function pointerToMapPoint(event, svg) {
    if (!svg) return null;
    const matrix = svg.getScreenCTM?.();
    if (matrix && svg.createSVGPoint) {
      const point = svg.createSVGPoint();
      point.x = event.clientX;
      point.y = event.clientY;
      const transformed = point.matrixTransform(matrix.inverse());
      return { x: transformed.x, y: transformed.y };
    }
    const bounds = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    if (!bounds.width || !bounds.height || !viewBox.width || !viewBox.height) return null;
    return {
      x: ((event.clientX - bounds.left) / bounds.width) * viewBox.width + viewBox.x,
      y: ((event.clientY - bounds.top) / bounds.height) * viewBox.height + viewBox.y,
    };
  }

  function updateDraftPosition(cityId, pixelX, pixelY) {
    const size = ui.snapshot.layout.size;
    const position = ui.layoutEditor.draft[cityId];
    position.x = Number((clamp(pixelX, 0, size.width) / size.width).toFixed(6));
    position.y = Number((clamp(pixelY, 0, size.height) / size.height).toFixed(6));
    const original = ui.layoutEditor.original[cityId];
    const changed = Math.abs(position.x - original.x) > 0.0000005 || Math.abs(position.y - original.y) > 0.0000005;
    if (changed) ui.layoutEditor.changed.add(cityId);
    else ui.layoutEditor.changed.delete(cityId);
  }

  function syncCoordinateMarker(marker, cityId) {
    const size = ui.snapshot.layout.size;
    const position = ui.layoutEditor.draft[cityId];
    const point = normalizedPoint(position, size.width, size.height);
    const coordinate = `${position.x.toFixed(5)}, ${position.y.toFixed(5)}`;
    marker.setAttribute("transform", `translate(${point.x} ${point.y})`);
    marker.setAttribute("aria-label", `拖动 ${cityName(cityId)}，当前坐标 ${coordinate}`);
    const value = marker.querySelector(".city-coordinate-value");
    if (value) value.textContent = coordinate;
    const title = marker.querySelector("title");
    if (title) title.textContent = `${cityName(cityId)} · ${coordinate}`;
  }

  function handleChange(event) {
    if (event.target.matches("[data-run-price]")) {
      const price = Number(event.target.dataset.runPrice);
      ui.runs[price].selected = event.target.checked;
      renderActionConsole();
      return;
    }
    if (event.target.matches("[data-hybrid-price]")) {
      const price = Number(event.target.dataset.hybridPrice);
      const plant = playerById(ui.snapshot.request.player_id).power_plants.find((candidate) => candidate.price === price);
      ui.runs[price].coal = clamp(Number(event.target.value || 0), 0, plant.resource_cost);
      renderActionConsole();
    }
  }

  function handleInput(event) {
    if (!event.target.matches("[data-number-input]")) return;
    const container = event.target.closest(".number-control");
    const value = clamp(Number(event.target.value || 0), Number(container.dataset.min), Number(container.dataset.max));
    if (event.target.dataset.numberInput === "bid") ui.bidValue = value;
    else ui.resourceAmount = value;
  }

  async function handleKeydown(event) {
    if (event.key === "Escape" && !refs.globalParametersDialog.hidden) {
      event.preventDefault();
      closeGlobalParameters();
      return;
    }
    if (ui.layoutEditor.active) {
      if (event.key === "Escape" && !ui.layoutEditor.saving) {
        event.preventDefault();
        cancelLayoutEditor();
        return;
      }
      const marker = event.target.closest?.("[data-layout-city]");
      const deltaByKey = {
        ArrowLeft: [-1, 0],
        ArrowRight: [1, 0],
        ArrowUp: [0, -1],
        ArrowDown: [0, 1],
      };
      const delta = deltaByKey[event.key];
      if (!marker || !delta || ui.layoutEditor.saving) return;
      event.preventDefault();
      const cityId = marker.dataset.layoutCity;
      const size = ui.snapshot.layout.size;
      const point = normalizedPoint(ui.layoutEditor.draft[cityId], size.width, size.height);
      const step = event.shiftKey ? 10 : 1;
      updateDraftPosition(cityId, point.x + delta[0] * step, point.y + delta[1] * step);
      syncCoordinateMarker(marker, cityId);
      renderActionConsole();
      return;
    }
    if ((event.key === "Enter" || event.key === " ") && event.target.matches("[data-city].clickable")) {
      event.preventDefault();
      await toggleBuildCity(event.target.dataset.city);
    }
  }

  function adjustNumber(control, delta) {
    const container = control.closest(".number-control");
    const input = container.querySelector("[data-number-input]");
    const value = clamp(Number(input.value || 0) + delta, Number(container.dataset.min), Number(container.dataset.max));
    if (input.dataset.numberInput === "bid") ui.bidValue = value;
    else ui.resourceAmount = value;
    renderActionConsole();
  }

  async function toggleBuildCity(cityId) {
    if (ui.buildCities.includes(cityId)) {
      ui.buildCities = ui.buildCities.filter((candidate) => candidate !== cityId);
    } else {
      ui.buildCities.push(cityId);
    }
    ui.buildQuote = null;
    renderBoard();
    renderActionConsole();
    if (!ui.buildCities.length) return;
    try {
      const quote = await api("/api/quote-build", {
        method: "POST",
        body: { player_id: ui.snapshot.request.player_id, city_ids: ui.buildCities },
      });
      ui.buildQuote = quote;
      renderActionConsole();
    } catch (error) {
      ui.buildQuote = { valid: false, message: error.message };
      renderActionConsole();
    }
  }

  async function submitRunPlans() {
    const player = playerById(ui.snapshot.request.player_id);
    const plans = player.power_plants
      .filter((plant) => ui.runs[plant.price]?.selected)
      .map((plant) => {
        const resourceMix = {};
        if (plant.is_hybrid) {
          const coal = ui.runs[plant.price].coal;
          const oil = plant.resource_cost - coal;
          if (coal) resourceMix.coal = coal;
          if (oil) resourceMix.oil = oil;
        }
        return { plant_price: plant.price, resource_mix: resourceMix };
      });
    await submitCurrentIntent("run_plants", { plans });
  }

  async function submitPendingOption(index) {
    const action = ui.snapshot.request.legal_actions[index];
    if (!action) return;
    if (action.action_type === "discard_power_plant") {
      await submitCurrentIntent("discard_power_plant", { plant_price: action.payload.price });
      return;
    }
    if (action.action_type === "discard_hybrid_resources") {
      await submitCurrentIntent("discard_hybrid_resources", {
        coal: Number(action.payload.coal || 0),
        oil: Number(action.payload.oil || 0),
      });
    }
  }

  async function submitCurrentIntent(intentType, payload, options = {}) {
    const request = ui.snapshot.request;
    if (!request) return;
    const result = await api("/api/intent", {
      method: "POST",
      body: { intent_type: intentType, player_id: request.player_id, payload },
      acceptErrorPayload: true,
    });
    if (result.has_game) {
      setSnapshot(result, Boolean(options.forceReset && !result.error));
      renderAll();
    }
    if (result.error) throw new Error(result.error);
  }

  function initializeRuns(player) {
    if (Object.keys(ui.runs).length) return;
    ui.runs = greedyRunDefaults(player);
  }

  function greedyRunDefaults(player) {
    const remaining = resourceTotals(player.resource_storage);
    const defaults = {};
    const plants = player.power_plants
      .filter((plant) => !plant.is_step_3_placeholder)
      .slice();
    const targetCities = player.network_city_ids.length;
    let selectedOutput = 0;

    for (const plant of plants) {
      defaults[plant.price] = { selected: false, coal: 0 };
    }

    for (const plant of plants.filter((plant) => plant.resource_cost === 0)) {
      defaults[plant.price].selected = true;
      selectedOutput += plant.output_cities;
    }

    const fueledPlants = plants
      .filter((plant) => plant.resource_cost > 0)
      .sort((left, right) => right.price - left.price);

    for (const plant of fueledPlants) {
      let selected = false;
      const coal = plant.is_hybrid ? Math.min(plant.resource_cost, remaining.coal) : 0;
      defaults[plant.price].coal = coal;
      if (selectedOutput >= targetCities) continue;

      if (plant.is_hybrid) {
        selected = remaining.coal + remaining.oil >= plant.resource_cost;
        if (selected) {
          const oil = plant.resource_cost - coal;
          remaining.coal -= coal;
          remaining.oil -= oil;
        }
      } else {
        const resource = plant.resource_types[0];
        selected = remaining[resource] >= plant.resource_cost;
        if (selected) remaining[resource] -= plant.resource_cost;
      }
      defaults[plant.price].selected = selected;
      if (selected) selectedOutput += plant.output_cities;
    }
    return defaults;
  }

  function toggleAi() {
    if (ui.layoutEditor.active) return;
    ui.aiPaused = !ui.aiPaused;
    clearTimeout(ui.aiTimer);
    renderHeader();
    renderActionConsole();
    if (!ui.aiPaused) scheduleAi();
  }

  function scheduleAi() {
    clearTimeout(ui.aiTimer);
    if (ui.layoutEditor.active || !ui.snapshot?.needs_ai_advance || ui.aiPaused || ui.aiWorking || ui.snapshot.winner) return;
    const delay = ui.snapshot.request ? 720 : 80;
    ui.aiTimer = setTimeout(advanceAi, delay);
  }

  async function advanceAi() {
    if (ui.aiWorking || ui.aiPaused) return;
    ui.aiWorking = true;
    try {
      const result = await api("/api/advance", { method: "POST", body: {} });
      setSnapshot(result);
      renderAll();
    } catch (error) {
      ui.aiPaused = true;
      showToast(`AI 推进失败：${error.message}`);
      renderHeader();
      renderActionConsole();
    } finally {
      ui.aiWorking = false;
      scheduleAi();
    }
  }

  function setZoom(value) {
    ui.zoom = clamp(value, 0.58, 1.84);
    refs.boardCanvas.style.width = `${Math.round(ui.zoom * 100)}%`;
    refs.zoomOutput.textContent = `${Math.round(ui.zoom * 100)}%`;
  }

  function quoteResourceCost(resource, amount) {
    const bands = ui.snapshot.state.resource_market.market[resource] || {};
    const prices = Object.entries(bands)
      .sort(([a], [b]) => Number(a) - Number(b))
      .flatMap(([price, count]) => Array.from({ length: Number(count) }, () => Number(price)));
    return prices.slice(0, amount).reduce((sum, price) => sum + price, 0);
  }

  function resourceTotals(storage) {
    return {
      coal: Number(storage.coal || 0) + Number(storage.hybrid_coal || 0),
      oil: Number(storage.oil || 0) + Number(storage.hybrid_oil || 0),
      garbage: Number(storage.garbage || 0),
      uranium: Number(storage.uranium || 0),
    };
  }

  function normalizedPoint(point, width, height) {
    if (!point || point.x == null || point.y == null) return null;
    return { x: Number(point.x) * width, y: Number(point.y) * height };
  }

  function playerById(playerId) {
    return ui.snapshot?.state.players.find((player) => player.player_id === playerId);
  }

  function playerName(playerId) {
    return playerById(playerId)?.name || playerId || "—";
  }

  function cityName(cityId) {
    return ui.snapshot?.state.game_map.cities.find((city) => city.id === cityId)?.name || cityId;
  }

  function playerColor(color) {
    return PLAYER_COLORS[color] || color || "#445154";
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  async function api(path, options = {}) {
    const fetchOptions = { method: options.method || "GET", headers: {} };
    if (options.body !== undefined) {
      fetchOptions.headers["Content-Type"] = "application/json";
      fetchOptions.body = JSON.stringify(options.body);
    }
    const response = await fetch(path, fetchOptions);
    let payload;
    try {
      payload = await response.json();
    } catch (_error) {
      throw new Error(`服务返回了无效响应（${response.status}）`);
    }
    if (!response.ok && !options.acceptErrorPayload) {
      throw new Error(payload.error || `请求失败（${response.status}）`);
    }
    return payload;
  }

  function showToast(message) {
    clearTimeout(ui.toastTimer);
    refs.toast.textContent = message;
    refs.toast.hidden = false;
    ui.toastTimer = setTimeout(() => {
      refs.toast.hidden = true;
    }, 4800);
  }
})();
