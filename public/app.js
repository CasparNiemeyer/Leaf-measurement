const $ = (id) => document.getElementById(id);

const els = {
  video: $('cameraVideo'),
  full: $('fullCanvas'),
  cropped: $('croppedCanvas'),
  draw: $('drawCanvas'),
  result: $('resultCanvas'),
  mask: $('maskCanvas'),
  drawStage: $('drawStage'),
  cameraStatus: $('cameraStatus'),
  cameraSelect: $('cameraSelect'),
  refreshCameras: $('refreshCameras'),
  startCamera: $('startCamera'),
  stopCamera: $('stopCamera'),
  freeze: $('freezeButton'),
  fullscreen: $('fullscreenDraw'),
  clearMasks: $('clearMasks'),
  brushSize: $('brushSize'),
  archive: $('archiveButton'),
  archiveStatus: $('archiveStatus'),
  csv: $('downloadCsv'),
  authStatus: $('authStatus'),
  authForms: $('authForms'),
  logout: $('logoutButton'),
  loginForm: $('loginForm'),
  registerForm: $('registerForm'),
  resetRequestForm: $('resetRequestForm'),
  resetPasswordPanel: $('resetPasswordPanel'),
  resetPasswordButton: $('resetPasswordButton'),
  platformMessage: $('platformMessage'),
  projectArea: $('projectArea'),
  projectSelect: $('projectSelect'),
  createProjectForm: $('createProjectForm'),
  joinProjectForm: $('joinProjectForm'),
  projectMeasurementCount: $('projectMeasurementCount'),
  projectLatestMeasurement: $('projectLatestMeasurement'),
  projectInviteLink: $('projectInviteLink'),
  copyInviteLink: $('copyInviteLink'),
  projectCsvLink: $('projectCsvLink'),
  memberList: $('memberList'),
  refreshMeasurements: $('refreshMeasurements'),
  measurementList: $('measurementList'),
  measurementDetail: $('measurementDetail'),
  area: $('areaValue'),
  convex: $('convexValue'),
  damage: $('damageValue'),
  status: $('statusValue'),
};

const inputs = {
  language: $('languageSelect'),
  darkMode: $('darkMode'),
  cameraMode: $('cameraMode'),
  upload: $('imageUpload'),
  physWidth: $('physWidth'),
  physHeight: $('physHeight'),
  digWidth: $('digWidth'),
  analysisFps: $('analysisFps'),
  kernelSize: $('kernelSize'),
  hueMin: $('hueRange'),
  hueMax: $('hueRangeMax'),
  satMin: $('satRange'),
  satMax: $('satRangeMax'),
  valMin: $('valRange'),
  valMax: $('valRangeMax'),
  hueLabel: $('hueLabel'),
  satLabel: $('satLabel'),
  valLabel: $('valLabel'),
  manualEnabled: $('manualEnabled'),
  autoEdgeDamage: $('autoEdgeDamage'),
  limitToLeaf: $('limitToLeaf'),
  shrinkMask: $('shrinkMask'),
  showAutoOnCrop: $('showAutoOnCrop'),
  drawMarkers: $('drawMarkers'),
  drawBoundary: $('drawBoundary'),
  drawContours: $('drawContours'),
  drawHull: $('drawHull'),
};

const state = {
  stream: null,
  sourceCanvas: document.createElement('canvas'),
  frozenCanvas: document.createElement('canvas'),
  uploadedImage: null,
  frozen: false,
  tool: 'damage',
  drawing: false,
  lastPoint: null,
  lastProcess: 0,
  lastDetection: null,
  masks: { width: 0, damage: null, correct: null, exclude: null },
  lastMeasurement: null,
  user: null,
  projects: [],
  activeProjectId: localStorage.getItem('leafActiveProjectId') || '',
  measurements: [],
  resetToken: new URLSearchParams(location.search).get('reset') || '',
};

const FROZEN_ANALYSIS_INTERVAL_MS = 220;

const translations = {
  de: {
    title: 'Leaf Measurement',
    cameraReady: 'Kamera: bereit',
    defaultCamera: 'Standardkamera',
    findCameras: 'Kameras suchen',
    startCamera: 'Kamera starten',
    stop: 'Stop',
    fullscreen: 'Vollbild',
    cropped: 'Cropped',
    result: 'Ergebnis',
    damageMask: 'Schadensmaske',
    freeze: 'Freeze',
    live: 'Live',
    brushSize: 'Größe',
    measurements: 'Messwerte',
    greenArea: 'Grüne Fläche',
    convexHull: 'Convex Hull',
    damage: 'Schaden',
    status: 'Status',
    noImage: 'Noch kein Bild',
    archive: 'Archivieren',
    downloadCsv: 'CSV herunterladen',
    settings: 'Einstellungen',
    language: 'Sprache',
    darkMode: 'Dunkelmodus',
    cameraMode: 'Kamera/Bildmodus',
    image: 'Bild',
    physWidth: 'Physische Breite cm',
    physHeight: 'Physische Höhe cm',
    digWidth: 'Digitale Auflösung px',
    analysisFps: 'Analyse-FPS',
    kernel: 'Kernelgröße',
    hsv: 'HSV Live-Grenzen',
    masks: 'Masken',
    manual: 'Manuelle Masken einrechnen',
    edge: 'Convex-Randschäden',
    limit: 'Zeichnen auf Blattmaske begrenzen',
    shrink: 'Maske schrumpfen px',
    showAuto: 'Auto-Schäden auf Cropped',
    display: 'Anzeige',
    drawMarkers: 'Marker anzeigen',
    drawBoundary: 'Fläche markieren',
    drawContours: 'Umrandung',
    drawHull: 'Convex Hull',
  },
  en: {
    title: 'Leaf Measurement',
    cameraReady: 'Camera: ready',
    defaultCamera: 'Default camera',
    findCameras: 'Find cameras',
    startCamera: 'Start camera',
    stop: 'Stop',
    fullscreen: 'Fullscreen',
    cropped: 'Cropped',
    result: 'Result',
    damageMask: 'Damage mask',
    freeze: 'Freeze',
    live: 'Live',
    brushSize: 'Size',
    measurements: 'Measurements',
    greenArea: 'Green area',
    convexHull: 'Convex hull',
    damage: 'Damage',
    status: 'Status',
    noImage: 'No image yet',
    archive: 'Archive',
    downloadCsv: 'Download CSV',
    settings: 'Settings',
    language: 'Language',
    darkMode: 'Dark mode',
    cameraMode: 'Camera/image mode',
    image: 'Image',
    physWidth: 'Physical width cm',
    physHeight: 'Physical height cm',
    digWidth: 'Digital resolution px',
    analysisFps: 'Analysis FPS',
    kernel: 'Kernel size',
    hsv: 'HSV live limits',
    masks: 'Masks',
    manual: 'Include manual masks',
    edge: 'Convex edge damage',
    limit: 'Limit drawing to leaf mask',
    shrink: 'Shrink mask px',
    showAuto: 'Show auto damage on cropped',
    display: 'Display',
    drawMarkers: 'Show markers',
    drawBoundary: 'Show boundary',
    drawContours: 'Outline',
    drawHull: 'Convex hull',
  },
};

function tr(key) {
  return translations[inputs.language?.value || 'de']?.[key] || translations.de[key] || key;
}

function setLabelText(label, text) {
  if (!label?.childNodes?.length) return;
  label.childNodes[0].nodeValue = `${text} `;
}

function applyLanguage() {
  document.documentElement.lang = inputs.language.value;
  document.title = tr('title');
  document.querySelector('h1').textContent = tr('title');
  els.cameraStatus.textContent = tr('cameraReady');
  els.cameraSelect.options[0].textContent = tr('defaultCamera');
  els.refreshCameras.textContent = tr('findCameras');
  els.startCamera.textContent = tr('startCamera');
  els.stopCamera.textContent = tr('stop');
  els.fullscreen.title = tr('fullscreen');
  els.freeze.textContent = state.frozen ? tr('live') : tr('freeze');
  document.querySelector('.tool-row .inline').childNodes[0].nodeValue = `${tr('brushSize')} `;
  document.querySelectorAll('.panel h2')[0].textContent = tr('cropped');
  document.querySelectorAll('.panel h2')[1].textContent = tr('result');
  document.querySelectorAll('.panel h2')[2].textContent = tr('damageMask');
  document.querySelectorAll('.panel h2')[3].textContent = tr('measurements');
  document.querySelectorAll('.metrics dt')[0].textContent = tr('greenArea');
  document.querySelectorAll('.metrics dt')[1].textContent = tr('convexHull');
  document.querySelectorAll('.metrics dt')[2].textContent = tr('damage');
  document.querySelectorAll('.metrics dt')[3].textContent = tr('status');
  if (!state.lastMeasurement) els.status.textContent = tr('noImage');
  els.archive.textContent = tr('archive');
  els.csv.textContent = tr('downloadCsv');
  document.querySelector('.settings h2').textContent = tr('settings');
  const fields = document.querySelectorAll('.settings .field');
  setLabelText(fields[0], tr('language'));
  setLabelText(fields[1], tr('image'));
  setLabelText(fields[2], tr('physWidth'));
  setLabelText(fields[3], tr('physHeight'));
  setLabelText(fields[4], tr('digWidth'));
  setLabelText(fields[5], tr('analysisFps'));
  setLabelText(fields[6], tr('kernel'));
  setLabelText(fields[7], tr('shrink'));
  const checks = document.querySelectorAll('.settings .check');
  setLabelText(checks[0], tr('darkMode'));
  setLabelText(checks[1], tr('cameraMode'));
  setLabelText(checks[2], tr('manual'));
  setLabelText(checks[3], tr('edge'));
  setLabelText(checks[4], tr('limit'));
  setLabelText(checks[5], tr('showAuto'));
  setLabelText(checks[6], tr('drawMarkers'));
  setLabelText(checks[7], tr('drawBoundary'));
  setLabelText(checks[8], tr('drawContours'));
  setLabelText(checks[9], tr('drawHull'));
  document.querySelectorAll('.settings h3')[0].textContent = tr('hsv');
  document.querySelectorAll('.settings h3')[1].textContent = tr('masks');
  document.querySelectorAll('.settings h3')[2].textContent = tr('display');
}

function numberValue(input, fallback) {
  const value = Number(input.value);
  return Number.isFinite(value) ? value : fallback;
}

function settings() {
  let hueMin = numberValue(inputs.hueMin, 0);
  let hueMax = numberValue(inputs.hueMax, 179);
  let satMin = numberValue(inputs.satMin, 30);
  let satMax = numberValue(inputs.satMax, 255);
  let valMin = numberValue(inputs.valMin, 40);
  let valMax = numberValue(inputs.valMax, 255);
  if (hueMin > hueMax) [hueMin, hueMax] = [hueMax, hueMin];
  if (satMin > satMax) [satMin, satMax] = [satMax, satMin];
  if (valMin > valMax) [valMin, valMax] = [valMax, valMin];
  return {
    physWidth: Math.max(0.001, numberValue(inputs.physWidth, 13.4)),
    physHeight: Math.max(0.001, numberValue(inputs.physHeight, 13.4)),
    digWidth: Math.max(200, Math.min(1400, Math.round(numberValue(inputs.digWidth, 700)))),
    analysisFps: Math.max(1, Math.min(30, numberValue(inputs.analysisFps, 8))),
    kernelSize: Math.max(1, Math.min(31, Math.round(numberValue(inputs.kernelSize, 5)))),
    lowerHsv: [hueMin, satMin, valMin],
    upperHsv: [hueMax, satMax, valMax],
    manualEnabled: inputs.manualEnabled.checked,
    autoEdgeDamage: inputs.autoEdgeDamage.checked,
    limitToLeaf: inputs.limitToLeaf.checked,
    shrinkMask: Math.max(0, Math.min(80, Math.round(numberValue(inputs.shrinkMask, 0)))),
    showAutoOnCrop: inputs.showAutoOnCrop.checked,
    drawMarkers: inputs.drawMarkers.checked,
    drawBoundary: inputs.drawBoundary.checked,
    drawContours: inputs.drawContours.checked,
    drawHull: inputs.drawHull.checked,
  };
}

function updateRangeLabels() {
  const s = settings();
  inputs.hueLabel.textContent = `${s.lowerHsv[0]} - ${s.upperHsv[0]}`;
  inputs.satLabel.textContent = `${s.lowerHsv[1]} - ${s.upperHsv[1]}`;
  inputs.valLabel.textContent = `${s.lowerHsv[2]} - ${s.upperHsv[2]}`;
  updateDualRange(inputs.hueMin, inputs.hueMax);
  updateDualRange(inputs.satMin, inputs.satMax);
  updateDualRange(inputs.valMin, inputs.valMax);
}

function updateDualRange(minInput, maxInput) {
  const wrapper = minInput.closest('.dual-range');
  if (!wrapper) return;
  const min = Number(minInput.min || wrapper.dataset.min || 0);
  const max = Number(minInput.max || wrapper.dataset.max || 100);
  const low = Math.min(Number(minInput.value), Number(maxInput.value));
  const high = Math.max(Number(minInput.value), Number(maxInput.value));
  const span = Math.max(1, max - min);
  wrapper.style.setProperty('--range-start', `${((low - min) / span) * 100}%`);
  wrapper.style.setProperty('--range-end', `${((high - min) / span) * 100}%`);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    credentials: 'same-origin',
    headers: { 'content-type': 'application/json', ...(options.headers || {}) },
    ...options,
    body: options.body && typeof options.body !== 'string' ? JSON.stringify(options.body) : options.body,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function setPlatformMessage(message) {
  els.platformMessage.textContent = message || '';
}

function formatDate(iso) {
  if (!iso) return '-';
  return new Intl.DateTimeFormat(inputs.language.value === 'en' ? 'en' : 'de', {
    dateStyle: 'short',
    timeStyle: 'short',
  }).format(new Date(iso));
}

function activeProject() {
  return state.projects.find((project) => project.id === state.activeProjectId) || state.projects[0] || null;
}

function extractInviteToken(value) {
  const text = String(value || '').trim();
  if (!text) return '';
  try {
    const url = new URL(text, location.origin);
    return url.searchParams.get('join') || text;
  } catch {
    return text;
  }
}

async function refreshAccount() {
  const data = await api('/api/me');
  state.user = data.user;
  state.projects = data.projects || [];
  if (state.activeProjectId && !state.projects.some((project) => project.id === state.activeProjectId)) {
    state.activeProjectId = '';
  }
  if (!state.activeProjectId && state.projects[0]) state.activeProjectId = state.projects[0].id;
  if (state.activeProjectId) localStorage.setItem('leafActiveProjectId', state.activeProjectId);
  renderAccount();
  if (state.user && activeProject()) await refreshMeasurements();
}

function renderAccount() {
  const loggedIn = Boolean(state.user);
  els.authForms.classList.toggle('hidden', loggedIn);
  els.logout.classList.toggle('hidden', !loggedIn);
  els.projectArea.classList.toggle('hidden', !loggedIn);
  els.authStatus.textContent = loggedIn
    ? `${state.user.email}${state.user.verified ? '' : ' (E-Mail unbestätigt)'}`
    : 'Nicht angemeldet';
  els.archive.disabled = !loggedIn || !activeProject();
  if (!loggedIn) return;

  els.projectSelect.replaceChildren();
  for (const project of state.projects) {
    els.projectSelect.add(new Option(project.name, project.id, false, project.id === state.activeProjectId));
  }
  renderProjectDashboard();
}

function renderProjectDashboard() {
  const project = activeProject();
  if (!project) {
    els.projectMeasurementCount.textContent = '-';
    els.projectLatestMeasurement.textContent = '-';
    els.projectInviteLink.value = '';
    els.projectCsvLink.href = '#';
    els.csv.href = '#';
    els.memberList.replaceChildren();
    return;
  }
  els.projectMeasurementCount.textContent = String(project.measurementCount ?? 0);
  els.projectLatestMeasurement.textContent = formatDate(project.latestMeasurementAt);
  els.projectInviteLink.value = project.inviteUrl || '';
  els.projectCsvLink.href = `/api/projects/${project.id}/archive.csv`;
  els.csv.href = `/api/projects/${project.id}/archive.csv`;
  els.memberList.replaceChildren(...(project.members || []).map((member) => {
    const li = document.createElement('li');
    const name = document.createElement('span');
    name.textContent = member.email;
    const role = document.createElement('span');
    role.className = 'member-role';
    role.textContent = member.role === 'owner' ? 'Owner' : 'Member';
    li.append(name, role);
    return li;
  }));
}

function renderMeasurementList() {
  els.measurementList.replaceChildren();
  if (!state.measurements.length) {
    const li = document.createElement('li');
    li.className = 'hint';
    li.textContent = 'Noch keine Messungen im Projekt.';
    els.measurementList.append(li);
    els.measurementDetail.innerHTML = '<p class="hint">Archiviere eine Messung oder wähle einen anderen Projektkontext.</p>';
    return;
  }
  for (const measurement of state.measurements) {
    const li = document.createElement('li');
    const button = document.createElement('button');
    button.type = 'button';
    button.innerHTML = `
      <span class="measurement-title">
        <span>${formatDate(measurement.createdAt)}</span>
        <span>${Number(measurement.damagePercent || 0).toFixed(1)}%</span>
      </span>
      <span class="measurement-meta">${measurement.userEmail || ''} · ${Number(measurement.greenArea || 0).toFixed(3)} cm² Blatt · ${Number(measurement.damageArea || 0).toFixed(3)} cm² Schaden</span>
    `;
    button.addEventListener('click', () => selectMeasurement(measurement.id, button));
    li.append(button);
    els.measurementList.append(li);
  }
}

async function refreshMeasurements() {
  const project = activeProject();
  if (!project) return;
  const data = await api(`/api/projects/${project.id}/measurements`);
  state.measurements = data.measurements || [];
  renderMeasurementList();
}

async function selectMeasurement(measurementId, button) {
  const project = activeProject();
  if (!project) return;
  document.querySelectorAll('.measurement-list button').forEach((item) => item.classList.toggle('active', item === button));
  const data = await api(`/api/projects/${project.id}/measurements/${measurementId}`);
  renderMeasurementDetail(data.measurement);
}

function renderMeasurementDetail(measurement) {
  const stats = [
    ['Grüne Fläche', `${Number(measurement.greenArea || 0).toFixed(3)} cm²`],
    ['Convex Hull', `${Number(measurement.convexArea || 0).toFixed(3)} cm²`],
    ['Schaden', `${Number(measurement.damageArea || 0).toFixed(3)} cm² (${Number(measurement.damagePercent || 0).toFixed(1)}%)`],
    ['Status', measurement.status || '-'],
    ['Erstellt von', measurement.userEmail || '-'],
    ['Zeitpunkt', formatDate(measurement.createdAt)],
  ];
  const imageLabels = { full: 'Fullframe', cropped: 'Cropped', result: 'Ergebnis', mask: 'Schadensmaske' };
  els.measurementDetail.innerHTML = `
    <h3>Messung ${formatDate(measurement.createdAt)}</h3>
    <dl class="metrics compact">
      ${stats.map(([key, value]) => `<div><dt>${key}</dt><dd>${value}</dd></div>`).join('')}
    </dl>
    <div class="measurement-images">
      ${Object.entries(measurement.images || {}).filter(([, src]) => src).map(([key, src]) => `
        <div class="measurement-image-card">
          <strong>${imageLabels[key] || key}</strong>
          <img src="${src}" alt="${imageLabels[key] || key}">
        </div>
      `).join('')}
    </div>
  `;
}

async function handleUrlTokens() {
  const params = new URLSearchParams(location.search);
  const verifyToken = params.get('verify');
  const joinToken = params.get('join');
  if (verifyToken) {
    await api('/api/auth/verify', { method: 'POST', body: { token: verifyToken } });
    setPlatformMessage('E-Mail bestätigt. Du kannst dich jetzt einloggen.');
    history.replaceState(null, '', location.pathname);
  }
  if (state.resetToken) {
    els.resetPasswordPanel.classList.remove('hidden');
    setPlatformMessage('Bitte setze ein neues Passwort.');
  }
  if (joinToken) {
    await refreshAccount();
    if (state.user) {
      await api('/api/projects/join', { method: 'POST', body: { token: joinToken } });
      setPlatformMessage('Projekt beigetreten.');
      history.replaceState(null, '', location.pathname);
      await refreshAccount();
    } else {
      setPlatformMessage('Bitte einloggen oder registrieren, um dem Projekt beizutreten.');
    }
  }
}

function setStatus(text) {
  const prefix = inputs.language.value === 'en' ? 'Camera' : 'Kamera';
  els.cameraStatus.textContent = `${prefix}: ${text}`;
}

function fitCanvas(canvas, width, height) {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function ensureMasks(width) {
  if (state.masks.width === width && state.masks.damage) return;
  state.masks.width = width;
  state.masks.damage = new Uint8Array(width * width);
  state.masks.correct = new Uint8Array(width * width);
  state.masks.exclude = new Uint8Array(width * width);
}

async function refreshCameras() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === 'videoinput');
  els.cameraSelect.replaceChildren(new Option('Standardkamera', ''));
  cameras.forEach((camera, index) => {
    els.cameraSelect.add(new Option(camera.label || `Kamera ${index + 1}`, camera.deviceId));
  });
}

async function startCamera() {
  stopCamera(false);
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('nicht verfügbar');
    return;
  }
  try {
    setStatus('startet');
    const deviceId = els.cameraSelect.value;
    const constraints = {
      video: deviceId
        ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
        : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: 'environment' } },
      audio: false,
    };
    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
    els.video.srcObject = state.stream;
    await els.video.play();
    state.frozen = false;
    els.freeze.textContent = 'Freeze';
    setStatus('aktiv');
    await refreshCameras();
  } catch (error) {
    setStatus(`Fehler (${error.name || error.message})`);
  }
}

function stopCamera(markStopped = true) {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  els.video.srcObject = null;
  if (markStopped) setStatus('gestoppt');
}

function captureSource() {
  if (state.frozen && state.frozenCanvas.width) return state.frozenCanvas;
  if (inputs.cameraMode.checked && els.video.videoWidth && els.video.videoHeight && state.stream) {
    fitCanvas(state.sourceCanvas, els.video.videoWidth, els.video.videoHeight);
    state.sourceCanvas.getContext('2d').drawImage(els.video, 0, 0);
    return state.sourceCanvas;
  }
  if (state.uploadedImage) {
    fitCanvas(state.sourceCanvas, state.uploadedImage.naturalWidth, state.uploadedImage.naturalHeight);
    state.sourceCanvas.getContext('2d').drawImage(state.uploadedImage, 0, 0);
    return state.sourceCanvas;
  }
  return null;
}

function luminance(r, g, b) {
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function rgbToHsv(r, g, b) {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;
  let h = 0;
  if (delta !== 0) {
    if (max === rn) h = 60 * (((gn - bn) / delta) % 6);
    else if (max === gn) h = 60 * ((bn - rn) / delta + 2);
    else h = 60 * ((rn - gn) / delta + 4);
  }
  if (h < 0) h += 360;
  const s = max === 0 ? 0 : delta / max;
  return [Math.round(h / 2), Math.round(s * 255), Math.round(max * 255)];
}

function connectedComponents(binary, width, height, minArea = 1) {
  const visited = new Uint8Array(width * height);
  const stack = new Int32Array(width * height);
  const components = [];
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const start = y * width + x;
      if (!binary[start] || visited[start]) continue;
      let top = 0;
      let area = 0;
      let sumX = 0;
      let sumY = 0;
      let minX = x;
      let minY = y;
      let maxX = x;
      let maxY = y;
      stack[top++] = start;
      visited[start] = 1;
      while (top) {
        const index = stack[--top];
        const px = index % width;
        const py = Math.floor(index / width);
        area += 1;
        sumX += px;
        sumY += py;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
        const neighbors = [index - 1, index + 1, index - width, index + width];
        for (const next of neighbors) {
          if (next < 0 || next >= binary.length || visited[next] || !binary[next]) continue;
          const nx = next % width;
          if ((next === index - 1 && nx !== px - 1) || (next === index + 1 && nx !== px + 1)) continue;
          visited[next] = 1;
          stack[top++] = next;
        }
      }
      if (area >= minArea) {
        components.push({ area, minX, minY, maxX, maxY, cx: sumX / area, cy: sumY / area });
      }
    }
  }
  return components;
}

function orderPoints(points) {
  const ordered = new Array(4);
  ordered[0] = points.reduce((best, p) => (p.x + p.y < best.x + best.y ? p : best));
  ordered[2] = points.reduce((best, p) => (p.x + p.y > best.x + best.y ? p : best));
  ordered[1] = points.reduce((best, p) => (p.x - p.y > best.x - best.y ? p : best));
  ordered[3] = points.reduce((best, p) => (p.x - p.y < best.x - best.y ? p : best));
  return ordered;
}

function polygonArea(points) {
  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % points.length];
    area += a.x * b.y - b.x * a.y;
  }
  return Math.abs(area) / 2;
}

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function isReasonableQuad(points, width, height) {
  if (new Set(points.map((p) => `${Math.round(p.x)},${Math.round(p.y)}`)).size < 4) return false;
  const marginX = width * 0.04;
  const marginY = height * 0.04;
  if (points.some((p) => p.x < -marginX || p.x > width + marginX || p.y < -marginY || p.y > height + marginY)) return false;
  const area = polygonArea(points);
  const frameArea = width * height;
  if (area < frameArea * 0.03 || area > frameArea * 0.98) return false;
  const sides = [distance(points[0], points[1]), distance(points[1], points[2]), distance(points[2], points[3]), distance(points[3], points[0])];
  if (Math.min(...sides) < Math.min(width, height) * 0.08) return false;
  if (Math.max(...sides) / Math.max(1, Math.min(...sides)) > 4.2) return false;
  return true;
}

function detectMarkers(source) {
  const maxSide = 720;
  const scale = Math.min(1, maxSide / Math.max(source.width, source.height));
  const w = Math.max(1, Math.round(source.width * scale));
  const h = Math.max(1, Math.round(source.height * scale));
  const temp = document.createElement('canvas');
  temp.width = w;
  temp.height = h;
  const ctx = temp.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(source, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  const binary = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < data.length; i += 4, p += 1) {
    binary[p] = luminance(data[i], data[i + 1], data[i + 2]) < 85 ? 1 : 0;
  }
  const minArea = Math.max(40, Math.round(w * h * 0.00012));
  const maxArea = w * h * 0.12;
  const candidates = connectedComponents(binary, w, h, minArea)
    .filter((c) => {
      const bw = c.maxX - c.minX + 1;
      const bh = c.maxY - c.minY + 1;
      const aspect = bw / Math.max(1, bh);
      const fill = c.area / Math.max(1, bw * bh);
      return c.area <= maxArea && bw > 12 && bh > 12 && aspect > 0.45 && aspect < 2.2 && fill > 0.18 && fill < 0.9;
    })
    .sort((a, b) => b.area - a.area)
    .slice(0, 24)
    .map((c) => ({
      x: c.cx / scale,
      y: c.cy / scale,
      area: c.area / (scale * scale),
      box: { x: c.minX / scale, y: c.minY / scale, w: (c.maxX - c.minX + 1) / scale, h: (c.maxY - c.minY + 1) / scale },
    }));

  let best = null;
  for (let a = 0; a < candidates.length - 3; a += 1) {
    for (let b = a + 1; b < candidates.length - 2; b += 1) {
      for (let c = b + 1; c < candidates.length - 1; c += 1) {
        for (let d = c + 1; d < candidates.length; d += 1) {
          const group = [candidates[a], candidates[b], candidates[c], candidates[d]];
          const ordered = orderPoints(group);
          if (!isReasonableQuad(ordered, source.width, source.height)) continue;
          const areas = group.map((p) => p.area);
          const areaRatio = Math.max(...areas) / Math.max(1, Math.min(...areas));
          if (areaRatio > 9) continue;
          const sides = [distance(ordered[0], ordered[1]), distance(ordered[1], ordered[2]), distance(ordered[2], ordered[3]), distance(ordered[3], ordered[0])];
          const score = Math.log(areaRatio) + Math.abs(Math.log(sides[0] / Math.max(1, sides[2]))) + Math.abs(Math.log(sides[1] / Math.max(1, sides[3]))) - polygonArea(ordered) / (source.width * source.height) * 0.2;
          if (!best || score < best.score) best = { score, points: ordered, group };
        }
      }
    }
  }
  return best || { points: [], group: candidates };
}

function solveLinearSystem(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const div = a[col][col] || 1e-12;
    for (let j = col; j <= n; j += 1) a[col][j] /= div;
    for (let row = 0; row < n; row += 1) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= n; j += 1) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map((row) => row[n]);
}

function homography(from, to) {
  const matrix = [];
  const vector = [];
  for (let i = 0; i < 4; i += 1) {
    const x = from[i].x;
    const y = from[i].y;
    const u = to[i].x;
    const v = to[i].y;
    matrix.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    vector.push(u);
    matrix.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    vector.push(v);
  }
  const h = solveLinearSystem(matrix, vector);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

function warpPerspective(source, points, size) {
  const srcCtx = source.getContext('2d', { willReadFrequently: true });
  const src = srcCtx.getImageData(0, 0, source.width, source.height);
  const dest = new ImageData(size, size);
  const h = homography(
    [{ x: 0, y: 0 }, { x: size - 1, y: 0 }, { x: size - 1, y: size - 1 }, { x: 0, y: size - 1 }],
    points,
  );
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const denom = h[6] * x + h[7] * y + h[8];
      const sx = Math.round((h[0] * x + h[1] * y + h[2]) / denom);
      const sy = Math.round((h[3] * x + h[4] * y + h[5]) / denom);
      const di = (y * size + x) * 4;
      if (sx >= 0 && sx < source.width && sy >= 0 && sy < source.height) {
        const si = (sy * source.width + sx) * 4;
        dest.data[di] = src.data[si];
        dest.data[di + 1] = src.data[si + 1];
        dest.data[di + 2] = src.data[si + 2];
        dest.data[di + 3] = 255;
      } else {
        dest.data[di + 3] = 255;
      }
    }
  }
  return dest;
}

function erode(mask, width, height, radius) {
  if (radius <= 0) return mask.slice();
  const out = new Uint8Array(mask.length);
  for (let y = radius; y < height - radius; y += 1) {
    for (let x = radius; x < width - radius; x += 1) {
      let ok = 1;
      for (let yy = -radius; yy <= radius && ok; yy += 1) {
        for (let xx = -radius; xx <= radius; xx += 1) {
          if (!mask[(y + yy) * width + x + xx]) {
            ok = 0;
            break;
          }
        }
      }
      out[y * width + x] = ok;
    }
  }
  return out;
}

function dilate(mask, width, height, radius) {
  if (radius <= 0) return mask.slice();
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let ok = 0;
      for (let yy = -radius; yy <= radius && !ok; yy += 1) {
        const py = y + yy;
        if (py < 0 || py >= height) continue;
        for (let xx = -radius; xx <= radius; xx += 1) {
          const px = x + xx;
          if (px >= 0 && px < width && mask[py * width + px]) {
            ok = 1;
            break;
          }
        }
      }
      out[y * width + x] = ok;
    }
  }
  return out;
}

function morphOpen(mask, width, height, kernelSize) {
  const radius = Math.max(0, Math.floor(kernelSize / 2));
  return dilate(erode(mask, width, height, radius), width, height, radius);
}

function largestComponentMask(mask, width, height) {
  const visited = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  let bestPixels = [];
  for (let start = 0; start < mask.length; start += 1) {
    if (!mask[start] || visited[start]) continue;
    let top = 0;
    const pixels = [];
    stack[top++] = start;
    visited[start] = 1;
    while (top) {
      const index = stack[--top];
      pixels.push(index);
      const x = index % width;
      const neighbors = [index - 1, index + 1, index - width, index + width];
      for (const next of neighbors) {
        if (next < 0 || next >= mask.length || visited[next] || !mask[next]) continue;
        const nx = next % width;
        if ((next === index - 1 && nx !== x - 1) || (next === index + 1 && nx !== x + 1)) continue;
        visited[next] = 1;
        stack[top++] = next;
      }
    }
    if (pixels.length > bestPixels.length) bestPixels = pixels;
  }
  const out = new Uint8Array(mask.length);
  bestPixels.forEach((index) => {
    out[index] = 1;
  });
  return out;
}

function fillHoles(mask, width, height) {
  const exterior = new Uint8Array(mask.length);
  const stack = [];
  const push = (index) => {
    if (index >= 0 && index < mask.length && !mask[index] && !exterior[index]) {
      exterior[index] = 1;
      stack.push(index);
    }
  };
  for (let x = 0; x < width; x += 1) {
    push(x);
    push((height - 1) * width + x);
  }
  for (let y = 0; y < height; y += 1) {
    push(y * width);
    push(y * width + width - 1);
  }
  while (stack.length) {
    const index = stack.pop();
    const x = index % width;
    const neighbors = [index - 1, index + 1, index - width, index + width];
    for (const next of neighbors) {
      if (next < 0 || next >= mask.length) continue;
      const nx = next % width;
      if ((next === index - 1 && nx !== x - 1) || (next === index + 1 && nx !== x + 1)) continue;
      push(next);
    }
  }
  const filled = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i += 1) filled[i] = mask[i] || !exterior[i] ? 1 : 0;
  return filled;
}

function convexHull(points) {
  if (points.length <= 3) return points;
  const sorted = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i -= 1) {
    const p = sorted[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

function polygonMask(points, width, height) {
  const out = new Uint8Array(width * height);
  if (points.length < 3) return out;
  for (let y = 0; y < height; y += 1) {
    const intersections = [];
    for (let i = 0; i < points.length; i += 1) {
      const a = points[i];
      const b = points[(i + 1) % points.length];
      if ((a.y <= y && b.y > y) || (b.y <= y && a.y > y)) {
        intersections.push(a.x + ((y - a.y) * (b.x - a.x)) / (b.y - a.y));
      }
    }
    intersections.sort((a, b) => a - b);
    for (let i = 0; i < intersections.length; i += 2) {
      const start = Math.max(0, Math.ceil(intersections[i]));
      const end = Math.min(width - 1, Math.floor(intersections[i + 1] ?? intersections[i]));
      for (let x = start; x <= end; x += 1) out[y * width + x] = 1;
    }
  }
  return out;
}

function hullFromMask(mask, width, height) {
  const points = [];
  const step = Math.max(1, Math.floor(width / 260));
  for (let y = 1; y < height - 1; y += step) {
    for (let x = 1; x < width - 1; x += step) {
      const i = y * width + x;
      if (!mask[i]) continue;
      if (!mask[i - 1] || !mask[i + 1] || !mask[i - width] || !mask[i + width]) points.push({ x, y });
    }
  }
  return convexHull(points);
}

function count(mask) {
  let total = 0;
  for (let i = 0; i < mask.length; i += 1) total += mask[i] ? 1 : 0;
  return total;
}

function applyManualMasks(baseGreen, contourLimit, hullMask, width, s) {
  let correct = state.masks.correct;
  let damage = state.masks.damage;
  let exclude = state.masks.exclude;
  if (!s.manualEnabled) {
    correct = new Uint8Array(baseGreen.length);
    damage = new Uint8Array(baseGreen.length);
    exclude = new Uint8Array(baseGreen.length);
  } else if (s.limitToLeaf) {
    const limit = s.shrinkMask > 0 ? erode(contourLimit, width, width, s.shrinkMask) : contourLimit;
    correct = correct.map((value, i) => (value && limit[i] ? 1 : 0));
    damage = damage.map((value, i) => (value && limit[i] ? 1 : 0));
  }
  const green = new Uint8Array(baseGreen.length);
  for (let i = 0; i < green.length; i += 1) green[i] = (baseGreen[i] || correct[i]) && !exclude[i] ? 1 : 0;
  return { green, correct, damage, exclude };
}

function edgeOverlay(image, mask, color) {
  const width = image.width;
  const data = image.data;
  for (let y = 1; y < width - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const i = y * width + x;
      if (!mask[i]) continue;
      if (!mask[i - 1] || !mask[i + 1] || !mask[i - width] || !mask[i + width]) {
        const p = i * 4;
        data[p] = color[0];
        data[p + 1] = color[1];
        data[p + 2] = color[2];
      }
    }
  }
}

function drawProcessed(cropped, masks, s, measurement) {
  fitCanvas(els.cropped, cropped.width, cropped.height);
  fitCanvas(els.result, cropped.width, cropped.height);
  fitCanvas(els.mask, cropped.width, cropped.height);
  fitCanvas(els.draw, cropped.width, cropped.height);
  els.cropped.getContext('2d').putImageData(cropped, 0, 0);
  if (s.showAutoOnCrop) {
    const ctx = els.cropped.getContext('2d');
    const overlay = ctx.getImageData(0, 0, cropped.width, cropped.height);
    for (let i = 0; i < masks.autoDamage.length; i += 1) {
      if (!masks.autoDamage[i]) continue;
      const p = i * 4;
      overlay.data[p] = 220;
      overlay.data[p + 1] = 38;
      overlay.data[p + 2] = 38;
    }
    ctx.putImageData(overlay, 0, 0);
  }

  const result = new ImageData(new Uint8ClampedArray(cropped.data), cropped.width, cropped.height);
  for (let i = 0; i < masks.damage.length; i += 1) {
    if (!masks.damage[i]) continue;
    const p = i * 4;
    result.data[p] = Math.round(result.data[p] * 0.35 + 220 * 0.65);
    result.data[p + 1] = Math.round(result.data[p + 1] * 0.35 + 38 * 0.65);
    result.data[p + 2] = Math.round(result.data[p + 2] * 0.35 + 38 * 0.65);
  }
  if (s.drawHull) edgeOverlay(result, masks.hullMask, [37, 99, 235]);
  if (s.drawContours) edgeOverlay(result, masks.green, [34, 197, 94]);
  els.result.getContext('2d').putImageData(result, 0, 0);

  const maskImage = new ImageData(cropped.width, cropped.height);
  for (let i = 0; i < masks.damage.length; i += 1) {
    const p = i * 4;
    if (masks.damage[i]) {
      maskImage.data[p] = 220;
      maskImage.data[p + 1] = 38;
      maskImage.data[p + 2] = 38;
    }
    maskImage.data[p + 3] = 255;
  }
  els.mask.getContext('2d').putImageData(maskImage, 0, 0);
  drawManualOverlay();

  els.area.textContent = measurement.area == null ? '-' : `${measurement.area.toFixed(3)} cm²`;
  els.convex.textContent = measurement.convexArea == null ? '-' : `${measurement.convexArea.toFixed(3)} cm²`;
  els.damage.textContent = measurement.damageArea == null ? '-' : `${measurement.damageArea.toFixed(3)} cm² (${measurement.damagePercent.toFixed(1)}%)`;
  els.status.textContent = measurement.status;
  state.lastMeasurement = measurement;
}

function drawFullPreview(source) {
  fitCanvas(els.full, source.width, source.height);
  const fullCtx = els.full.getContext('2d');
  fullCtx.drawImage(source, 0, 0);
  const detection = state.lastDetection;
  if (!detection) return;
  const s = settings();
  if (s.drawMarkers) {
    fullCtx.strokeStyle = '#16a34a';
    fullCtx.lineWidth = Math.max(2, source.width / 700);
    for (const candidate of detection.group || []) {
      if (!candidate.box) continue;
      fullCtx.strokeRect(candidate.box.x, candidate.box.y, candidate.box.w, candidate.box.h);
    }
  }
  if (s.drawBoundary && detection.points?.length) {
    fullCtx.strokeStyle = '#f59e0b';
    fullCtx.lineWidth = Math.max(2, source.width / 850);
    fullCtx.beginPath();
    detection.points.forEach((point, index) => {
      if (index === 0) fullCtx.moveTo(point.x, point.y);
      else fullCtx.lineTo(point.x, point.y);
    });
    fullCtx.closePath();
    fullCtx.stroke();
  }
}

function processFrame(source) {
  const s = settings();
  ensureMasks(s.digWidth);
  const detection = detectMarkers(source);
  state.lastDetection = detection;
  if (!detection.points?.length) {
    els.status.textContent = 'Suche Marker';
    state.lastMeasurement = { status: 'Suche Marker', markers: 0 };
    return;
  }

  const cropped = warpPerspective(source, detection.points, s.digWidth);
  const greenRaw = new Uint8Array(s.digWidth * s.digWidth);
  for (let i = 0, p = 0; i < cropped.data.length; i += 4, p += 1) {
    const hsv = rgbToHsv(cropped.data[i], cropped.data[i + 1], cropped.data[i + 2]);
    greenRaw[p] = hsv[0] >= s.lowerHsv[0] && hsv[0] <= s.upperHsv[0] && hsv[1] >= s.lowerHsv[1] && hsv[1] <= s.upperHsv[1] && hsv[2] >= s.lowerHsv[2] && hsv[2] <= s.upperHsv[2] ? 1 : 0;
  }
  const opened = morphOpen(greenRaw, s.digWidth, s.digWidth, s.kernelSize);
  const baseGreen = largestComponentMask(opened, s.digWidth, s.digWidth);
  if (!count(baseGreen)) {
    drawProcessed(cropped, {
      green: baseGreen,
      hullMask: new Uint8Array(baseGreen.length),
      damage: new Uint8Array(baseGreen.length),
      autoDamage: new Uint8Array(baseGreen.length),
    }, s, { status: 'Kein Blatt erkannt', area: null, convexArea: null, damageArea: null, damagePercent: 0, markers: 4 });
    return;
  }

  const contourLimit = fillHoles(baseGreen, s.digWidth, s.digWidth);
  const preliminaryHull = polygonMask(hullFromMask(baseGreen, s.digWidth, s.digWidth), s.digWidth, s.digWidth);
  const manual = applyManualMasks(baseGreen, contourLimit, preliminaryHull, s.digWidth, s);
  const green = manual.green;
  const contourFilled = fillHoles(green, s.digWidth, s.digWidth);
  const hullPoints = hullFromMask(green, s.digWidth, s.digWidth);
  const hullMask = polygonMask(hullPoints, s.digWidth, s.digWidth);
  const autoDamage = new Uint8Array(green.length);
  const damage = new Uint8Array(green.length);
  for (let i = 0; i < green.length; i += 1) {
    const internal = contourFilled[i] && !green[i];
    const edge = hullMask[i] && !contourFilled[i];
    autoDamage[i] = internal || (s.autoEdgeDamage && edge) ? 1 : 0;
    const manualDamage = s.manualEnabled && manual.damage[i] && hullMask[i];
    damage[i] = (autoDamage[i] || manualDamage) && !manual.correct[i] && !manual.exclude[i] ? 1 : 0;
  }
  const pixelArea = (s.physWidth / s.digWidth) * (s.physHeight / s.digWidth);
  const convexArea = count(hullMask) * pixelArea;
  const measurement = {
    status: 'OK',
    area: count(green) * pixelArea,
    convexArea,
    damageArea: count(damage) * pixelArea,
    damagePercent: convexArea > 0 ? (count(damage) * pixelArea / convexArea) * 100 : 0,
    markers: 4,
  };
  drawProcessed(cropped, { green, hullMask, damage, autoDamage }, s, measurement);
}

function drawManualOverlay() {
  if (!els.draw.width) return;
  const width = els.draw.width;
  const image = new ImageData(width, width);
  const colors = {
    damage: [255, 80, 30, 150],
    correct: [34, 197, 94, 135],
    exclude: [139, 92, 246, 135],
  };
  for (let i = 0; i < width * width; i += 1) {
    let color = null;
    if (state.masks.damage?.[i]) color = colors.damage;
    if (state.masks.correct?.[i]) color = colors.correct;
    if (state.masks.exclude?.[i]) color = colors.exclude;
    if (!color) continue;
    const p = i * 4;
    image.data[p] = color[0];
    image.data[p + 1] = color[1];
    image.data[p + 2] = color[2];
    image.data[p + 3] = color[3];
  }
  els.draw.getContext('2d').putImageData(image, 0, 0);
}

function syncDrawCanvasSize() {
  const width = state.masks.width || settings().digWidth;
  fitCanvas(els.draw, width, width);
}

function paintCircle(mask, x, y, radius, value) {
  const width = state.masks.width;
  const r2 = radius * radius;
  const minX = Math.max(0, Math.floor(x - radius));
  const maxX = Math.min(width - 1, Math.ceil(x + radius));
  const minY = Math.max(0, Math.floor(y - radius));
  const maxY = Math.min(width - 1, Math.ceil(y + radius));
  for (let yy = minY; yy <= maxY; yy += 1) {
    for (let xx = minX; xx <= maxX; xx += 1) {
      if ((xx - x) ** 2 + (yy - y) ** 2 <= r2) mask[yy * width + xx] = value;
    }
  }
}

function drawLine(from, to) {
  ensureMasks(settings().digWidth);
  syncDrawCanvasSize();
  const width = state.masks.width;
  const brush = Math.max(1, Number(els.brushSize.value) || 18) / 2;
  const steps = Math.max(1, Math.ceil(Math.hypot(to.x - from.x, to.y - from.y) / Math.max(1, brush * 0.65)));
  for (let step = 0; step <= steps; step += 1) {
    const t = step / steps;
    const x = from.x + (to.x - from.x) * t;
    const y = from.y + (to.y - from.y) * t;
    if (state.tool === 'eraser') {
      paintCircle(state.masks.damage, x, y, brush, 0);
      paintCircle(state.masks.correct, x, y, brush, 0);
      paintCircle(state.masks.exclude, x, y, brush, 0);
    } else {
      paintCircle(state.masks[state.tool], x, y, brush, 1);
      for (const other of ['damage', 'correct', 'exclude']) {
        if (other !== state.tool) paintCircle(state.masks[other], x, y, brush, 0);
      }
    }
  }
  drawManualOverlay();
  state.lastProcess = 0;
}

function pointerToCrop(event) {
  syncDrawCanvasSize();
  const rect = els.draw.getBoundingClientRect();
  const width = state.masks.width || settings().digWidth;
  return {
    x: Math.max(0, Math.min(width - 1, ((event.clientX - rect.left) / rect.width) * width)),
    y: Math.max(0, Math.min(width - 1, ((event.clientY - rect.top) / rect.height) * width)),
  };
}

function loop(now) {
  const source = captureSource();
  if (source) drawFullPreview(source);
  const interval = state.frozen ? FROZEN_ANALYSIS_INTERVAL_MS : 1000 / settings().analysisFps;
  if (source && now - state.lastProcess >= interval) {
    state.lastProcess = now;
    processFrame(source);
  }
  requestAnimationFrame(loop);
}

function toggleFreeze() {
  if (state.frozen) {
    state.frozen = false;
    els.freeze.textContent = tr('freeze');
    if (inputs.cameraMode.checked) startCamera();
    return;
  }
  const source = captureSource();
  if (!source) return;
  fitCanvas(state.frozenCanvas, source.width, source.height);
  state.frozenCanvas.getContext('2d').drawImage(source, 0, 0);
  state.frozen = true;
  els.freeze.textContent = tr('live');
  stopCamera(false);
  setStatus('eingefroren');
}

async function archiveCurrent() {
  if (!state.lastMeasurement || state.lastMeasurement.area == null) {
    els.archiveStatus.textContent = 'Kein Messbild zum Archivieren vorhanden';
    return;
  }
  const project = activeProject();
  if (!state.user || !project) {
    els.archiveStatus.textContent = 'Bitte zuerst einloggen und ein Projekt wählen.';
    return;
  }
  const body = {
    projectId: project.id,
    measurement: state.lastMeasurement,
    settings: settings(),
    images: {
      full: els.full.toDataURL('image/png'),
      cropped: els.cropped.toDataURL('image/png'),
      result: els.result.toDataURL('image/png'),
      mask: els.mask.toDataURL('image/png'),
    },
  };
  try {
    const result = await api('/api/archive', { method: 'POST', body });
    els.archiveStatus.textContent = result.ok ? 'Archiviert' : `Fehler: ${result.error || 'unbekannt'}`;
    await refreshAccount();
  } catch (error) {
    els.archiveStatus.textContent = `Fehler: ${error.message}`;
  }
}

function setupEvents() {
  els.loginForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      await api('/api/auth/login', {
        method: 'POST',
        body: {
          email: $('loginEmail').value,
          password: $('loginPassword').value,
        },
      });
      setPlatformMessage('Eingeloggt.');
      await refreshAccount();
    } catch (error) {
      setPlatformMessage(`Login fehlgeschlagen: ${error.message}`);
    }
  });
  els.registerForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      await api('/api/auth/register', {
        method: 'POST',
        body: {
          email: $('registerEmail').value,
          password: $('registerPassword').value,
        },
      });
      setPlatformMessage('Account erstellt. Bitte E-Mail-Link bestätigen. Lokal liegt die Mail in data/mail-outbox.jsonl.');
    } catch (error) {
      setPlatformMessage(`Registrierung fehlgeschlagen: ${error.message}`);
    }
  });
  els.resetRequestForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      await api('/api/auth/forgot-password', {
        method: 'POST',
        body: { email: $('resetEmail').value },
      });
      setPlatformMessage('Falls der Account existiert, wurde ein Reset-Link erzeugt. Lokal liegt er in data/mail-outbox.jsonl.');
    } catch (error) {
      setPlatformMessage(`Reset fehlgeschlagen: ${error.message}`);
    }
  });
  els.resetPasswordButton.addEventListener('click', async () => {
    try {
      await api('/api/auth/reset-password', {
        method: 'POST',
        body: { token: state.resetToken, password: $('newPassword').value },
      });
      state.resetToken = '';
      els.resetPasswordPanel.classList.add('hidden');
      history.replaceState(null, '', location.pathname);
      setPlatformMessage('Passwort geändert. Bitte neu einloggen.');
    } catch (error) {
      setPlatformMessage(`Passwort konnte nicht geändert werden: ${error.message}`);
    }
  });
  els.logout.addEventListener('click', async () => {
    await api('/api/auth/logout', { method: 'POST', body: {} });
    state.user = null;
    state.projects = [];
    state.measurements = [];
    renderAccount();
    renderMeasurementList();
    setPlatformMessage('Ausgeloggt.');
  });
  els.createProjectForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      const result = await api('/api/projects', {
        method: 'POST',
        body: { name: $('projectName').value },
      });
      state.activeProjectId = result.project.id;
      localStorage.setItem('leafActiveProjectId', state.activeProjectId);
      $('projectName').value = '';
      setPlatformMessage('Projekt erstellt.');
      await refreshAccount();
    } catch (error) {
      setPlatformMessage(`Projekt konnte nicht erstellt werden: ${error.message}`);
    }
  });
  els.joinProjectForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      const result = await api('/api/projects/join', {
        method: 'POST',
        body: { token: extractInviteToken($('joinLinkInput').value) },
      });
      state.activeProjectId = result.project.id;
      localStorage.setItem('leafActiveProjectId', state.activeProjectId);
      $('joinLinkInput').value = '';
      setPlatformMessage('Projekt beigetreten.');
      await refreshAccount();
    } catch (error) {
      setPlatformMessage(`Beitritt fehlgeschlagen: ${error.message}`);
    }
  });
  els.projectSelect.addEventListener('change', async () => {
    state.activeProjectId = els.projectSelect.value;
    localStorage.setItem('leafActiveProjectId', state.activeProjectId);
    renderProjectDashboard();
    await refreshMeasurements();
  });
  els.copyInviteLink.addEventListener('click', async () => {
    if (!els.projectInviteLink.value) return;
    await navigator.clipboard?.writeText(els.projectInviteLink.value);
    setPlatformMessage('Join-Link kopiert.');
  });
  els.refreshMeasurements.addEventListener('click', refreshMeasurements);
  els.refreshCameras.addEventListener('click', refreshCameras);
  els.startCamera.addEventListener('click', startCamera);
  els.stopCamera.addEventListener('click', () => stopCamera(true));
  els.freeze.addEventListener('click', toggleFreeze);
  els.archive.addEventListener('click', archiveCurrent);
  inputs.language.addEventListener('change', () => {
    localStorage.setItem('leafLanguage', inputs.language.value);
    applyLanguage();
  });
  els.fullscreen.addEventListener('click', () => els.drawStage.classList.toggle('fullscreen'));
  els.clearMasks.addEventListener('click', () => {
    ensureMasks(settings().digWidth);
    state.masks.damage.fill(0);
    state.masks.correct.fill(0);
    state.masks.exclude.fill(0);
    drawManualOverlay();
  });
  document.querySelectorAll('.tool[data-tool]').forEach((button) => {
    button.addEventListener('click', () => {
      state.tool = button.dataset.tool;
      document.querySelectorAll('.tool[data-tool]').forEach((tool) => tool.classList.toggle('active', tool === button));
    });
  });
  inputs.upload.addEventListener('change', () => {
    const file = inputs.upload.files?.[0];
    if (!file) return;
    const image = new Image();
    image.onload = () => {
      state.uploadedImage = image;
      inputs.cameraMode.checked = false;
      stopCamera(false);
      setStatus('Bild geladen');
    };
    image.src = URL.createObjectURL(file);
  });
  inputs.darkMode.addEventListener('change', () => {
    document.body.classList.toggle('dark', inputs.darkMode.checked);
    localStorage.setItem('leafDarkMode', inputs.darkMode.checked ? '1' : '0');
    document.cookie = `leafDarkMode=${inputs.darkMode.checked ? '1' : '0'}; path=/; max-age=31536000`;
  });
  for (const input of Object.values(inputs)) {
    if (input instanceof HTMLInputElement || input instanceof HTMLSelectElement) input.addEventListener('input', updateRangeLabels);
  }
  els.draw.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    state.drawing = true;
    if (els.draw.setPointerCapture) els.draw.setPointerCapture(event.pointerId);
    state.lastPoint = pointerToCrop(event);
    drawLine(state.lastPoint, state.lastPoint);
  });
  els.draw.addEventListener('pointermove', (event) => {
    if (!state.drawing) return;
    event.preventDefault();
    const point = pointerToCrop(event);
    drawLine(state.lastPoint, point);
    state.lastPoint = point;
  });
  const stopDrawing = (event) => {
    event?.preventDefault();
    state.drawing = false;
    state.lastPoint = null;
    state.lastProcess = 0;
  };
  els.draw.addEventListener('pointerup', stopDrawing);
  els.draw.addEventListener('pointercancel', stopDrawing);
}

async function init() {
  inputs.language.value = localStorage.getItem('leafLanguage') || 'de';
  inputs.darkMode.checked = localStorage.getItem('leafDarkMode') === '1';
  document.body.classList.toggle('dark', inputs.darkMode.checked);
  updateRangeLabels();
  setupEvents();
  applyLanguage();
  try {
    await handleUrlTokens();
    await refreshAccount();
  } catch (error) {
    setPlatformMessage(error.message);
    await refreshAccount().catch(() => {});
  }
  await refreshCameras();
  requestAnimationFrame(loop);
}

init();
