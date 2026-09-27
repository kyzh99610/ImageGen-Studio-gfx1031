// ImageGen Studio — desktop window.
//
// Starts the Python/Gradio backend through ImageGenApp\launch.bat (so the ZLUDA + ROCm
// runtime selection is exactly the same as a normal launch), shows a loading screen with
// the backend's progress, then shows the UI in its own window. Closing the window stops
// the backend (the whole cmd -> zluda.exe -> python tree).
'use strict';

const { app, BrowserWindow, Menu, shell, dialog, ipcMain } = require('electron');
const { spawn, execFileSync } = require('child_process');
const fs = require('fs');
const http = require('http');
const net = require('net');
const path = require('path');

const PORT_RANGE = [7860, 7880];
const READY_TIMEOUT_MS = 10 * 60 * 1000;

let win = null;
let server = null;          // child process we started (null when attached to one already running)
let serverUrl = null;
let quitting = false;
let logStream = null;
let logFile = null;
const recentLines = [];

app.setAppUserModelId('ImageGenStudio.Desktop');
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', () => {
    if (win) { if (win.isMinimized()) win.restore(); win.show(); win.focus(); }
  });
}

// ── Locate the repo (works from desktop\ and from the packaged dist\...\ImageGen Studio.exe) ──
function findRepo() {
  const starts = [__dirname, path.dirname(process.execPath), process.cwd()];
  for (const s of starts) {
    let dir = s;
    for (let i = 0; i < 8; i++) {
      if (fs.existsSync(path.join(dir, 'ImageGenApp', 'app.py'))) return dir;
      const up = path.dirname(dir);
      if (up === dir) break;
      dir = up;
    }
  }
  return null;
}

// ── Small helpers ──────────────────────────────────────────────────────────────
function httpGet(url, timeoutMs = 2000) {
  return new Promise((resolve) => {
    const req = http.get(url, { timeout: timeoutMs }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (c) => { if (body.length < 20000) body += c; });
      res.on('end', () => resolve({ status: res.statusCode, body }));
    });
    req.on('timeout', () => { req.destroy(); resolve(null); });
    req.on('error', () => resolve(null));
  });
}

function portFree(port) {
  return new Promise((resolve) => {
    const srv = net.createServer();
    srv.once('error', () => resolve(false));
    srv.once('listening', () => srv.close(() => resolve(true)));
    srv.listen(port, '127.0.0.1');
  });
}

async function findRunningInstance() {
  for (let p = PORT_RANGE[0]; p <= PORT_RANGE[1]; p++) {
    if (await portFree(p)) continue;
    const r = await httpGet(`http://127.0.0.1:${p}/`, 1500);
    if (r && r.status === 200 && r.body.includes('ImageGen Studio')) return p;
  }
  return null;
}

async function pickPort() {
  for (let p = PORT_RANGE[0]; p <= PORT_RANGE[1]; p++) if (await portFree(p)) return p;
  return null;
}

const send = (channel, payload) => {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
};

function status(text, detail) { send('status', { text, detail: detail || '' }); }

function logLine(line) {
  if (logStream) logStream.write(line + '\n');
  // Hide tqdm redraw noise and blank lines from the loading screen
  const clean = line.replace(/\x1b\[[0-9;]*[A-Za-z]/g, '').trim();
  if (!clean) return;
  recentLines.push(clean);
  if (recentLines.length > 200) recentLines.shift();
  send('log', clean);
  if (/Found AMD|ROCm runtime|GPU gfx/.test(clean)) status('Starting the AI backend…', clean);
  else if (/Launching with ZLUDA/.test(clean)) status('Loading the app (about 20 seconds)…', clean);
  else if (/Running on local URL/.test(clean)) status('Almost ready…', clean);
}

// ── Backend process ────────────────────────────────────────────────────────────
async function startBackend(repo) {
  const appDir = path.join(repo, 'ImageGenApp');
  const logDir = path.join(appDir, 'logs');
  fs.mkdirSync(logDir, { recursive: true });
  logFile = path.join(logDir, 'desktop-backend.log');
  try { if (fs.existsSync(logFile) && fs.statSync(logFile).size > 5e6) fs.renameSync(logFile, logFile + '.old'); } catch (_) {}
  logStream = fs.createWriteStream(logFile, { flags: 'a' });
  logStream.write(`\n===== ${new Date().toISOString()} desktop launch =====\n`);

  const running = await findRunningInstance();
  if (running) {
    logLine(`[desktop] ImageGen Studio is already running on port ${running} — attaching to it.`);
    return `http://127.0.0.1:${running}/`;
  }
  const port = await pickPort();
  if (!port) throw new Error(`No free port between ${PORT_RANGE[0]} and ${PORT_RANGE[1]}.`);

  const bat = path.join(appDir, 'launch.bat');
  status('Starting the AI backend…', `launch.bat --port ${port}`);
  server = spawn('cmd.exe', ['/d', '/s', '/c', `""${bat}" --port ${port} --no-browser --no-pause"`], {
    cwd: appDir,
    windowsHide: true,
    windowsVerbatimArguments: true,
    env: { ...process.env, PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8', IMAGEGEN_PARENT_PID: String(process.pid) },
  });
  let buf = '';
  const onData = (chunk) => {
    buf += chunk.toString('utf8').replace(/\r(?!\n)/g, '\n');
    const lines = buf.split(/\r?\n/);
    buf = lines.pop();
    lines.forEach(logLine);
  };
  server.stdout.on('data', onData);
  server.stderr.on('data', onData);
  server.on('exit', (code) => {
    logLine(`[desktop] backend exited with code ${code}`);
    const wasReady = !!serverUrl;
    server = null;
    if (!quitting) {
      if (wasReady) {
        dialog.showMessageBox(win, {
          type: 'error', title: 'ImageGen Studio',
          message: 'The image generation backend stopped unexpectedly.',
          detail: 'The last lines of its log are in ' + logFile + '.\n\nClick "Restart" to start it again.',
          buttons: ['Restart', 'Open log', 'Quit'], defaultId: 0, cancelId: 2,
        }).then(({ response }) => {
          if (response === 0) boot();
          else if (response === 1) { shell.openPath(logFile); }
          else { app.quit(); }
        });
      } else {
        send('failed', { code, logFile, tail: recentLines.slice(-25) });
      }
    }
  });
  return `http://127.0.0.1:${port}/`;
}

async function waitUntilReady(url) {
  const t0 = Date.now();
  while (Date.now() - t0 < READY_TIMEOUT_MS) {
    if (quitting) return false;
    const r = await httpGet(url, 2500);
    if (r && r.status === 200 && r.body.includes('ImageGen Studio')) return true;
    if (server === null && !(await findRunningInstance())) return false;   // backend died
    send('elapsed', Math.round((Date.now() - t0) / 1000));
    await new Promise((res) => setTimeout(res, 1000));
  }
  return false;
}

function stopBackend() {
  if (!server) return;
  const pid = server.pid;
  server = null;
  // Synchronous: an async taskkill can lose the race with app exit and leave the
  // backend (and its GPU memory) running. The backend also watches our PID (see app.py).
  try { execFileSync('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true, timeout: 10000, stdio: 'ignore' }); } catch (_) {}
}

// ── Window ─────────────────────────────────────────────────────────────────────
const stateFile = () => path.join(app.getPath('userData'), 'window-state.json');

function loadState() {
  try { return JSON.parse(fs.readFileSync(stateFile(), 'utf8')); } catch (_) { return {}; }
}

function saveState() {
  if (!win || win.isDestroyed()) return;
  const b = win.getNormalBounds();
  const s = { ...b, maximized: win.isMaximized(), zoom: win.webContents.getZoomFactor() };
  try { fs.writeFileSync(stateFile(), JSON.stringify(s)); } catch (_) {}
}

function createWindow() {
  const st = loadState();
  win = new BrowserWindow({
    width: st.width || 1500, height: st.height || 950, x: st.x, y: st.y,
    minWidth: 900, minHeight: 600,
    title: 'ImageGen Studio',
    icon: path.join(__dirname, 'assets', 'icon.png'),
    backgroundColor: '#1e1e2e',
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: true,
    },
  });
  if (st.maximized) win.maximize();
  win.once('ready-to-show', () => win.show());
  win.on('close', saveState);
  win.on('closed', () => { win = null; });

  // Links to other sites open in the normal browser; the app itself stays in this window.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/i.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (e, url) => {
    const local = serverUrl && url.startsWith(serverUrl);
    const splash = url.startsWith('file:');
    if (!local && !splash) { e.preventDefault(); if (/^https?:/i.test(url)) shell.openExternal(url); }
  });
  win.webContents.on('did-finish-load', () => {
    if (st.zoom && serverUrl && win.webContents.getURL().startsWith(serverUrl)) {
      win.webContents.setZoomFactor(st.zoom);
    }
  });
  return win;
}

function buildMenu(repo) {
  const appDir = repo ? path.join(repo, 'ImageGenApp') : null;
  const zoom = (d) => () => {
    if (!win) return;
    const z = d === 0 ? 1 : Math.min(3, Math.max(0.5, win.webContents.getZoomFactor() + d));
    win.webContents.setZoomFactor(z);
  };
  const template = [
    { label: 'File', submenu: [
      { label: 'Open output folder', click: () => appDir && shell.openPath(path.join(appDir, 'outputs')) },
      { label: 'Open backend log', click: () => logFile && shell.openPath(logFile) },
      { type: 'separator' },
      { label: 'Open in web browser', click: () => serverUrl && shell.openExternal(serverUrl) },
      { type: 'separator' },
      { role: 'quit', label: 'Quit' },
    ] },
    { label: 'View', submenu: [
      { label: 'Reload', accelerator: 'CmdOrCtrl+R', click: () => win && serverUrl && win.loadURL(serverUrl + '?__theme=dark') },
      { type: 'separator' },
      { label: 'Zoom in', accelerator: 'CmdOrCtrl+=', click: zoom(0.1) },
      { label: 'Zoom in', accelerator: 'CmdOrCtrl+Plus', visible: false, acceleratorWorksWhenHidden: true, click: zoom(0.1) },
      { label: 'Zoom out', accelerator: 'CmdOrCtrl+-', click: zoom(-0.1) },
      { label: 'Actual size', accelerator: 'CmdOrCtrl+0', click: zoom(0) },
      { type: 'separator' },
      { role: 'togglefullscreen', label: 'Full screen' },
      { label: 'Developer tools', accelerator: 'CmdOrCtrl+Shift+I', click: () => win && win.webContents.toggleDevTools() },
    ] },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

// ── Boot sequence ──────────────────────────────────────────────────────────────
async function boot() {
  const repo = findRepo();
  buildMenu(repo);
  serverUrl = null;
  await win.loadFile(path.join(__dirname, 'splash.html'));
  if (!repo) {
    send('failed', { code: null, logFile: '', tail: ['Could not find the ImageGenApp folder next to this program.'] });
    return;
  }
  try {
    const url = await startBackend(repo);
    status('Loading the app (about 20 seconds)…', '');
    const ok = await waitUntilReady(url);
    if (!ok) {
      if (!quitting && server) send('failed', { code: 'timeout', logFile, tail: recentLines.slice(-25) });
      return;
    }
    serverUrl = url;
    await win.loadURL(url + '?__theme=dark');
  } catch (e) {
    send('failed', { code: null, logFile, tail: [String(e && e.message || e)] });
  }
}

ipcMain.on('splash-action', (_e, action) => {
  if (action === 'retry') { stopBackend(); boot(); }
  else if (action === 'log' && logFile) shell.openPath(logFile);
  else if (action === 'quit') app.quit();
});

// Test hook: IMAGEGEN_SNAPSHOT_DIR=<dir> saves PNGs of the loading screen and the loaded UI
// (lets the window be checked without driving the desktop).
function snapshot(name) {
  const dir = process.env.IMAGEGEN_SNAPSHOT_DIR;
  if (!dir || !win || win.isDestroyed()) return;
  win.webContents.capturePage().then((img) => {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, name + '.png'), img.toPNG());
  }).catch(() => {});
}

app.whenReady().then(() => {
  createWindow();
  win.webContents.on('did-finish-load', () => {
    const onUi = serverUrl && win.webContents.getURL().startsWith(serverUrl);
    setTimeout(() => snapshot(onUi ? 'ui' : 'splash'), onUi ? 6000 : 2500);
  });
  boot();
});

app.on('before-quit', () => { quitting = true; saveState(); stopBackend(); });
app.on('window-all-closed', () => app.quit());
