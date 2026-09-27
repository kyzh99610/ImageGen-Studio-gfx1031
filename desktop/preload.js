'use strict';
// Only the loading screen uses this bridge; the Gradio UI never needs Node access.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktop', {
  on: (channel, cb) => {
    if (['status', 'log', 'elapsed', 'failed'].includes(channel)) {
      ipcRenderer.on(channel, (_e, data) => cb(data));
    }
  },
  action: (name) => ipcRenderer.send('splash-action', name),
});
