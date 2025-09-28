const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Mastra server control
  startMastra: () => ipcRenderer.invoke('start-mastra'),
  stopMastra: () => ipcRenderer.invoke('stop-mastra'),
  getMastraStatus: () => ipcRenderer.invoke('get-mastra-status'),

  // Environment variables management
  saveEnvVariables: (envVars) => ipcRenderer.invoke('save-env-variables', envVars),
  getStoredEnv: () => ipcRenderer.invoke('get-stored-env'),

  // Event listeners for Mastra server logs and status
  onMastraLog: (callback) => {
    ipcRenderer.on('mastra-log', callback);
    // Return cleanup function
    return () => ipcRenderer.removeListener('mastra-log', callback);
  },

  onMastraStatus: (callback) => {
    ipcRenderer.on('mastra-status', callback);
    // Return cleanup function
    return () => ipcRenderer.removeListener('mastra-status', callback);
  },

  // Navigation helpers
  navigateToPage: (page) => ipcRenderer.invoke('navigate-to-page', page),

  // Platform info
  platform: process.platform,

  // Version info
  versions: {
    node: process.versions.node,
    chrome: process.versions.chrome,
    electron: process.versions.electron
  }
});
