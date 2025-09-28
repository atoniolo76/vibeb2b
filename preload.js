const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  // Mastra server control
  startMastra: () => ipcRenderer.invoke('start-mastra'),
  stopMastra: () => ipcRenderer.invoke('stop-mastra'),
  getMastraStatus: () => ipcRenderer.invoke('get-mastra-status'),

  // Recall server control
  startRecall: () => ipcRenderer.invoke('start-recall'),
  stopRecall: () => ipcRenderer.invoke('stop-recall'),
  getRecallStatus: () => ipcRenderer.invoke('get-recall-status'),

  // Bot creation
  createBot: (meetingUrl) => ipcRenderer.invoke('create-bot', meetingUrl),

  // Health checking
  checkHealth: () => ipcRenderer.invoke('check-health'),

  // Environment variables management
  saveEnvVariables: (envVars) => ipcRenderer.invoke('save-env-variables', envVars),
  getStoredEnv: () => ipcRenderer.invoke('get-stored-env'),

  // Event listeners for server logs and status
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

  onRecallLog: (callback) => {
    ipcRenderer.on('recall-log', callback);
    // Return cleanup function
    return () => ipcRenderer.removeListener('recall-log', callback);
  },

  onRecallStatus: (callback) => {
    ipcRenderer.on('recall-status', callback);
    // Return cleanup function
    return () => ipcRenderer.removeListener('recall-status', callback);
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
