const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { app, ipcMain, BrowserWindow } = require('electron');

// Global references for processes
let mastraProcess = null;
let mainWindow = null;
let ENV_FILE_PATH = null;

// Function to save environment variables
function saveEnvironmentVariables(envVars) {
  try {
    fs.writeFileSync(ENV_FILE_PATH, JSON.stringify(envVars, null, 2));
    return true;
  } catch (error) {
    console.error('Failed to save environment variables:', error);
    return false;
  }
}

// Function to load environment variables
function loadEnvironmentVariables() {
  try {
    if (fs.existsSync(ENV_FILE_PATH)) {
      const data = fs.readFileSync(ENV_FILE_PATH, 'utf8');
      return JSON.parse(data);
    }
  } catch (error) {
    console.error('Failed to load environment variables:', error);
  }
  return {};
}


// Function to initialize the application with GUI
function initializeApp() {
  console.log('VibeB2B running with GUI - initializing main window');
  logToWindow('VibeB2B application starting...');

  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    },
    title: 'VibeB2B - AI Assistant',
    icon: path.join(__dirname, 'assets', 'icon.png'), // Optional icon
    show: false // Don't show until ready
  });

  // Load the index.html file
  const indexPath = path.join(__dirname, 'src', 'index.html');
  mainWindow.loadFile(indexPath);

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    console.log('Main window ready and shown');
    logToWindow('Main window ready and shown');

    // Check if API keys are already configured
    const storedEnv = loadEnvironmentVariables();
    const hasApiKeys = storedEnv.GOOGLE_GENERATIVE_AI_API_KEY &&
                      storedEnv.SLACK_BOT_TOKEN &&
                      storedEnv.SLACK_CHANNEL_ID &&
                      storedEnv.SLACK_SIGNING_SECRET &&
                      storedEnv.ATTIO_API_TOKEN;

    // Send message to renderer to show appropriate interface
    if (hasApiKeys) {
      mainWindow.webContents.send('show-main-interface');
    } else {
      mainWindow.webContents.send('show-setup-page');
    }
  });

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Open DevTools in development
  if (process.env.NODE_ENV === 'development' || process.env.ELECTRON_IS_DEV === 'true') {
    mainWindow.webContents.openDevTools();
  }
}

// Function to start the Mastra server as a child process
function startMastraServer() {
  // Check multiple ways to determine if we're in development
  const isDev = process.env.ELECTRON_IS_DEV === 'true' ||
                process.env.NODE_ENV === 'development' ||
                !fs.existsSync(path.join(__dirname, 'dist'));

  console.log(`Starting Mastra server in ${isDev ? 'development' : 'production'} mode`);
  logToWindow(`Starting Mastra server in ${isDev ? 'development' : 'production'} mode`);
  console.log(`ELECTRON_IS_DEV: ${process.env.ELECTRON_IS_DEV}`);
  console.log(`NODE_ENV: ${process.env.NODE_ENV}`);
  console.log(`Dist exists: ${fs.existsSync(path.join(__dirname, 'dist'))}`);

  // Load stored environment variables
  const storedEnvVars = loadEnvironmentVariables();

  // Run npm run dev for the Mastra server
  const command = 'npm';
  const args = ['run', 'dev'];

  // Spawn the Mastra process with stored environment variables
  mastraProcess = spawn(command, args, {
    cwd: __dirname,
    stdio: ['pipe', 'pipe', 'pipe', 'ipc'],
    env: {
      ...process.env,
      ...storedEnvVars, // Add stored environment variables
      PORT: '4111', // Set the port to 4111
      NODE_ENV: isDev ? 'development' : 'production'
    }
  });

  // Handle stdout
  mastraProcess.stdout.on('data', (data) => {
    const output = data.toString();
    console.log(`[Mastra Server] ${output.trim()}`);
  });

  // Handle stderr
  mastraProcess.stderr.on('data', (data) => {
    const output = data.toString();
    console.error(`[Mastra Server Error] ${output.trim()}`);

    // Check for specific error patterns and provide better feedback
    if (output.includes('EADDRINUSE') && output.includes('3000')) {
      console.error('Port 3000 is already in use. Mastra may not be respecting the PORT environment variable.');
    }
  });

  // Handle process exit
  mastraProcess.on('exit', (code, signal) => {
    console.log(`Mastra server exited with code ${code} and signal ${signal}`);
    mastraProcess = null;
  });

  // Handle process errors
  mastraProcess.on('error', (error) => {
    console.error('Failed to start Mastra server:', error);
    mastraProcess = null;
  });


  return mastraProcess;
}

// Function to stop the Mastra server
function stopMastraServer() {
  if (mastraProcess) {
    console.log('Stopping Mastra server...');
    mastraProcess.kill('SIGTERM');

    // Give it 5 seconds to shut down gracefully, then force kill
    setTimeout(() => {
      if (mastraProcess) {
        console.log('Force killing Mastra server...');
        mastraProcess.kill('SIGKILL');
      }
    }, 5000);
  }
}


// Function to send log messages to the renderer process
function logToWindow(message) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('log-message', message);
  }
}

// Function to check server health
function checkServerHealth(port, name) {
  return new Promise((resolve) => {
    const options = {
      hostname: 'localhost',
      port: port,
      path: '/health',
      method: 'GET',
      timeout: 2000
    };

    const req = http.request(options, (res) => {
      resolve({
        status: res.statusCode === 200 ? 'running' : 'error',
        message: res.statusCode === 200
          ? `✅ ${name} running on port ${port}`
          : `⚠️ ${name} responding with status ${res.statusCode}`
      });
    });

    req.on('error', () => {
      resolve({
        status: 'not_running',
        message: `❌ ${name} not responding on port ${port}`
      });
    });

    req.on('timeout', () => {
      req.destroy();
      resolve({
        status: 'not_running',
        message: `❌ ${name} not responding on port ${port}`
      });
    });

    req.end();
  });
}

// Setup IPC handlers (called from app.whenReady)
function setupIPCHandlers() {
  ipcMain.handle('start-mastra', async () => {
    if (!mastraProcess) {
      logToWindow('Starting Mastra server from UI...');
      startMastraServer();
      return { success: true };
    } else {
      logToWindow('Mastra server is already running');
      return { success: false, message: 'Mastra server is already running' };
    }
  });

  ipcMain.handle('stop-mastra', async () => {
    if (mastraProcess) {
      stopMastraServer();
      return { success: true };
    } else {
      return { success: false, message: 'Mastra server is not running' };
    }
  });

  ipcMain.handle('get-mastra-status', async () => {
    const isRunning = mastraProcess !== null && !mastraProcess.killed;
    return {
      isRunning: isRunning,
      pid: isRunning ? mastraProcess.pid : null
    };
  });

  // Health check IPC handler
  ipcMain.handle('check-health', async () => {
    // Check Mastra server on multiple common ports
    const mastraPorts = [4111, 3001, 5173, 4000, 8000];
    let mastraHealth = { status: 'not_running', message: '❌ Mastra Server not found on common ports' };

    for (const port of mastraPorts) {
      const health = await checkServerHealth(port, 'Mastra Server');
      if (health.status === 'running') {
        mastraHealth = health;
        break;
      }
    }

    return {
      mastra: mastraHealth,
      processes: {
        mastra: {
          isRunning: mastraProcess !== null && !mastraProcess.killed,
          pid: (mastraProcess !== null && !mastraProcess.killed) ? mastraProcess.pid : null
        }
      }
    };
  });

  // Environment variables management
  ipcMain.handle('save-env-variables', async (event, envVars) => {
    const success = saveEnvironmentVariables(envVars);
    return { success };
  });

  ipcMain.handle('get-stored-env', async () => {
    return loadEnvironmentVariables();
  });

  // API keys management
  ipcMain.handle('save-api-keys', async (event, apiKeys) => {
    try {
      // Convert API keys to environment variable format
      const envVars = {
        GOOGLE_GENERATIVE_AI_API_KEY: apiKeys.geminiApiKey,
        SLACK_BOT_TOKEN: apiKeys.slackBotToken,
        SLACK_CHANNEL_ID: apiKeys.slackChannelId,
        SLACK_SIGNING_SECRET: apiKeys.slackSigningSecret,
        ATTIO_API_TOKEN: apiKeys.attioToken
      };

      const success = saveEnvironmentVariables(envVars);
      if (success) {
        logToWindow('API keys saved successfully');
        return { success: true };
      } else {
        return { success: false, message: 'Failed to save API keys' };
      }
    } catch (error) {
      console.error('Error saving API keys:', error);
      return { success: false, message: error.message };
    }
  });
}


// App event handlers
setImmediate(() => {
  app.whenReady().then(() => {
  // Set the environment file path now that app is available
  ENV_FILE_PATH = path.join(app.getPath('userData'), 'environment.json');

  // Setup IPC handlers now that Electron APIs are available
  setupIPCHandlers();

  initializeApp();

  // Start Mastra server automatically when app launches
  setTimeout(() => {
    startMastraServer();
  }, 1000); // Small delay to ensure app is ready
  });
});

app.on('window-all-closed', () => {
  // On macOS, keep app running even when all windows are closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS, re-create window when dock icon is clicked
  if (BrowserWindow.getAllWindows().length === 0) {
    initializeApp();
  }
});

app.on('before-quit', () => {
  console.log('App is quitting, stopping servers...');
  stopMastraServer();
});

app.on('will-quit', () => {
  // Final cleanup
  if (mastraProcess) {
    mastraProcess.kill();
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  stopMastraServer();
  app.quit();
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  stopMastraServer();
  app.quit();
});
