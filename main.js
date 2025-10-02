const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { app, ipcMain, BrowserWindow } = require('electron');

// Global references for processes
let mastraProcess = null;
let recallProcess = null;
let webhookProcess = null;
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


// Function to preload API keys from .env file
function preloadApiKeys() {
  const fs = require('fs');
  const path = require('path');
  const envPath = path.join(__dirname, '.env');
  const storedEnv = loadEnvironmentVariables();

  // Load API keys from .env file if it exists
  let envKeys = {};
  if (fs.existsSync(envPath)) {
    try {
      const envContent = fs.readFileSync(envPath, 'utf8');
      envContent.split('\n').forEach(line => {
        const [key, value] = line.split('=');
        if (key && value && !key.startsWith('#')) {
          envKeys[key.trim()] = value.trim();
        }
      });
    } catch (error) {
      console.error('Error reading .env file:', error);
    }
  }

  // Save the env keys to stored environment if not already stored
  let needsSave = false;
  Object.keys(envKeys).forEach(key => {
    if (!storedEnv[key]) {
      storedEnv[key] = envKeys[key];
      needsSave = true;
    }
  });

  if (needsSave) {
    saveEnvironmentVariables(storedEnv);
    console.log('API keys loaded from .env and saved');
  }

  return storedEnv;
}

// Function to initialize the application with GUI
function initializeApp() {
  console.log('VibeB2B running with GUI - initializing main window');
  logToWindow('VibeB2B application starting...');

  // Preload API keys
  const apiKeys = preloadApiKeys();

  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      enableRemoteModule: true
    },
    title: '', // Remove title
    frame: false, // Remove default title bar for custom white appearance
    backgroundColor: '#ffffff', // White background
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

    // Check if API keys are available (they should be preloaded)
    const hasApiKeys = apiKeys.GOOGLE_GENERATIVE_AI_API_KEY &&
                      apiKeys.SLACK_BOT_TOKEN &&
                      apiKeys.SLACK_CHANNEL_ID &&
                      apiKeys.SLACK_SIGNING_SECRET &&
                      apiKeys.ATTIO_API_TOKEN &&
                      apiKeys.RECALL_AI_API_KEY;

    // Always show main interface since keys are preloaded
    if (hasApiKeys) {
      mainWindow.webContents.send('show-main-interface');
      logToWindow('API keys preloaded successfully');
    } else {
      logToWindow('Warning: Some API keys may be missing');
      mainWindow.webContents.send('show-main-interface'); // Still show main interface
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

// Function to cleanup processes using port 4111
function cleanupPort4111() {
  console.log('Cleaning up any processes using port 4111...');
  try {
    // Use lsof to find and kill processes using port 4111
    const { execSync } = require('child_process');
    execSync('lsof -ti:4111 | xargs kill -9 2>/dev/null || true', { stdio: 'inherit' });
    console.log('Port 4111 cleanup completed');
  } catch (error) {
    console.log('Port cleanup completed (or no processes were using the port)');
  }
}

// Function to start the Mastra server as a child process
function startMastraServer() {
  // First cleanup any existing processes using port 4111
  cleanupPort4111();

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

// Function to toggle the Recall listening subprocess
function toggleRecallListening() {
  if (recallProcess && !recallProcess.killed) {
    // Stop the process
    console.log('Stopping Recall listening subprocess...');
    logToWindow('Stopping Recall listening subprocess...');
    recallProcess.kill('SIGTERM');

    // Give it 5 seconds to shut down gracefully, then force kill
    setTimeout(() => {
      if (recallProcess && !recallProcess.killed) {
        console.log('Force killing Recall listening subprocess...');
        recallProcess.kill('SIGKILL');
      }
    }, 5000);

    recallProcess = null;
    return { action: 'stopped' };
  } else {
    // Start the process
    console.log('Starting Recall listening subprocess...');
    logToWindow('Starting Recall listening subprocess...');

    // Load stored environment variables
    const storedEnvVars = loadEnvironmentVariables();

    // Use tsx to run the TypeScript file
    const command = 'npx';
    const args = ['tsx', 'recall-ai-test.ts'];

    // Spawn the Recall process with stored environment variables
    recallProcess = spawn(command, args, {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        ...storedEnvVars, // Add stored environment variables
        NODE_ENV: 'production'
      }
    });

    // Handle stdout - stream all output to logs
    recallProcess.stdout.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        console.log(`[Recall] ${output}`);
        logToWindow(`[Recall] ${output}`);
      }
    });

    // Handle stderr - stream all output to logs
    recallProcess.stderr.on('data', (data) => {
      const output = data.toString().trim();
      if (output) {
        console.error(`[Recall Error] ${output}`);
        logToWindow(`[Recall Error] ${output}`);
      }
    });

    // Handle process exit
    recallProcess.on('exit', (code, signal) => {
      console.log(`Recall listening process exited with code ${code} and signal ${signal}`);
      logToWindow(`Recall listening process exited with code ${code}`);
      recallProcess = null;
      // Notify renderer that process stopped
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send('recall-process-stopped');
      }
    });

    // Handle process errors
    recallProcess.on('error', (error) => {
      console.error('Failed to start Recall listening process:', error);
      logToWindow(`Failed to start Recall listening: ${error.message}`);
      recallProcess = null;
    });

    return { action: 'started' };
  }
}

// Function to start the webhook server subprocess
function startWebhookServer() {
  if (webhookProcess) {
    console.log('Webhook server is already running');
    logToWindow('Webhook server is already running');
    return webhookProcess;
  }

  console.log('Starting webhook server...');
  logToWindow('Starting webhook server...');

  // Load stored environment variables
  const storedEnvVars = loadEnvironmentVariables();

  // Use tsx to run the TypeScript file
  const command = 'npx';
  const args = ['tsx', 'webhook-server.ts'];

  // Spawn the webhook process
  webhookProcess = spawn(command, args, {
    cwd: __dirname,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      ...storedEnvVars,
      NODE_ENV: 'production'
    }
  });

  // Handle stdout
  webhookProcess.stdout.on('data', (data) => {
    const output = data.toString();
    console.log(`[Webhook Server] ${output.trim()}`);
    logToWindow(`[Webhook] ${output.trim()}`);
  });

  // Handle stderr
  webhookProcess.stderr.on('data', (data) => {
    const output = data.toString();
    console.error(`[Webhook Server Error] ${output.trim()}`);
    logToWindow(`[Webhook Error] ${output.trim()}`);
  });

  // Handle process exit
  webhookProcess.on('exit', (code, signal) => {
    console.log(`Webhook server exited with code ${code} and signal ${signal}`);
    logToWindow(`Webhook server exited with code ${code}`);
    webhookProcess = null;
  });

  // Handle process errors
  webhookProcess.on('error', (error) => {
    console.error('Failed to start webhook server:', error);
    logToWindow(`Failed to start webhook server: ${error.message}`);
    webhookProcess = null;
  });

  return webhookProcess;
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

// Global reference for svix process
let svixProcess = null;

// Function to setup svix webhook relay
async function setupSvixRelay() {
  return new Promise((resolve, reject) => {
    // Check if svix is already running
    if (svixProcess && !svixProcess.killed) {
      console.log('Svix relay is already running');
      reject(new Error('Svix relay is already running'));
      return;
    }

    console.log('Setting up svix webhook relay...');
    logToWindow('Starting svix webhook relay...');

    // Set up environment with Homebrew PATH
    const env = {
      ...process.env,
      PATH: '/opt/homebrew/bin:/opt/homebrew/sbin:' + process.env.PATH
    };

    // Run svix listen command as background process
    svixProcess = spawn('svix', ['listen', 'http://localhost:3001/webhook'], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: env
    });

    let relayUrl = null;
    let outputBuffer = '';

    // Handle stdout - look for the first https URL (relay URL)
    svixProcess.stdout.on('data', (data) => {
      const output = data.toString();
      outputBuffer += output;
      console.log(`[Svix] ${output.trim()}`);

      // Look for the first https URL in the output (this is the relay URL)
      const urlMatch = output.match(/https:\/\/[^\s]+/);
      if (urlMatch && !relayUrl) {
        relayUrl = urlMatch[0];
        console.log(`Found relay URL: ${relayUrl}`);
        logToWindow(`Webhook relay URL: ${relayUrl}`);

        // Relay URL found, resolve the promise
        resolve({
          success: true,
          relayUrl: relayUrl,
          message: 'Webhook relay setup successfully'
        });
      }
    });

    // Handle stderr
    svixProcess.stderr.on('data', (data) => {
      const output = data.toString();
      console.error(`[Svix Error] ${output.trim()}`);
    });

    // Handle process exit (unexpected exit)
    svixProcess.on('exit', (code, signal) => {
      console.log(`Svix process exited unexpectedly with code ${code}, signal ${signal}`);
      logToWindow(`Svix relay stopped unexpectedly (code: ${code})`);
      svixProcess = null;
    });

    // Handle process errors
    svixProcess.on('error', (error) => {
      console.error('Failed to start svix process:', error);
      logToWindow('Failed to start svix relay');
      svixProcess = null;
      reject(error);
    });

    // Timeout after 30 seconds if no relay URL found
    setTimeout(() => {
      if (!relayUrl) {
        console.error('Timeout waiting for svix relay URL');
        if (svixProcess) {
          svixProcess.kill('SIGTERM');
          svixProcess = null;
        }
        reject(new Error('Timeout waiting for svix relay URL'));
      }
    }, 30000);
  });
}

// Function to stop svix relay
function stopSvixRelay() {
  if (svixProcess && !svixProcess.killed) {
    console.log('Stopping svix relay...');
    logToWindow('Stopping svix relay...');
    svixProcess.kill('SIGTERM');

    // Give it 5 seconds to shut down gracefully, then force kill
    setTimeout(() => {
      if (svixProcess && !svixProcess.killed) {
        console.log('Force killing svix relay...');
        svixProcess.kill('SIGKILL');
      }
    }, 5000);

    svixProcess = null;
  }
}

// Setup IPC handlers (called from app.whenReady)
function setupIPCHandlers() {
  // Window control handlers
  ipcMain.handle('minimize-window', () => {
    if (mainWindow) mainWindow.minimize();
  });

  ipcMain.handle('maximize-window', () => {
    if (mainWindow) {
      if (mainWindow.isMaximized()) {
        mainWindow.unmaximize();
      } else {
        mainWindow.maximize();
      }
    }
  });

  ipcMain.handle('close-window', () => {
    if (mainWindow) mainWindow.close();
  });

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

  ipcMain.handle('toggle-recall-listening', async () => {
    const result = toggleRecallListening();
    if (result.action === 'started') {
      logToWindow('Recall listening started from UI');
      return { success: true, action: 'started' };
    } else {
      logToWindow('Recall listening stopped from UI');
      return { success: true, action: 'stopped' };
    }
  });

  ipcMain.handle('get-recall-status', async () => {
    const isRunning = recallProcess !== null && !recallProcess.killed;
    return {
      isRunning: isRunning,
      pid: isRunning ? recallProcess.pid : null
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
        },
        recall: {
          isRunning: recallProcess !== null && !recallProcess.killed,
          pid: (recallProcess !== null && !recallProcess.killed) ? recallProcess.pid : null
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

  // API keys management (legacy - keys are now preloaded)
  ipcMain.handle('save-api-keys', async (event, apiKeys) => {
    // API keys are preloaded, just acknowledge
    logToWindow('API keys are preloaded and ready');

    // Start webhook server (called during preload)
    if (!webhookProcess) {
      setTimeout(() => {
        startWebhookServer();
      }, 1000);
    }

    return { success: true };
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

  // Start servers automatically when app launches
  setTimeout(() => {
    startMastraServer();
    startWebhookServer();
  }, 2000); // Increased delay to ensure port cleanup completes
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
  stopSvixRelay();

  // Synchronous cleanup for processes
  if (recallProcess) {
    console.log('Force stopping recall listening process...');
    recallProcess.kill('SIGKILL');
    recallProcess = null;
  }
  if (webhookProcess) {
    console.log('Force stopping webhook server...');
    webhookProcess.kill('SIGKILL');
    webhookProcess = null;
  }
  if (svixProcess) {
    console.log('Force stopping svix relay...');
    svixProcess.kill('SIGKILL');
    svixProcess = null;
  }
});

app.on('will-quit', () => {
  // Final cleanup - force kill any remaining processes
  if (mastraProcess) {
    mastraProcess.kill('SIGKILL');
  }
  if (recallProcess) {
    recallProcess.kill('SIGKILL');
  }
  if (webhookProcess) {
    webhookProcess.kill('SIGKILL');
  }
  if (svixProcess) {
    svixProcess.kill('SIGKILL');
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  stopMastraServer();
  stopSvixRelay();

  // Synchronous cleanup for processes
  if (recallProcess) {
    console.log('Force stopping recall listening process due to error...');
    recallProcess.kill('SIGKILL');
    recallProcess = null;
  }
  if (webhookProcess) {
    console.log('Force stopping webhook server due to error...');
    webhookProcess.kill('SIGKILL');
    webhookProcess = null;
  }
  if (svixProcess) {
    console.log('Force stopping svix relay due to error...');
    svixProcess.kill('SIGKILL');
    svixProcess = null;
  }

  app.quit();
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  stopMastraServer();
  stopSvixRelay();

  // Synchronous cleanup for processes
  if (recallProcess) {
    console.log('Force stopping recall listening process due to error...');
    recallProcess.kill('SIGKILL');
    recallProcess = null;
  }
  if (webhookProcess) {
    console.log('Force stopping webhook server due to error...');
    webhookProcess.kill('SIGKILL');
    webhookProcess = null;
  }
  if (svixProcess) {
    console.log('Force stopping svix relay due to error...');
    svixProcess.kill('SIGKILL');
    svixProcess = null;
  }

  app.quit();
});
