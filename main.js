import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import pkg from 'electron';
const { app, BrowserWindow, ipcMain } = pkg;

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Environment variables storage
const ENV_FILE_PATH = path.join(app.getPath('userData'), 'environment.json');

// Keep a global reference of the window object
let mainWindow;
let mastraProcess = null;
let recallProcess = null;

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

// Function to determine which page to load
function getInitialPage() {
  // Start with main app page
  return 'index.html';
}

// Function to create the main application window
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    show: false // Don't show until ready-to-show
  });

  // Load the appropriate initial page
  const initialPage = getInitialPage();
  mainWindow.loadFile(`src/renderer/${initialPage}`);

  // Note: Removed auto-start from main.js to prevent conflicts

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  // Open DevTools in development
  if (process.env.ELECTRON_IS_DEV) {
    mainWindow.webContents.openDevTools();
  }

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// Function to start the Mastra server as a child process
function startMastraServer() {
  // Check multiple ways to determine if we're in development
  const isDev = process.env.ELECTRON_IS_DEV === 'true' ||
                process.env.NODE_ENV === 'development' ||
                !fs.existsSync(path.join(__dirname, 'dist'));

  console.log(`Starting Mastra server in ${isDev ? 'development' : 'production'} mode`);
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
    // Send to renderer process for logging in UI
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('mastra-log', { type: 'stdout', data: output });
    }
  });

  // Handle stderr
  mastraProcess.stderr.on('data', (data) => {
    const output = data.toString();
    console.error(`[Mastra Server Error] ${output.trim()}`);

    // Check for specific error patterns and provide better feedback
    if (output.includes('EADDRINUSE') && output.includes('3000')) {
      console.error('Port 3000 is already in use. Mastra may not be respecting the PORT environment variable.');
    }

    // Send to renderer process for logging in UI
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('mastra-log', { type: 'stderr', data: output });
    }
  });

  // Handle process exit
  mastraProcess.on('exit', (code, signal) => {
    console.log(`Mastra server exited with code ${code} and signal ${signal}`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('mastra-status', {
        status: code === 0 ? 'stopped' : 'error',
        code,
        signal,
        error: code !== 0 ? `Server exited with code ${code}` : null
      });
    }
    mastraProcess = null;
  });

  // Handle process errors
  mastraProcess.on('error', (error) => {
    console.error('Failed to start Mastra server:', error);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('mastra-status', {
        status: 'error',
        error: error.message
      });
    }
    mastraProcess = null;
  });

  // Notify renderer that server is starting
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('mastra-status', { status: 'starting' });
  }

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

// Function to start the Recall server
function startRecallServer() {
  if (recallProcess && !recallProcess.killed) {
    console.log('Recall server is already running');
    return recallProcess;
  }

  console.log('Starting Recall Flask server...');

  // Load stored environment variables
  const storedEnvVars = loadEnvironmentVariables();

  // Start the Python Recall server
  recallProcess = spawn('python3', ['src/recall/recallAPI.py'], {
    cwd: __dirname,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      ...storedEnvVars
    }
  });

  // Handle stdout
  recallProcess.stdout.on('data', (data) => {
    const output = data.toString();
    console.log(`[Recall Server] ${output.trim()}`);
    // Send to renderer process for logging in UI
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('recall-log', { type: 'stdout', data: output });
    }
  });

  // Handle stderr
  recallProcess.stderr.on('data', (data) => {
    const output = data.toString();
    console.error(`[Recall Server Error] ${output.trim()}`);
    // Send to renderer process for logging in UI
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('recall-log', { type: 'stderr', data: output });
    }
  });

  // Handle process exit
  recallProcess.on('exit', (code, signal) => {
    console.log(`Recall server exited with code ${code} and signal ${signal}`);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('recall-status', {
        status: code === 0 ? 'stopped' : 'error',
        code,
        signal,
        error: code !== 0 ? `Server exited with code ${code}` : null
      });
    }
    recallProcess = null;
  });

  // Handle process errors
  recallProcess.on('error', (error) => {
    console.error('Failed to start Recall server:', error);
    if (mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('recall-status', {
        status: 'error',
        error: error.message
      });
    }
    recallProcess = null;
  });

  // Notify renderer that server is starting
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('recall-status', { status: 'starting' });
  }

  return recallProcess;
}

// Function to stop the Recall server
function stopRecallServer() {
  if (recallProcess && !recallProcess.killed) {
    console.log('Stopping Recall server...');
    recallProcess.kill('SIGTERM');

    // Give it 5 seconds to shut down gracefully, then force kill
    setTimeout(() => {
      if (recallProcess && !recallProcess.killed) {
        console.log('Force killing Recall server...');
        recallProcess.kill('SIGKILL');
      }
    }, 5000);
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

// IPC handlers
ipcMain.handle('start-mastra', async () => {
  if (!mastraProcess) {
    startMastraServer();
    return { success: true };
  } else {
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

// Recall server IPC handlers
ipcMain.handle('start-recall', async () => {
  if (!recallProcess) {
    startRecallServer();
    return { success: true };
  } else {
    return { success: false, message: 'Recall server is already running' };
  }
});

ipcMain.handle('stop-recall', async () => {
  if (recallProcess) {
    stopRecallServer();
    return { success: true };
  } else {
    return { success: false, message: 'Recall server is not running' };
  }
});

ipcMain.handle('get-recall-status', async () => {
  const isRunning = recallProcess !== null && !recallProcess.killed;
  return {
    isRunning: isRunning,
    pid: isRunning ? recallProcess.pid : null
  };
});

// Bot creation IPC handler
ipcMain.handle('create-bot', async (event, meetingUrl) => {
  if (!meetingUrl || !meetingUrl.trim()) {
    return { success: false, message: 'Meeting URL is required' };
  }

  try {
    const response = await fetch('http://localhost:3000/start_VB2B', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ meeting_url: meetingUrl.trim() })
    });

    if (response.ok) {
      const result = await response.json();
      return {
        success: true,
        message: 'Bot created successfully!',
        data: result
      };
    } else {
      const errorText = await response.text();
      return {
        success: false,
        message: `Failed to create bot: ${errorText}`
      };
    }
  } catch (error) {
    if (error.code === 'ECONNREFUSED') {
      return {
        success: false,
        message: 'Cannot connect to Recall server. Make sure it\'s running.'
      };
    }
    return {
      success: false,
      message: `Error creating bot: ${error.message}`
    };
  }
});

// Health check IPC handler
ipcMain.handle('check-health', async () => {
  const recallHealth = await checkServerHealth(3000, 'Recall Server');

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
    recall: recallHealth,
    mastra: mastraHealth,
    processes: {
      recall: {
        isRunning: recallProcess !== null && !recallProcess.killed,
        pid: (recallProcess !== null && !recallProcess.killed) ? recallProcess.pid : null
      },
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

// Navigation handler
ipcMain.handle('navigate-to-page', async (event, page) => {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.loadFile(`src/renderer/${page}`);
    return { success: true };
  }
  return { success: false, error: 'Window not available' };
});

// App event handlers
app.whenReady().then(() => {
  createWindow();

  // Start Mastra server automatically when app launches
  setTimeout(() => {
    startMastraServer();
  }, 1000); // Small delay to ensure window is ready

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  // On macOS, keep app running even when all windows are closed
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  console.log('App is quitting, stopping servers...');
  stopMastraServer();
  stopRecallServer();
});

app.on('will-quit', () => {
  // Final cleanup
  if (mastraProcess) {
    mastraProcess.kill();
  }
  if (recallProcess) {
    recallProcess.kill();
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
  stopMastraServer();
  stopRecallServer();
  app.quit();
});

process.on('unhandledRejection', (reason, promise) => {
  console.error('Unhandled Rejection at:', promise, 'reason:', reason);
  stopMastraServer();
  stopRecallServer();
  app.quit();
});
