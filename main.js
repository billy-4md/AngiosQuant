const { app, BrowserWindow } = require('electron');
const cp = require("child_process");
const path = require('path');
const fs = require("fs");

let flaskProcess = null;
const IS_DEV = !app.isPackaged;

function createMainWindow() {
  const winMain = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false,
      enableRemoteModule: false,
    }
  });
  process.env.FLASK_RESOURCES_PATH = IS_DEV ? __dirname : process.resourcesPath;
  startFlaskApp(winMain);
}

function startFlaskApp(winMain) {
  const pythonPath = findPython();
  const scriptPath = IS_DEV ? path.join(__dirname, "python", "server", "app.py") : path.join(process.resourcesPath, "python", "server", "app.py");
  flaskProcess = cp.spawn(pythonPath, [scriptPath]);

  const handleFlaskOutput = (data) => {
    const output = data.toString();
    console.log(`Flask output: ${output}`);
    if (output.includes("Running on http://")) {
      const indexPath = IS_DEV ? path.join(__dirname, "web/index.html") : path.join(process.resourcesPath, "web/index.html");
      const fileUrl = `file://${indexPath}`;
      winMain.loadURL(fileUrl); 
    }
  };

  flaskProcess.stdout.on('data', handleFlaskOutput);
  flaskProcess.stderr.on('data', handleFlaskOutput);

  flaskProcess.on('close', (code) => {
    console.log(`Flask process terminated with code: ${code}`);
  });
}

app.whenReady().then(createMainWindow);

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createMainWindow();
  }
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('quit', () => {
  if (flaskProcess) {
    flaskProcess.kill();
  }
});

function findPython() {
  const possibilities = [
    path.join(__dirname, "python", "bin", "python.exe"), // Path in development
    path.join(process.resourcesPath, "python", "bin", "python.exe"), 
  ];

  
  for (const possibility of possibilities) {
    if (fs.existsSync(possibility)) {
      return possibility;
    }
  }

  console.error("Unable to find python3, checked paths:", possibilities);
  app.quit();
}



