const { app, BrowserWindow } = require('electron');
const cp = require("child_process");
const path = require('path');
const fs = require("fs");

let flaskProcess = null;
const IS_DEV = !app.isPackaged;

function createMainWindow() {
  const winMain = new BrowserWindow({
    width: 1000,
    height: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: false,
      enableRemoteModule: false,
    }
  });

  //process.env.FLASK_RESOURCES_PATH = process.resourcesPath
  process.env.FLASK_RESOURCES_PATH = __dirname;
  startFlaskApp(winMain);
}

function startFlaskApp(winMain) {
  //const pythonPath = findPython();
  const pythonPath = "C:\\Users\\MFE\\.conda\\envs\\xavier2\\python.exe";
  const scriptPath = IS_DEV ? path.join(__dirname, "python/server/app.py") : path.join(process.resourcesPath, "python/server/app.py");
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
    path.join(__dirname, "python", "bin", "python3"), // Path in development
    path.join(process.resourcesPath, "python", "bin", "python3"), 
  ];

  
  for (const possibility of possibilities) {
    if (fs.existsSync(possibility)) {
      return possibility;
    }
  }

  console.error("Unable to find python3, checked paths:", possibilities);
  app.quit();
}




// function createSetupWindow() {
  //   winSetup = new BrowserWindow({
  //     width: 400,
  //     height: 200,
  //     webPreferences: {
  //       nodeIntegration: true,
  //       contextIsolation: false
  //     }
  //   });
  
  //   winSetup.loadFile('html/setup.html');
  //   startFlaskApp();
  // }
  
  // function runInstallScript() {
  //   const installProcess = spawn('python3', ['server/install.py']);
  
  //   installProcess.stdout.on('data', (data) => {
  //     console.log(`stdout (Install): ${data.toString()}`);
  //   });
  
  //   installProcess.stderr.on('data', (data) => {
  //     console.error(`stderr (Install): ${data.toString()}`);
  //   });
  
  //   installProcess.on('close', (code) => {
  //     if (code === 0) {
  //       console.log('Package installation succeeded.');
  //       winSetup.close(); 
  //       winSetup = null;
  //       createMainWindow(); 
  //     } else {
  //       console.error(`Package installation failed with code: ${code}`);
  //     }
  //   });
  // }
