const fs = require("fs");
const PORT = 8080;
const HOST = "0.0.0.0";

function readFile(p) {
  return fs.readFileSync(p, "utf8");
}

const handler = (req, res) => {
  res.end(req.url);
};

export function exported(a) {
  return a * 2;
}

class Server {
  constructor(port) {
    this.port = port;
  }
  listen() {
    return this.port;
  }
}
