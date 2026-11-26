// Container loopback bridge to a single, run-owned, capture-only Unix socket.
'use strict';
const net = require('node:net');

const tcp = net.createServer({ allowHalfOpen: true }, peer => {
  const capture = net.createConnection({ path: '/run/carl/capture.sock', allowHalfOpen: true });
  peer.setTimeout(240000, () => peer.destroy());
  capture.setTimeout(240000, () => capture.destroy());
  peer.on('error', () => capture.destroy());
  capture.on('error', () => peer.destroy());
  peer.on('close', () => capture.destroy());
  capture.on('close', () => peer.destroy());
  peer.pipe(capture).pipe(peer);
});
tcp.maxConnections = 16;
tcp.listen(18808, '127.0.0.1');
