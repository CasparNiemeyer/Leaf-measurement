import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync, promises as fs } from 'node:fs';
import crypto from 'node:crypto';
import net from 'node:net';
import path from 'node:path';
import tls from 'node:tls';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, 'public');
const archiveDir = path.join(__dirname, 'archive');
const dataDir = path.join(__dirname, 'data');
const dbPath = path.join(dataDir, 'app-db.json');
const mailOutboxPath = path.join(dataDir, 'mail-outbox.jsonl');
loadEnvFile();
const port = Number(process.env.PORT || 8080);
const host = process.env.HOST || '0.0.0.0';
const publicUrl = (process.env.PUBLIC_URL || 'http://localhost:8080').replace(/\/$/, '');
const allowedOrigins = new Set([
  publicUrl,
  'http://localhost:8080',
  'http://127.0.0.1:8080',
  'https://leafmeasurement.casparniemeyer.com',
  'https://leafmeasure.casparniemeyer.com',
].map((origin) => origin.replace(/\/$/, '')));
const trustProxy = parseBoolean(process.env.TRUST_PROXY, false);
const sessionDays = 14;
const jwtIssuer = 'leaf-measurement';
const jwtAudience = 'leaf-measurement-web';
const jwtSecret = getJwtSecret();
const rateLimits = new Map();
const smtpConfig = {
  host: process.env.SMTP_HOST || '',
  port: Number(process.env.SMTP_PORT || 587),
  servername: process.env.SMTP_SERVERNAME || process.env.SMTP_HOST || '',
  user: process.env.SMTP_USER || '',
  pass: process.env.SMTP_PASS || '',
  from: process.env.SMTP_FROM || process.env.SMTP_USER || '',
  fromName: process.env.SMTP_FROM_NAME || 'Leaf Measurement',
  secure: parseBoolean(process.env.SMTP_SECURE, false),
  rejectUnauthorized: parseBoolean(process.env.SMTP_REJECT_UNAUTHORIZED, true),
};

const mimeTypes = new Map([
  ['.html', 'text/html; charset=utf-8'],
  ['.js', 'text/javascript; charset=utf-8'],
  ['.css', 'text/css; charset=utf-8'],
  ['.json', 'application/json; charset=utf-8'],
  ['.png', 'image/png'],
  ['.jpg', 'image/jpeg'],
  ['.jpeg', 'image/jpeg'],
  ['.svg', 'image/svg+xml; charset=utf-8'],
  ['.ico', 'image/x-icon'],
]);

const csvFields = [
  'timestamp',
  'measurement_id',
  'project_id',
  'user_email',
  'description',
  'notes',
  'status',
  'green_area_cm2',
  'convex_hull_cm2',
  'damage_cm2',
  'damage_percent',
  'markers',
  'physical_width_cm',
  'physical_height_cm',
  'digital_width_px',
  'analysis_fps',
  'full_image',
  'cropped_image',
  'result_image',
  'mask_image',
];

let writeQueue = Promise.resolve();

function loadEnvFile() {
  const envPath = path.join(__dirname, '.env');
  if (!existsSync(envPath)) return;
  const lines = readFileSync(envPath, 'utf8').split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#') || !trimmed.includes('=')) continue;
    const index = trimmed.indexOf('=');
    const key = trimmed.slice(0, index).trim();
    let value = trimmed.slice(index + 1).trim();
    if ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
      value = value.slice(1, -1);
    }
    if (key && process.env[key] == null) process.env[key] = value;
  }
}

function parseBoolean(value, fallback = false) {
  if (value == null || value === '') return fallback;
  return ['1', 'true', 'yes', 'on'].includes(String(value).trim().toLowerCase());
}

function getJwtSecret() {
  if (process.env.JWT_SECRET && process.env.JWT_SECRET.length >= 32) {
    return Buffer.from(process.env.JWT_SECRET, 'utf8');
  }
  mkdirSync(dataDir, { recursive: true });
  const secretPath = path.join(dataDir, 'jwt-secret.key');
  if (!existsSync(secretPath)) {
    writeFileSync(secretPath, crypto.randomBytes(48).toString('base64url'), { encoding: 'utf8', mode: 0o600 });
  }
  return Buffer.from(readFileSync(secretPath, 'utf8').trim(), 'utf8');
}

function emptyDb() {
  return {
    users: [],
    sessions: [],
    projects: [],
    measurements: [],
    verifyTokens: [],
    resetTokens: [],
    createdAt: new Date().toISOString(),
  };
}

function normalizeDb(db) {
  db.users ||= [];
  db.sessions ||= [];
  db.projects ||= [];
  db.measurements ||= [];
  db.verifyTokens ||= [];
  db.resetTokens ||= [];
  return db;
}

function id(prefix) {
  return `${prefix}_${crypto.randomBytes(16).toString('hex')}`;
}

function token() {
  return crypto.randomBytes(32).toString('base64url');
}

function normalizeEmail(email) {
  return String(email || '').trim().toLowerCase();
}

function archiveText(value, maxLength) {
  return String(value || '').trim().slice(0, maxLength);
}

function nowIso() {
  return new Date().toISOString();
}

function expiresIn(hours) {
  return new Date(Date.now() + hours * 60 * 60 * 1000).toISOString();
}

function isExpired(iso) {
  return !iso || new Date(iso).getTime() < Date.now();
}

function csvEscape(value) {
  const text = value == null ? '' : String(value);
  const safeText = /^[=+\-@\t\r]/.test(text) ? `'${text}` : text;
  return `"${safeText.replaceAll('"', '""')}"`;
}

function isPathInside(root, candidate) {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative === '' || (relative && !relative.startsWith('..') && !path.isAbsolute(relative));
}

function measurementCsvRow(measurement, db) {
  const user = db.users.find((item) => item.id === measurement.userId);
  return {
    timestamp: measurement.createdAt,
    measurement_id: measurement.id,
    project_id: measurement.projectId,
    user_email: user?.email || '',
    description: measurement.description || '',
    notes: measurement.notes || '',
    status: measurement.status || '',
    green_area_cm2: measurement.greenArea ?? '',
    convex_hull_cm2: measurement.convexArea ?? '',
    damage_cm2: measurement.damageArea ?? '',
    damage_percent: measurement.damagePercent ?? '',
    markers: measurement.markers ?? '',
    physical_width_cm: measurement.settings?.physWidth ?? '',
    physical_height_cm: measurement.settings?.physHeight ?? '',
    digital_width_px: measurement.settings?.digWidth ?? '',
    analysis_fps: measurement.settings?.analysisFps ?? '',
    full_image: measurement.images?.full || '',
    cropped_image: measurement.images?.cropped || '',
    result_image: measurement.images?.result || '',
    mask_image: measurement.images?.mask || '',
  };
}

function projectCsvContent(db, projectId) {
  const rows = db.measurements
    .filter((item) => item.projectId === projectId)
    .sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)))
    .map((measurement) => measurementCsvRow(measurement, db));
  return [
    `${csvFields.map(csvEscape).join(',')}`,
    ...rows.map((row) => csvFields.map((field) => csvEscape(row[field])).join(',')),
  ].join('\n') + '\n';
}

async function rewriteProjectCsv(db, projectId) {
  const projectArchive = await ensureProjectArchive(projectId);
  await fs.writeFile(projectArchive.csvPath, projectCsvContent(db, projectId), 'utf8');
}

const crcTable = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let value = i;
    for (let bit = 0; bit < 8; bit += 1) value = value & 1 ? 0xedb88320 ^ (value >>> 1) : value >>> 1;
    table[i] = value >>> 0;
  }
  return table;
})();

function crc32(buffer) {
  let crc = 0xffffffff;
  for (const byte of buffer) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function dosDateTime(date = new Date()) {
  const year = Math.max(1980, date.getFullYear());
  const time = (date.getHours() << 11) | (date.getMinutes() << 5) | Math.floor(date.getSeconds() / 2);
  const day = ((year - 1980) << 9) | ((date.getMonth() + 1) << 5) | date.getDate();
  return { time, day };
}

function createZip(entries) {
  const localParts = [];
  const centralParts = [];
  let offset = 0;
  const { time, day } = dosDateTime();

  for (const entry of entries) {
    const name = Buffer.from(entry.name.replaceAll('\\', '/'), 'utf8');
    const data = Buffer.isBuffer(entry.data) ? entry.data : Buffer.from(entry.data);
    const checksum = crc32(data);
    const local = Buffer.alloc(30);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(20, 4);
    local.writeUInt16LE(0, 6);
    local.writeUInt16LE(0, 8);
    local.writeUInt16LE(time, 10);
    local.writeUInt16LE(day, 12);
    local.writeUInt32LE(checksum, 14);
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(name.length, 26);
    local.writeUInt16LE(0, 28);
    localParts.push(local, name, data);

    const central = Buffer.alloc(46);
    central.writeUInt32LE(0x02014b50, 0);
    central.writeUInt16LE(20, 4);
    central.writeUInt16LE(20, 6);
    central.writeUInt16LE(0, 8);
    central.writeUInt16LE(0, 10);
    central.writeUInt16LE(time, 12);
    central.writeUInt16LE(day, 14);
    central.writeUInt32LE(checksum, 16);
    central.writeUInt32LE(data.length, 20);
    central.writeUInt32LE(data.length, 24);
    central.writeUInt16LE(name.length, 28);
    central.writeUInt16LE(0, 30);
    central.writeUInt16LE(0, 32);
    central.writeUInt16LE(0, 34);
    central.writeUInt16LE(0, 36);
    central.writeUInt32LE(0, 38);
    central.writeUInt32LE(offset, 42);
    centralParts.push(central, name);
    offset += local.length + name.length + data.length;
  }

  const centralSize = centralParts.reduce((sum, part) => sum + part.length, 0);
  const end = Buffer.alloc(22);
  end.writeUInt32LE(0x06054b50, 0);
  end.writeUInt16LE(0, 4);
  end.writeUInt16LE(0, 6);
  end.writeUInt16LE(entries.length, 8);
  end.writeUInt16LE(entries.length, 10);
  end.writeUInt32LE(centralSize, 12);
  end.writeUInt32LE(offset, 16);
  end.writeUInt16LE(0, 20);
  return Buffer.concat([...localParts, ...centralParts, end]);
}

function base64url(input) {
  return Buffer.from(input).toString('base64url');
}

function signJwt(payload) {
  const header = { alg: 'HS256', typ: 'JWT' };
  const encodedHeader = base64url(JSON.stringify(header));
  const encodedPayload = base64url(JSON.stringify(payload));
  const signature = crypto
    .createHmac('sha256', jwtSecret)
    .update(`${encodedHeader}.${encodedPayload}`)
    .digest('base64url');
  return `${encodedHeader}.${encodedPayload}.${signature}`;
}

function verifyJwt(jwt) {
  try {
    if (!jwt || typeof jwt !== 'string') return null;
    const parts = jwt.split('.');
    if (parts.length !== 3) return null;
    const [encodedHeader, encodedPayload, signature] = parts;
    const expected = crypto
      .createHmac('sha256', jwtSecret)
      .update(`${encodedHeader}.${encodedPayload}`)
      .digest('base64url');
    const a = Buffer.from(signature);
    const b = Buffer.from(expected);
    if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
    const header = JSON.parse(Buffer.from(encodedHeader, 'base64url').toString('utf8'));
    if (header.alg !== 'HS256') return null;
    const payload = JSON.parse(Buffer.from(encodedPayload, 'base64url').toString('utf8'));
    if (payload.iss !== jwtIssuer || payload.aud !== jwtAudience) return null;
    if (!payload.exp || payload.exp * 1000 < Date.now()) return null;
    return payload;
  } catch {
    return null;
  }
}

function tokenHash(value) {
  return crypto.createHash('sha256').update(String(value)).digest('base64url');
}

function createSessionJwt(session) {
  const iat = Math.floor(Date.now() / 1000);
  return signJwt({
    iss: jwtIssuer,
    aud: jwtAudience,
    sub: session.userId,
    sid: session.id,
    iat,
    exp: Math.floor(new Date(session.expiresAt).getTime() / 1000),
  });
}

function securityHeaders(extra = {}) {
  const headers = {
    'x-content-type-options': 'nosniff',
    'x-frame-options': 'DENY',
    'referrer-policy': 'strict-origin-when-cross-origin',
    'cross-origin-opener-policy': 'same-origin',
    'permissions-policy': 'camera=(self), microphone=(), geolocation=(), payment=()',
    'content-security-policy': [
      "default-src 'self'",
      "script-src 'self'",
      "style-src 'self' 'unsafe-inline'",
      "img-src 'self' data: blob:",
      "media-src 'self' blob:",
      "connect-src 'self'",
      "object-src 'none'",
      "base-uri 'self'",
      "form-action 'self'",
      "frame-ancestors 'none'",
    ].join('; '),
    ...extra,
  };
  if (publicUrl.startsWith('https://')) {
    headers['strict-transport-security'] = 'max-age=31536000; includeSubDomains; preload';
  }
  return headers;
}

function sendJson(response, status, payload, extraHeaders = {}) {
  const body = JSON.stringify(payload);
  response.writeHead(status, securityHeaders({
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
    ...extraHeaders,
  }));
  response.end(body);
}

function parseCookies(request) {
  const header = request.headers.cookie || '';
  const cookies = new Map();
  for (const part of header.split(';')) {
    const index = part.indexOf('=');
    if (index === -1) continue;
    cookies.set(part.slice(0, index).trim(), decodeURIComponent(part.slice(index + 1)));
  }
  return cookies;
}

function sessionCookie(value, maxAgeSeconds) {
  const parts = [
    `lm_session=${encodeURIComponent(value || '')}`,
    'Path=/',
    'HttpOnly',
    'SameSite=Lax',
    `Max-Age=${maxAgeSeconds}`,
  ];
  if (publicUrl.startsWith('https://')) parts.push('Secure');
  return parts.join('; ');
}

async function readBody(request, maxBytes = 40 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBytes) throw new Error('payload_too_large');
    chunks.push(chunk);
  }
  const text = Buffer.concat(chunks).toString('utf8');
  return text ? JSON.parse(text) : {};
}

async function ensureDataDir() {
  await fs.mkdir(dataDir, { recursive: true });
  await fs.mkdir(archiveDir, { recursive: true });
}

async function readDb() {
  await ensureDataDir();
  try {
    const db = JSON.parse(await fs.readFile(dbPath, 'utf8'));
    const changed = normalizeDb(db);
    if (changed) await fs.writeFile(dbPath, JSON.stringify(db, null, 2), 'utf8');
    return db;
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const db = emptyDb();
    normalizeDb(db);
    await fs.writeFile(dbPath, JSON.stringify(db, null, 2), 'utf8');
    return db;
  }
}

async function writeDb(db) {
  await ensureDataDir();
  await fs.writeFile(dbPath, JSON.stringify(db, null, 2), 'utf8');
}

async function mutateDb(mutator) {
  return writeQueue = writeQueue.then(async () => {
    const db = await readDb();
    const result = await mutator(db);
    await writeDb(db);
    return result;
  });
}

function hashPassword(password, salt = crypto.randomBytes(16).toString('base64url')) {
  const hash = crypto.scryptSync(password, salt, 64).toString('base64url');
  return { salt, hash };
}

function verifyPassword(password, salt, expectedHash) {
  const { hash } = hashPassword(password, salt);
  const a = Buffer.from(hash);
  const b = Buffer.from(expectedHash || '');
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

function publicUser(user) {
  if (!user) return null;
  return {
    id: user.id,
    email: user.email,
    verified: Boolean(user.verified),
    createdAt: user.createdAt,
  };
}

function requirePassword(password) {
  if (typeof password !== 'string' || password.length < 8) {
    const error = new Error('password_too_short');
    error.status = 400;
    throw error;
  }
}

function requireEmail(email) {
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    const error = new Error('invalid_email');
    error.status = 400;
    throw error;
  }
}

function clientIp(request) {
  if (trustProxy) {
    return String(request.headers['x-forwarded-for'] || request.socket.remoteAddress || 'unknown').split(',')[0].trim();
  }
  return String(request.socket.remoteAddress || 'unknown');
}

function checkRateLimit(request, bucket, limit, windowMs) {
  const key = `${bucket}:${clientIp(request)}`;
  const current = Date.now();
  const entry = rateLimits.get(key) || { count: 0, resetAt: current + windowMs };
  if (entry.resetAt < current) {
    entry.count = 0;
    entry.resetAt = current + windowMs;
  }
  entry.count += 1;
  rateLimits.set(key, entry);
  if (entry.count > limit) {
    const error = new Error('rate_limited');
    error.status = 429;
    throw error;
  }
}

function assertSameOrigin(request) {
  if (!['POST', 'PUT', 'PATCH', 'DELETE'].includes(request.method || '')) return;
  const origin = request.headers.origin;
  if (!origin) return;
  const expected = new URL(publicUrl);
  const actual = new URL(origin);
  const sameOrigin = actual.protocol === expected.protocol && actual.host === expected.host;
  const localDev = ['localhost', '127.0.0.1'].includes(actual.hostname) && ['localhost', '127.0.0.1'].includes(expected.hostname);
  const allowedOrigin = allowedOrigins.has(origin.replace(/\/$/, ''));
  if (!sameOrigin && !localDev && !allowedOrigin) {
    const error = new Error('invalid_origin');
    error.status = 403;
    throw error;
  }
}

function assertJsonRequest(request) {
  if (!request.url?.startsWith('/api/') || request.method === 'GET' || request.method === 'HEAD') return;
  const contentType = String(request.headers['content-type'] || '');
  if (!contentType.includes('application/json')) {
    const error = new Error('json_required');
    error.status = 415;
    throw error;
  }
}

async function sendMail(to, subject, html, text) {
  if (smtpConfig.host && smtpConfig.user && smtpConfig.pass && smtpConfig.from) {
    await sendSmtpMail({ to, subject, html, text });
    console.log(`[mail smtp] ${subject} -> ${to}`);
    return;
  }

  await ensureDataDir();
  const mail = { to, subject, html, text, createdAt: nowIso() };
  await fs.appendFile(mailOutboxPath, `${JSON.stringify(mail)}\n`, 'utf8');
  console.log(`[mail outbox] ${subject} -> ${to}\n${text}`);
}

function encodeMailHeader(value) {
  const text = String(value || '');
  return /^[\x20-\x7e]*$/.test(text) ? text : `=?UTF-8?B?${Buffer.from(text, 'utf8').toString('base64')}?=`;
}

function foldBase64(value) {
  return Buffer.from(String(value || ''), 'utf8').toString('base64').replace(/.{1,76}/g, '$&\r\n').trimEnd();
}

function dotStuff(message) {
  return message.replace(/^\./gm, '..');
}

function formatAddress(address, name = '') {
  return name ? `${encodeMailHeader(name)} <${address}>` : address;
}

function buildMimeMessage({ to, subject, html, text }) {
  const boundary = `leaf-${crypto.randomBytes(12).toString('hex')}`;
  return [
    `From: ${formatAddress(smtpConfig.from, smtpConfig.fromName)}`,
    `To: ${to}`,
    `Subject: ${encodeMailHeader(subject)}`,
    'MIME-Version: 1.0',
    `Date: ${new Date().toUTCString()}`,
    `Message-ID: <${crypto.randomBytes(16).toString('hex')}@leafmeasurement>`,
    `Content-Type: multipart/alternative; boundary="${boundary}"`,
    '',
    `--${boundary}`,
    'Content-Type: text/plain; charset=utf-8',
    'Content-Transfer-Encoding: base64',
    '',
    foldBase64(text),
    `--${boundary}`,
    'Content-Type: text/html; charset=utf-8',
    'Content-Transfer-Encoding: base64',
    '',
    foldBase64(html),
    `--${boundary}--`,
    '',
  ].join('\r\n');
}

function waitForSocket(socket) {
  return new Promise((resolve, reject) => {
    const done = () => {
      socket.off('error', reject);
      resolve(socket);
    };
    socket.once('connect', done);
    socket.once('secureConnect', done);
    socket.once('error', reject);
  });
}

function writeSocket(socket, data) {
  return new Promise((resolve, reject) => {
    socket.write(data, (error) => (error ? reject(error) : resolve()));
  });
}

async function sendSmtpMail({ to, subject, html, text }) {
  let socket = smtpConfig.secure
    ? tls.connect({
      host: smtpConfig.host,
      port: smtpConfig.port,
      servername: smtpConfig.servername,
      rejectUnauthorized: smtpConfig.rejectUnauthorized,
    })
    : net.connect({ host: smtpConfig.host, port: smtpConfig.port });
  socket.setTimeout(20000);
  await waitForSocket(socket);

  let buffer = '';
  let responseLines = [];
  const parseResponse = () => {
    const parts = buffer.split(/\r?\n/);
    buffer = parts.pop() || '';
    for (const line of parts) {
      if (!line) continue;
      responseLines.push(line);
      if (/^\d{3} /.test(line)) {
        const lines = responseLines;
        responseLines = [];
        return { code: Number(line.slice(0, 3)), lines };
      }
    }
    return null;
  };
  const readResponse = () => {
    const parsed = parseResponse();
    if (parsed) return Promise.resolve(parsed);
    return new Promise((resolve, reject) => {
      const onData = (chunk) => {
        buffer += chunk.toString('utf8');
        const response = parseResponse();
        if (response) {
          socket.off('data', onData);
          socket.off('error', onError);
          socket.off('timeout', onTimeout);
          resolve(response);
        }
      };
      const onError = (error) => {
        socket.off('data', onData);
        socket.off('timeout', onTimeout);
        reject(error);
      };
      const onTimeout = () => {
        socket.destroy();
        onError(new Error('smtp_timeout'));
      };
      socket.on('data', onData);
      socket.once('error', onError);
      socket.once('timeout', onTimeout);
    });
  };
  const command = async (line, expectedCodes) => {
    await writeSocket(socket, `${line}\r\n`);
    const response = await readResponse();
    if (!expectedCodes.includes(response.code)) {
      throw new Error(`smtp_${response.code}: ${response.lines.join(' | ')}`);
    }
    return response;
  };

  try {
    let response = await readResponse();
    if (response.code !== 220) throw new Error(`smtp_${response.code}: ${response.lines.join(' | ')}`);
    response = await command('EHLO leafmeasurement.local', [250]);
    if (!smtpConfig.secure && response.lines.some((line) => /STARTTLS/i.test(line))) {
      await command('STARTTLS', [220]);
      socket.removeAllListeners('data');
      socket.removeAllListeners('error');
      socket.removeAllListeners('timeout');
      socket = tls.connect({
      socket,
      servername: smtpConfig.servername,
      rejectUnauthorized: smtpConfig.rejectUnauthorized,
      });
      socket.setTimeout(20000);
      await waitForSocket(socket);
      buffer = '';
      responseLines = [];
      await command('EHLO leafmeasurement.local', [250]);
    }

    await command('AUTH LOGIN', [334]);
    await command(Buffer.from(smtpConfig.user, 'utf8').toString('base64'), [334]);
    await command(Buffer.from(smtpConfig.pass, 'utf8').toString('base64'), [235]);
    await command(`MAIL FROM:<${smtpConfig.from}>`, [250]);
    await command(`RCPT TO:<${to}>`, [250, 251]);
    await command('DATA', [354]);
    await writeSocket(socket, `${dotStuff(buildMimeMessage({ to, subject, html, text }))}\r\n.\r\n`);
    response = await readResponse();
    if (response.code !== 250) throw new Error(`smtp_${response.code}: ${response.lines.join(' | ')}`);
    await command('QUIT', [221]);
  } finally {
    socket.end();
  }
}

async function sendVerifyMail(user, verifyToken) {
  const link = `${publicUrl}/?verify=${encodeURIComponent(verifyToken)}`;
  await sendMail(
    user.email,
    'Leaf Measurement E-Mail bestätigen',
    `<p>Bitte bestätige deine E-Mail-Adresse:</p><p><a href="${link}">${link}</a></p>`,
    `Bitte bestätige deine E-Mail-Adresse: ${link}`,
  );
}

async function sendResetMail(user, resetToken) {
  const link = `${publicUrl}/?reset=${encodeURIComponent(resetToken)}`;
  await sendMail(
    user.email,
    'Leaf Measurement Passwort zurücksetzen',
    `<p>Setze dein Passwort hier zurück:</p><p><a href="${link}">${link}</a></p>`,
    `Setze dein Passwort hier zurück: ${link}`,
  );
}

async function getAuth(request) {
  const db = await readDb();
  const sessionJwt = parseCookies(request).get('lm_session');
  const payload = verifyJwt(sessionJwt);
  if (!payload?.sid || !payload?.sub) return { db, user: null, session: null };
  const session = db.sessions.find((item) => item.id === payload.sid && item.userId === payload.sub);
  if (!session || isExpired(session.expiresAt) || session.tokenHash !== tokenHash(sessionJwt)) {
    return { db, user: null, session: null };
  }
  const user = db.users.find((item) => item.id === session.userId) || null;
  return { db, user, session };
}

function requireUser(auth) {
  if (!auth.user) {
    const error = new Error('auth_required');
    error.status = 401;
    throw error;
  }
  return auth.user;
}

function requireVerified(user) {
  if (!user.verified) {
    const error = new Error('email_not_verified');
    error.status = 403;
    throw error;
  }
}

function findProjectForUser(db, projectId, userId) {
  const project = db.projects.find((item) => item.id === projectId);
  if (!project || !project.memberIds.includes(userId)) {
    const error = new Error('project_not_found');
    error.status = 404;
    throw error;
  }
  return project;
}

function requireProjectOwner(project, userId) {
  if (project.ownerId !== userId) {
    const error = new Error('forbidden');
    error.status = 403;
    throw error;
  }
}

function projectSummary(db, project, user) {
  const measurements = db.measurements.filter((item) => item.projectId === project.id);
  return {
    id: project.id,
    name: project.name,
    ownerId: project.ownerId,
    createdAt: project.createdAt,
    inviteUrl: `${publicUrl}/?join=${encodeURIComponent(project.inviteToken)}`,
    measurementCount: measurements.length,
    latestMeasurementAt: measurements.at(-1)?.createdAt || null,
    members: project.memberIds
      .map((memberId) => db.users.find((candidate) => candidate.id === memberId))
      .filter(Boolean)
      .map((member) => ({
        id: member.id,
        email: member.email,
        role: member.id === project.ownerId ? 'owner' : 'member',
        isCurrentUser: member.id === user.id,
      })),
  };
}

async function ensureProjectArchive(projectId) {
  const base = path.join(archiveDir, 'projects', projectId);
  const images = path.join(base, 'images');
  await fs.mkdir(images, { recursive: true });
  const csvPath = path.join(base, 'measurements.csv');
  try {
    await fs.access(csvPath);
  } catch {
    await fs.writeFile(csvPath, `${csvFields.map(csvEscape).join(',')}\n`, 'utf8');
  }
  return { base, images, csvPath };
}

async function saveDataUrl(dataUrl, directory, prefix, stamp) {
  if (!dataUrl || typeof dataUrl !== 'string') return '';
  const match = dataUrl.match(/^data:image\/(png|jpeg);base64,(.+)$/);
  if (!match) return '';
  const ext = match[1] === 'jpeg' ? 'jpg' : 'png';
  const filename = `${stamp}_${prefix}.${ext}`;
  await fs.writeFile(path.join(directory, filename), Buffer.from(match[2], 'base64'));
  return filename;
}

function measurementListItem(measurement, db) {
  const user = db.users.find((item) => item.id === measurement.userId);
  return {
    id: measurement.id,
    createdAt: measurement.createdAt,
    userEmail: user?.email || '',
    description: measurement.description || '',
    notes: measurement.notes || '',
    status: measurement.status,
    greenArea: measurement.greenArea,
    convexArea: measurement.convexArea,
    damageArea: measurement.damageArea,
    damagePercent: measurement.damagePercent,
    markers: measurement.markers,
  };
}

function measurementDetail(measurement, db) {
  return {
    ...measurementListItem(measurement, db),
    settings: measurement.settings,
    images: Object.fromEntries(
      Object.entries(measurement.images || {}).map(([key, filename]) => [
        key,
        filename ? `/api/projects/${measurement.projectId}/measurements/${measurement.id}/images/${key}` : '',
      ]),
    ),
  };
}

async function handleRegister(request, response) {
  const body = await readBody(request);
  const email = normalizeEmail(body.email);
  const password = String(body.password || '');
  requireEmail(email);
  requirePassword(password);

  const result = await mutateDb(async (db) => {
    const existingUser = db.users.find((user) => user.email === email);
    if (existingUser && !existingUser.verified) {
      const verifyToken = token();
      db.verifyTokens = db.verifyTokens.filter((entry) => entry.userId !== existingUser.id);
      db.verifyTokens.push({ tokenHash: tokenHash(verifyToken), userId: existingUser.id, expiresAt: expiresIn(48), createdAt: nowIso() });
      return { user: existingUser, verifyToken, resent: true };
    }
    if (existingUser) {
      const error = new Error('email_already_registered');
      error.status = 409;
      throw error;
    }
    const passwordData = hashPassword(password);
    const user = {
      id: id('user'),
      email,
      passwordSalt: passwordData.salt,
      passwordHash: passwordData.hash,
      verified: false,
      createdAt: nowIso(),
    };
    const verifyToken = token();
    db.users.push(user);
    db.verifyTokens.push({ tokenHash: tokenHash(verifyToken), userId: user.id, expiresAt: expiresIn(48), createdAt: nowIso() });
    return { user, verifyToken };
  });
  await sendVerifyMail(result.user, result.verifyToken);
  sendJson(response, result.resent ? 200 : 201, {
    ok: true,
    user: publicUser(result.user),
    message: result.resent ? 'verification_mail_resent' : 'verification_mail_sent',
  });
}

async function handleResendVerification(request, response) {
  const body = await readBody(request);
  const email = normalizeEmail(body.email);
  requireEmail(email);
  const result = await mutateDb(async (db) => {
    const user = db.users.find((item) => item.email === email);
    if (!user || user.verified) return null;
    const verifyToken = token();
    db.verifyTokens = db.verifyTokens.filter((entry) => entry.userId !== user.id);
    db.verifyTokens.push({ tokenHash: tokenHash(verifyToken), userId: user.id, expiresAt: expiresIn(48), createdAt: nowIso() });
    return { user, verifyToken };
  });
  if (result) await sendVerifyMail(result.user, result.verifyToken);
  sendJson(response, 200, { ok: true, message: 'verification_mail_sent_if_needed' });
}

async function handleLogin(request, response) {
  const body = await readBody(request);
  const email = normalizeEmail(body.email);
  const password = String(body.password || '');
  const result = await mutateDb(async (db) => {
    const user = db.users.find((item) => item.email === email);
    if (!user || !verifyPassword(password, user.passwordSalt, user.passwordHash)) {
      const error = new Error('invalid_login');
      error.status = 401;
      throw error;
    }
    const session = {
      id: id('session'),
      userId: user.id,
      createdAt: nowIso(),
      expiresAt: new Date(Date.now() + sessionDays * 24 * 60 * 60 * 1000).toISOString(),
    };
    const jwt = createSessionJwt(session);
    session.tokenHash = tokenHash(jwt);
    db.sessions = db.sessions.filter((item) => !isExpired(item.expiresAt));
    db.sessions.push(session);
    return { user, session, jwt };
  });
  sendJson(response, 200, { ok: true, user: publicUser(result.user) }, {
    'set-cookie': sessionCookie(result.jwt, sessionDays * 24 * 60 * 60),
  });
}

async function handleLogout(request, response) {
  const payload = verifyJwt(parseCookies(request).get('lm_session'));
  if (payload?.sid) {
    await mutateDb(async (db) => {
      db.sessions = db.sessions.filter((session) => session.id !== payload.sid);
    });
  }
  sendJson(response, 200, { ok: true }, { 'set-cookie': sessionCookie('', 0) });
}

async function handleVerify(request, response) {
  const body = await readBody(request);
  const verifyToken = String(body.token || '');
  const user = await mutateDb(async (db) => {
    const verifyTokenHash = tokenHash(verifyToken);
    const entry = db.verifyTokens.find((item) => item.tokenHash === verifyTokenHash || item.token === verifyToken);
    if (!entry || isExpired(entry.expiresAt)) {
      const error = new Error('invalid_or_expired_token');
      error.status = 400;
      throw error;
    }
    const found = db.users.find((item) => item.id === entry.userId);
    if (!found) {
      const error = new Error('user_not_found');
      error.status = 404;
      throw error;
    }
    found.verified = true;
    found.verifiedAt = nowIso();
    db.verifyTokens = db.verifyTokens.filter((item) => item !== entry);
    return found;
  });
  sendJson(response, 200, { ok: true, user: publicUser(user) });
}

async function handleForgotPassword(request, response) {
  const body = await readBody(request);
  const email = normalizeEmail(body.email);
  const result = await mutateDb(async (db) => {
    const user = db.users.find((item) => item.email === email);
    if (!user) return null;
    const resetToken = token();
    db.resetTokens.push({ tokenHash: tokenHash(resetToken), userId: user.id, expiresAt: expiresIn(2), createdAt: nowIso() });
    return { user, resetToken };
  });
  if (result) await sendResetMail(result.user, result.resetToken);
  sendJson(response, 200, { ok: true, message: 'reset_mail_sent_if_account_exists' });
}

async function handleResetPassword(request, response) {
  const body = await readBody(request);
  const resetToken = String(body.token || '');
  const password = String(body.password || '');
  requirePassword(password);
  await mutateDb(async (db) => {
    const resetTokenHash = tokenHash(resetToken);
    const entry = db.resetTokens.find((item) => item.tokenHash === resetTokenHash || item.token === resetToken);
    if (!entry || isExpired(entry.expiresAt)) {
      const error = new Error('invalid_or_expired_token');
      error.status = 400;
      throw error;
    }
    const user = db.users.find((item) => item.id === entry.userId);
    if (!user) {
      const error = new Error('user_not_found');
      error.status = 404;
      throw error;
    }
    const passwordData = hashPassword(password);
    user.passwordSalt = passwordData.salt;
    user.passwordHash = passwordData.hash;
    db.resetTokens = db.resetTokens.filter((item) => item !== entry);
    db.sessions = db.sessions.filter((item) => item.userId !== user.id);
  });
  sendJson(response, 200, { ok: true });
}

async function handleAccountUpdate(request, response) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  const body = await readBody(request);
  const currentPassword = String(body.currentPassword || '');
  const requestedEmail = normalizeEmail(body.email || user.email);
  const newPassword = String(body.newPassword || '');

  requireEmail(requestedEmail);
  if (!verifyPassword(currentPassword, user.passwordSalt, user.passwordHash)) {
    const error = new Error('invalid_current_password');
    error.status = 401;
    throw error;
  }
  if (newPassword) requirePassword(newPassword);

  const result = await mutateDb(async (db) => {
    const storedUser = db.users.find((item) => item.id === user.id);
    if (!storedUser) {
      const error = new Error('user_not_found');
      error.status = 404;
      throw error;
    }
    const emailChanged = requestedEmail !== storedUser.email;
    if (emailChanged && db.users.some((item) => item.id !== storedUser.id && item.email === requestedEmail)) {
      const error = new Error('email_already_registered');
      error.status = 409;
      throw error;
    }
    let verifyToken = '';
    if (emailChanged) {
      storedUser.email = requestedEmail;
      storedUser.verified = false;
      delete storedUser.verifiedAt;
      verifyToken = token();
      db.verifyTokens = db.verifyTokens.filter((entry) => entry.userId !== storedUser.id);
      db.verifyTokens.push({ tokenHash: tokenHash(verifyToken), userId: storedUser.id, expiresAt: expiresIn(48), createdAt: nowIso() });
    }
    if (newPassword) {
      const passwordData = hashPassword(newPassword);
      storedUser.passwordSalt = passwordData.salt;
      storedUser.passwordHash = passwordData.hash;
    }
    return { user: storedUser, verifyToken, emailChanged };
  });

  if (result.verifyToken) await sendVerifyMail(result.user, result.verifyToken);
  sendJson(response, 200, {
    ok: true,
    user: publicUser(result.user),
    message: result.emailChanged ? 'profile_updated_verification_required' : 'profile_updated',
  });
}

async function handleMe(request, response) {
  const auth = await getAuth(request);
  const projects = auth.user
    ? auth.db.projects
      .filter((project) => project.memberIds.includes(auth.user.id))
      .map((project) => projectSummary(auth.db, project, auth.user))
    : [];
  sendJson(response, 200, { ok: true, user: publicUser(auth.user), projects });
}

async function handleCreateProject(request, response) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  requireVerified(user);
  const body = await readBody(request);
  const name = String(body.name || '').trim();
  if (name.length < 2) {
    const error = new Error('project_name_required');
    error.status = 400;
    throw error;
  }
  const project = await mutateDb(async (db) => {
    const item = {
      id: id('project'),
      name,
      ownerId: user.id,
      memberIds: [user.id],
      inviteToken: token(),
      createdAt: nowIso(),
    };
    db.projects.push(item);
    return item;
  });
  sendJson(response, 201, { ok: true, project: projectSummary(await readDb(), project, user) });
}

async function handleJoinProject(request, response) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  requireVerified(user);
  const body = await readBody(request);
  const inviteToken = String(body.token || '');
  const project = await mutateDb(async (db) => {
    const found = db.projects.find((item) => item.inviteToken === inviteToken);
    if (!found) {
      const error = new Error('invalid_invite');
      error.status = 404;
      throw error;
    }
    if (!found.memberIds.includes(user.id)) found.memberIds.push(user.id);
    return found;
  });
  sendJson(response, 200, { ok: true, project: projectSummary(await readDb(), project, user) });
}

async function handleProject(request, response, projectId) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  const project = findProjectForUser(auth.db, projectId, user.id);
  sendJson(response, 200, { ok: true, project: projectSummary(auth.db, project, user) });
}

async function handleDeleteProject(request, response, projectId) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  const project = findProjectForUser(auth.db, projectId, user.id);
  requireProjectOwner(project, user.id);

  await mutateDb(async (db) => {
    db.projects = db.projects.filter((item) => item.id !== projectId);
    db.measurements = db.measurements.filter((item) => item.projectId !== projectId);
    return null;
  });

  const base = path.normalize(path.join(archiveDir, 'projects', projectId));
  const allowedRoot = path.normalize(path.join(archiveDir, 'projects'));
  if (base !== allowedRoot && isPathInside(allowedRoot, base)) await fs.rm(base, { recursive: true, force: true });
  sendJson(response, 200, { ok: true });
}

async function handleRemoveProjectMember(request, response, projectId, memberId) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  const project = findProjectForUser(auth.db, projectId, user.id);
  requireProjectOwner(project, user.id);
  if (memberId === project.ownerId) {
    const error = new Error('cannot_remove_owner');
    error.status = 400;
    throw error;
  }

  const updated = await mutateDb(async (db) => {
    const item = db.projects.find((candidate) => candidate.id === projectId);
    if (!item?.memberIds.includes(memberId)) {
      const error = new Error('member_not_found');
      error.status = 404;
      throw error;
    }
    item.memberIds = item.memberIds.filter((idValue) => idValue !== memberId);
    return item;
  });

  sendJson(response, 200, { ok: true, project: projectSummary(await readDb(), updated, user) });
}

async function handleProjectMeasurements(request, response, projectId) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  findProjectForUser(auth.db, projectId, user.id);
  const measurements = auth.db.measurements
    .filter((item) => item.projectId === projectId)
    .sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)))
    .map((measurement) => measurementListItem(measurement, auth.db));
  sendJson(response, 200, { ok: true, measurements });
}

async function handleMeasurementDetail(request, response, projectId, measurementId) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  findProjectForUser(auth.db, projectId, user.id);
  const measurement = auth.db.measurements.find((item) => item.id === measurementId && item.projectId === projectId);
  if (!measurement) {
    const error = new Error('measurement_not_found');
    error.status = 404;
    throw error;
  }
  sendJson(response, 200, { ok: true, measurement: measurementDetail(measurement, auth.db) });
}

async function handleDeleteMeasurement(request, response, projectId, measurementId) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  const project = findProjectForUser(auth.db, projectId, user.id);
  requireProjectOwner(project, user.id);
  let deleted = null;

  await mutateDb(async (db) => {
    const index = db.measurements.findIndex((item) => item.id === measurementId && item.projectId === projectId);
    if (index === -1) {
      const error = new Error('measurement_not_found');
      error.status = 404;
      throw error;
    }
    [deleted] = db.measurements.splice(index, 1);
    return null;
  });

  const imagesRoot = path.normalize(path.join(archiveDir, 'projects', projectId, 'images'));
  for (const filename of Object.values(deleted?.images || {})) {
    if (!filename) continue;
    const filePath = path.normalize(path.join(imagesRoot, filename));
    if (isPathInside(imagesRoot, filePath)) await fs.rm(filePath, { force: true });
  }
  await rewriteProjectCsv(await readDb(), projectId);
  sendJson(response, 200, { ok: true });
}

async function handleMeasurementImage(request, response, projectId, measurementId, imageKey) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  findProjectForUser(auth.db, projectId, user.id);
  const measurement = auth.db.measurements.find((item) => item.id === measurementId && item.projectId === projectId);
  const filename = measurement?.images?.[imageKey];
  if (!filename) {
    response.writeHead(404);
    response.end('Not found');
    return;
  }
  const filePath = path.normalize(path.join(archiveDir, 'projects', projectId, 'images', filename));
  const allowedRoot = path.normalize(path.join(archiveDir, 'projects', projectId, 'images'));
  if (!isPathInside(allowedRoot, filePath)) {
    response.writeHead(403);
    response.end('Forbidden');
    return;
  }
  const stat = await fs.stat(filePath);
  response.writeHead(200, securityHeaders({
    'content-type': mimeTypes.get(path.extname(filePath).toLowerCase()) || 'application/octet-stream',
    'content-length': stat.size,
  }));
  createReadStream(filePath).pipe(response);
}

async function handleProjectCsv(request, response, projectId) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  findProjectForUser(auth.db, projectId, user.id);
  await rewriteProjectCsv(auth.db, projectId);
  const { csvPath } = await ensureProjectArchive(projectId);
  response.writeHead(200, securityHeaders({
    'content-type': 'text/csv; charset=utf-8',
    'content-disposition': `attachment; filename="leaf-measurements-${projectId}.csv"`,
  }));
  createReadStream(csvPath).pipe(response);
}

async function handleProjectZip(request, response, projectId) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  findProjectForUser(auth.db, projectId, user.id);
  const projectArchive = await ensureProjectArchive(projectId);
  const entries = [{
    name: 'measurements.csv',
    data: Buffer.from(projectCsvContent(auth.db, projectId), 'utf8'),
  }];
  const addedImages = new Set();
  const imagesRoot = path.normalize(projectArchive.images);

  for (const measurement of auth.db.measurements.filter((item) => item.projectId === projectId)) {
    for (const filename of Object.values(measurement.images || {})) {
      if (!filename || addedImages.has(filename)) continue;
      const filePath = path.normalize(path.join(imagesRoot, filename));
      if (!isPathInside(imagesRoot, filePath)) continue;
      try {
        entries.push({ name: `images/${filename}`, data: await fs.readFile(filePath) });
        addedImages.add(filename);
      } catch {
        // Missing image files should not block a CSV-first archive export.
      }
    }
  }

  const zip = createZip(entries);
  response.writeHead(200, securityHeaders({
    'content-type': 'application/zip',
    'content-length': zip.length,
    'content-disposition': `attachment; filename="leaf-measurements-${projectId}.zip"`,
  }));
  response.end(zip);
}

async function handleArchive(request, response) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  requireVerified(user);
  const payload = await readBody(request);
  const projectId = String(payload.projectId || '');
  const project = findProjectForUser(auth.db, projectId, user.id);
  const measurement = payload.measurement || {};
  const description = archiveText(payload.description, 120);
  const notes = archiveText(payload.notes, 2000);
  const settings = payload.settings || {};
  const images = payload.images || {};
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const measurementId = id('measurement');
  const projectArchive = await ensureProjectArchive(project.id);

  const saved = {
    full: await saveDataUrl(images.full, projectArchive.images, `${measurementId}_full`, stamp),
    cropped: await saveDataUrl(images.cropped, projectArchive.images, `${measurementId}_cropped`, stamp),
    result: await saveDataUrl(images.result, projectArchive.images, `${measurementId}_result`, stamp),
    mask: await saveDataUrl(images.mask, projectArchive.images, `${measurementId}_mask`, stamp),
  };
  const row = {
    timestamp: nowIso(),
    measurement_id: measurementId,
    project_id: project.id,
    user_email: user.email,
    description,
    notes,
    status: measurement.status || '',
    green_area_cm2: measurement.area ?? '',
    convex_hull_cm2: measurement.convexArea ?? '',
    damage_cm2: measurement.damageArea ?? '',
    damage_percent: measurement.damagePercent ?? '',
    markers: measurement.markers ?? '',
    physical_width_cm: settings.physWidth ?? '',
    physical_height_cm: settings.physHeight ?? '',
    digital_width_px: settings.digWidth ?? '',
    analysis_fps: settings.analysisFps ?? '',
    full_image: saved.full,
    cropped_image: saved.cropped,
    result_image: saved.result,
    mask_image: saved.mask,
  };
  const savedMeasurement = await mutateDb(async (db) => {
    const item = {
      id: measurementId,
      projectId: project.id,
      userId: user.id,
      createdAt: row.timestamp,
      description,
      notes,
      status: row.status,
      greenArea: Number(measurement.area ?? 0),
      convexArea: Number(measurement.convexArea ?? 0),
      damageArea: Number(measurement.damageArea ?? 0),
      damagePercent: Number(measurement.damagePercent ?? 0),
      markers: Number(measurement.markers ?? 0),
      settings,
      images: saved,
    };
    db.measurements.push(item);
    return item;
  });
  const updatedDb = await readDb();
  await rewriteProjectCsv(updatedDb, project.id);
  sendJson(response, 200, { ok: true, measurement: measurementDetail(savedMeasurement, updatedDb) });
}

async function serveStatic(request, response) {
  const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
  let pathname = decodeURIComponent(url.pathname);
  if (pathname === '/') pathname = '/index.html';
  if (['/login', '/signup', '/dashboard', '/archive', '/measurements', '/profile'].includes(pathname)) {
    pathname = '/index.html';
  }
  const filePath = path.normalize(path.join(publicDir, pathname));
  if (!isPathInside(publicDir, filePath)) {
    response.writeHead(403, securityHeaders());
    response.end('Forbidden');
    return;
  }
  try {
    const stat = await fs.stat(filePath);
    if (!stat.isFile()) throw new Error('not_file');
    response.writeHead(200, securityHeaders({
      'content-type': mimeTypes.get(path.extname(filePath).toLowerCase()) || 'application/octet-stream',
      'content-length': stat.size,
    }));
    createReadStream(filePath).pipe(response);
  } catch {
    response.writeHead(404, securityHeaders({ 'content-type': 'text/plain; charset=utf-8' }));
    response.end('Not found');
  }
}

async function route(request, response) {
  assertSameOrigin(request);
  assertJsonRequest(request);
  const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
  const pathname = url.pathname;

  if (request.method === 'GET' && pathname === '/api/health') return sendJson(response, 200, { ok: true });
  if (request.method === 'GET' && pathname === '/api/me') return handleMe(request, response);
  if (request.method === 'POST' && pathname === '/api/auth/register') {
    checkRateLimit(request, 'register', 8, 15 * 60 * 1000);
    return handleRegister(request, response);
  }
  if (request.method === 'POST' && pathname === '/api/auth/resend-verification') {
    checkRateLimit(request, 'resend-verification', 8, 15 * 60 * 1000);
    return handleResendVerification(request, response);
  }
  if (request.method === 'POST' && pathname === '/api/auth/login') {
    checkRateLimit(request, 'login', 20, 15 * 60 * 1000);
    return handleLogin(request, response);
  }
  if (request.method === 'POST' && pathname === '/api/auth/logout') return handleLogout(request, response);
  if (request.method === 'POST' && pathname === '/api/auth/verify') return handleVerify(request, response);
  if (request.method === 'POST' && pathname === '/api/auth/forgot-password') {
    checkRateLimit(request, 'forgot-password', 8, 15 * 60 * 1000);
    return handleForgotPassword(request, response);
  }
  if (request.method === 'POST' && pathname === '/api/auth/reset-password') {
    checkRateLimit(request, 'reset-password', 12, 15 * 60 * 1000);
    return handleResetPassword(request, response);
  }
  if (request.method === 'POST' && pathname === '/api/account') {
    checkRateLimit(request, 'account', 20, 15 * 60 * 1000);
    return handleAccountUpdate(request, response);
  }
  if (request.method === 'POST' && pathname === '/api/projects') return handleCreateProject(request, response);
  if (request.method === 'POST' && pathname === '/api/projects/join') return handleJoinProject(request, response);
  if (request.method === 'POST' && pathname === '/api/archive') return handleArchive(request, response);

  const projectMatch = pathname.match(/^\/api\/projects\/([^/]+)$/);
  if (projectMatch && request.method === 'GET') return handleProject(request, response, projectMatch[1]);
  if (projectMatch && request.method === 'DELETE') return handleDeleteProject(request, response, projectMatch[1]);

  const memberMatch = pathname.match(/^\/api\/projects\/([^/]+)\/members\/([^/]+)$/);
  if (memberMatch && request.method === 'DELETE') return handleRemoveProjectMember(request, response, memberMatch[1], memberMatch[2]);

  const measurementsMatch = pathname.match(/^\/api\/projects\/([^/]+)\/measurements$/);
  if (measurementsMatch && request.method === 'GET') return handleProjectMeasurements(request, response, measurementsMatch[1]);

  const measurementMatch = pathname.match(/^\/api\/projects\/([^/]+)\/measurements\/([^/]+)$/);
  if (measurementMatch && request.method === 'GET') return handleMeasurementDetail(request, response, measurementMatch[1], measurementMatch[2]);
  if (measurementMatch && request.method === 'DELETE') return handleDeleteMeasurement(request, response, measurementMatch[1], measurementMatch[2]);

  const imageMatch = pathname.match(/^\/api\/projects\/([^/]+)\/measurements\/([^/]+)\/images\/([^/]+)$/);
  if (imageMatch && request.method === 'GET') return handleMeasurementImage(request, response, imageMatch[1], imageMatch[2], imageMatch[3]);

  const csvMatch = pathname.match(/^\/api\/projects\/([^/]+)\/archive\.csv$/);
  if (csvMatch && request.method === 'GET') return handleProjectCsv(request, response, csvMatch[1]);

  const zipMatch = pathname.match(/^\/api\/projects\/([^/]+)\/archive\.zip$/);
  if (zipMatch && request.method === 'GET') return handleProjectZip(request, response, zipMatch[1]);

  if (request.method === 'GET' || request.method === 'HEAD') return serveStatic(request, response);

  response.writeHead(405, securityHeaders());
  response.end('Method not allowed');
}

const server = createServer(async (request, response) => {
  try {
    await route(request, response);
  } catch (error) {
    const status = error?.status || (error?.message === 'payload_too_large' ? 413 : 500);
    sendJson(response, status, { ok: false, error: error?.message || 'server_error' });
  }
});

server.listen(port, host, () => {
  console.log(`Leaf measurement running at http://${host}:${port}/`);
});
