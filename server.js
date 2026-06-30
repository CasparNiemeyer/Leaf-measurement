import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, readFileSync, writeFileSync, promises as fs } from 'node:fs';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, 'public');
const archiveDir = path.join(__dirname, 'archive');
const dataDir = path.join(__dirname, 'data');
const dbPath = path.join(dataDir, 'app-db.json');
const mailOutboxPath = path.join(dataDir, 'mail-outbox.jsonl');
const port = Number(process.env.PORT || 8080);
const host = process.env.HOST || '0.0.0.0';
const publicUrl = (process.env.PUBLIC_URL || 'http://localhost:8080').replace(/\/$/, '');
const sessionDays = 14;
const jwtIssuer = 'leaf-measurement';
const jwtAudience = 'leaf-measurement-web';
const jwtSecret = getJwtSecret();
const rateLimits = new Map();

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

function id(prefix) {
  return `${prefix}_${crypto.randomBytes(16).toString('hex')}`;
}

function token() {
  return crypto.randomBytes(32).toString('base64url');
}

function normalizeEmail(email) {
  return String(email || '').trim().toLowerCase();
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
  return `"${text.replaceAll('"', '""')}"`;
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
    return JSON.parse(await fs.readFile(dbPath, 'utf8'));
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const db = emptyDb();
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
  return String(request.headers['x-forwarded-for'] || request.socket.remoteAddress || 'unknown').split(',')[0].trim();
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
  if (!sameOrigin && !localDev) {
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
  await ensureDataDir();
  const mail = { to, subject, html, text, createdAt: nowIso() };
  await fs.appendFile(mailOutboxPath, `${JSON.stringify(mail)}\n`, 'utf8');
  console.log(`[mail outbox] ${subject} -> ${to}\n${text}`);
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
  const sessionJwt = parseCookies(request).get('lm_session');
  const payload = verifyJwt(sessionJwt);
  if (!payload?.sid || !payload?.sub) return { db: await readDb(), user: null, session: null };
  const db = await readDb();
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
    if (db.users.some((user) => user.email === email)) {
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
  sendJson(response, 201, { ok: true, user: publicUser(result.user), message: 'verification_mail_sent' });
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
  if (!filePath.startsWith(allowedRoot)) {
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
  const { csvPath } = await ensureProjectArchive(projectId);
  response.writeHead(200, securityHeaders({
    'content-type': 'text/csv; charset=utf-8',
    'content-disposition': `attachment; filename="leaf-measurements-${projectId}.csv"`,
  }));
  createReadStream(csvPath).pipe(response);
}

async function handleArchive(request, response) {
  const auth = await getAuth(request);
  const user = requireUser(auth);
  requireVerified(user);
  const payload = await readBody(request);
  const projectId = String(payload.projectId || '');
  const project = findProjectForUser(auth.db, projectId, user.id);
  const measurement = payload.measurement || {};
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
  await fs.appendFile(projectArchive.csvPath, `${csvFields.map((field) => csvEscape(row[field])).join(',')}\n`, 'utf8');

  const savedMeasurement = await mutateDb(async (db) => {
    const item = {
      id: measurementId,
      projectId: project.id,
      userId: user.id,
      createdAt: row.timestamp,
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
  sendJson(response, 200, { ok: true, measurement: measurementDetail(savedMeasurement, await readDb()) });
}

async function serveStatic(request, response) {
  const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
  let pathname = decodeURIComponent(url.pathname);
  if (pathname === '/') pathname = '/index.html';
  const filePath = path.normalize(path.join(publicDir, pathname));
  if (!filePath.startsWith(publicDir)) {
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
  if (request.method === 'POST' && pathname === '/api/projects') return handleCreateProject(request, response);
  if (request.method === 'POST' && pathname === '/api/projects/join') return handleJoinProject(request, response);
  if (request.method === 'POST' && pathname === '/api/archive') return handleArchive(request, response);

  const projectMatch = pathname.match(/^\/api\/projects\/([^/]+)$/);
  if (projectMatch && request.method === 'GET') return handleProject(request, response, projectMatch[1]);

  const measurementsMatch = pathname.match(/^\/api\/projects\/([^/]+)\/measurements$/);
  if (measurementsMatch && request.method === 'GET') return handleProjectMeasurements(request, response, measurementsMatch[1]);

  const measurementMatch = pathname.match(/^\/api\/projects\/([^/]+)\/measurements\/([^/]+)$/);
  if (measurementMatch && request.method === 'GET') return handleMeasurementDetail(request, response, measurementMatch[1], measurementMatch[2]);

  const imageMatch = pathname.match(/^\/api\/projects\/([^/]+)\/measurements\/([^/]+)\/images\/([^/]+)$/);
  if (imageMatch && request.method === 'GET') return handleMeasurementImage(request, response, imageMatch[1], imageMatch[2], imageMatch[3]);

  const csvMatch = pathname.match(/^\/api\/projects\/([^/]+)\/archive\.csv$/);
  if (csvMatch && request.method === 'GET') return handleProjectCsv(request, response, csvMatch[1]);

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
