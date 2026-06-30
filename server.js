import { createServer } from 'node:http';
import { createReadStream, promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const publicDir = path.join(__dirname, 'public');
const archiveDir = path.join(__dirname, 'archive');
const imageDir = path.join(archiveDir, 'images');
const csvPath = path.join(archiveDir, 'measurements.csv');
const port = Number(process.env.PORT || 8080);
const host = process.env.HOST || '0.0.0.0';

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

function sendJson(response, status, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
  });
  response.end(body);
}

function csvEscape(value) {
  const text = value == null ? '' : String(value);
  return `"${text.replaceAll('"', '""')}"`;
}

async function ensureArchive() {
  await fs.mkdir(imageDir, { recursive: true });
  try {
    await fs.access(csvPath);
  } catch {
    await fs.writeFile(csvPath, `${csvFields.map(csvEscape).join(',')}\n`, 'utf8');
  }
}

async function readBody(request, maxBytes = 35 * 1024 * 1024) {
  const chunks = [];
  let size = 0;
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBytes) {
      throw new Error('payload_too_large');
    }
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString('utf8');
}

async function saveDataUrl(dataUrl, prefix, stamp) {
  if (!dataUrl || typeof dataUrl !== 'string') return '';
  const match = dataUrl.match(/^data:image\/(png|jpeg);base64,(.+)$/);
  if (!match) return '';
  const ext = match[1] === 'jpeg' ? 'jpg' : 'png';
  const filename = `${stamp}_${prefix}.${ext}`;
  await fs.writeFile(path.join(imageDir, filename), Buffer.from(match[2], 'base64'));
  return `images/${filename}`;
}

async function handleArchive(request, response) {
  await ensureArchive();
  const payload = JSON.parse(await readBody(request));
  const measurement = payload.measurement || {};
  const settings = payload.settings || {};
  const images = payload.images || {};
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');

  const saved = {
    full_image: await saveDataUrl(images.full, 'full', stamp),
    cropped_image: await saveDataUrl(images.cropped, 'cropped', stamp),
    result_image: await saveDataUrl(images.result, 'result', stamp),
    mask_image: await saveDataUrl(images.mask, 'mask', stamp),
  };

  const row = {
    timestamp: new Date().toISOString(),
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
    ...saved,
  };
  await fs.appendFile(csvPath, `${csvFields.map((field) => csvEscape(row[field])).join(',')}\n`, 'utf8');
  sendJson(response, 200, { ok: true, images: saved });
}

async function serveStatic(request, response) {
  const url = new URL(request.url, `http://${request.headers.host || 'localhost'}`);
  let pathname = decodeURIComponent(url.pathname);
  if (pathname === '/') pathname = '/index.html';
  if (pathname === '/api/archive.csv') {
    await ensureArchive();
    response.writeHead(200, {
      'content-type': 'text/csv; charset=utf-8',
      'content-disposition': 'attachment; filename="measurements.csv"',
    });
    createReadStream(csvPath).pipe(response);
    return;
  }

  const filePath = path.normalize(path.join(publicDir, pathname));
  if (!filePath.startsWith(publicDir)) {
    response.writeHead(403);
    response.end('Forbidden');
    return;
  }
  try {
    const stat = await fs.stat(filePath);
    if (!stat.isFile()) throw new Error('not_file');
    response.writeHead(200, {
      'content-type': mimeTypes.get(path.extname(filePath).toLowerCase()) || 'application/octet-stream',
      'content-length': stat.size,
    });
    createReadStream(filePath).pipe(response);
  } catch {
    response.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    response.end('Not found');
  }
}

const server = createServer(async (request, response) => {
  try {
    if (request.method === 'GET' && request.url === '/api/health') {
      sendJson(response, 200, { ok: true });
      return;
    }
    if (request.method === 'POST' && request.url === '/api/archive') {
      await handleArchive(request, response);
      return;
    }
    if (request.method === 'GET' || request.method === 'HEAD') {
      await serveStatic(request, response);
      return;
    }
    response.writeHead(405);
    response.end('Method not allowed');
  } catch (error) {
    const status = error?.message === 'payload_too_large' ? 413 : 500;
    sendJson(response, status, { ok: false, error: error?.message || 'server_error' });
  }
});

server.listen(port, host, () => {
  console.log(`Leaf measurement running at http://${host}:${port}/`);
});
