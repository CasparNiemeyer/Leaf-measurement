# Leaf Measurement

Client-side leaf measurement app with a small Node backend.

## Start

Install Node.js 20 or newer, then run:

```bash
npm start
```

The app is available at:

```text
http://localhost:8080/
```

## Architecture

- `public/app.js` captures camera frames and runs marker detection, perspective crop, HSV leaf masking, convex hull damage estimation, and manual masks in the browser.
- `server.js` serves the static UI and only receives data when you press `Archivieren`.
- `archive/measurements.csv` is the downloadable CSV archive.
- `archive/images/` stores archived full/cropped/result/mask images.

No webcam frames are streamed to the backend during live measurement.
