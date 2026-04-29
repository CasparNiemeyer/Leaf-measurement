import asyncio
import base64
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import Response
from nicegui import Client, app, core, run, ui


BASE_DIR = Path(__file__).resolve().parent
LANG_DIR = BASE_DIR / 'lang'


@dataclass
class MeasurementSettings:
    mode_camera: bool = True
    phys_width: float = 13.4
    phys_height: float = 13.4
    dig_width: int = 700
    kernel_size: int = 6
    lower_hsv: tuple[int, int, int] = (0, 60, 60)
    upper_hsv: tuple[int, int, int] = (179, 200, 200)
    draw_marker: bool = True
    draw_bound: bool = True
    draw_contours: bool = True
    draw_convex: bool = True


app_settings = MeasurementSettings()
video_capture: cv2.VideoCapture | None = None
uploaded_image: np.ndarray | None = None
frozen_frame: np.ndarray | None = None
freeze_enabled = False
input_revision = 0
processed_cache: dict[str, Any] = {}
processing_lock: asyncio.Lock | None = None
last_measurement: dict[str, Any] = {
    'status': 'Noch kein Bild verarbeitet',
    'area': None,
    'convex_area': None,
    'damage_area': None,
    'damage_percent': None,
    'markers': 0,
}

langlist = sorted(i.name for i in os.scandir(LANG_DIR) if i.is_file())
sellang = 'de.json' if 'de.json' in langlist else (langlist[0] if langlist else '')
language: dict[str, str] = {}

black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')


def read_language(name: str) -> dict[str, str]:
    if not name:
        return {}
    with open(LANG_DIR / name, encoding='utf-8') as file:
        return json.load(file)


language = read_language(sellang)


def text(key: str, fallback: str) -> str:
    return language.get(key, fallback)


def clamp_hsv(value: tuple[int, int, int]) -> tuple[int, int, int]:
    h, s, v = value
    return (
        max(0, min(179, int(h))),
        max(0, min(255, int(s))),
        max(0, min(255, int(v))),
    )


def hsv_to_hex(value: tuple[int, int, int]) -> str:
    hsv = np.uint8([[clamp_hsv(value)]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'


def hex_to_hsv(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    value = value.strip().lstrip('#')
    if len(value) >= 6:
        value = value[:6]
    try:
        rgb = [int(value[i:i + 2], 16) for i in (0, 2, 4)]
    except ValueError:
        return None
    hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0]
    return int(hsv[0]), int(hsv[1]), int(hsv[2])


def update_hsv_setting(attr: str, value: str | None) -> None:
    hsv = hex_to_hsv(value)
    if hsv is not None:
        setattr(app_settings, attr, hsv)


def snapshot_settings() -> dict[str, Any]:
    return {
        'mode_camera': bool(app_settings.mode_camera),
        'phys_width': float(app_settings.phys_width or 13.4),
        'phys_height': float(app_settings.phys_height or 13.4),
        'dig_width': int(app_settings.dig_width or 700),
        'kernel_size': int(app_settings.kernel_size or 1),
        'lower_hsv': clamp_hsv(app_settings.lower_hsv),
        'upper_hsv': clamp_hsv(app_settings.upper_hsv),
        'draw_marker': bool(app_settings.draw_marker),
        'draw_bound': bool(app_settings.draw_bound),
        'draw_contours': bool(app_settings.draw_contours),
        'draw_convex': bool(app_settings.draw_convex),
        'freeze_enabled': freeze_enabled,
        'input_revision': input_revision,
    }


def convert(frame: np.ndarray) -> bytes:
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    success, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes() if success else b''


def blank_frame(message: str, width: int = 700, height: int | None = None) -> np.ndarray:
    width = max(320, int(width or 700))
    height = max(180, int(height or width))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    y = max(48, height // 2 - 24)
    for line in message.splitlines():
        cv2.putText(frame, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        y += 42
    return frame


def find_external_contours(image: np.ndarray) -> list[np.ndarray]:
    result = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result[0] if len(result) == 2 else result[1]


def order_points(points: np.ndarray) -> np.ndarray:
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered


def detect_aruco(frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None, str | None]:
    if not hasattr(cv2, 'aruco'):
        return [], None, 'OpenCV ArUco Modul nicht gefunden'

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    if hasattr(cv2.aruco, 'ArucoDetector'):
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
        return corners, ids, None

    parameters = cv2.aruco.DetectorParameters_create()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return corners, ids, None


def process_frame(frame: np.ndarray, settings: dict[str, Any]) -> dict[str, Any]:
    dig_width = max(100, min(2500, int(settings['dig_width'])))
    kernel_size = max(1, min(100, int(settings['kernel_size'])))
    phys_width = max(0.001, float(settings['phys_width']))
    phys_height = max(0.001, float(settings['phys_height']))
    pixel_area = (phys_height / dig_width) * (phys_width / dig_width)

    display_frame = frame.copy()
    fallback = blank_frame('Warte auf 4 ArUco-Marker', dig_width, dig_width)
    images = {
        'full': display_frame,
        'cropped': fallback,
        'mask': fallback,
        'result': fallback,
    }
    measurement = {
        'status': 'Suche 4 ArUco-Marker',
        'area': None,
        'convex_area': None,
        'damage_area': None,
        'damage_percent': None,
        'markers': 0,
    }

    corners, ids, detection_error = detect_aruco(frame)
    if detection_error is not None:
        measurement['status'] = detection_error
        images['cropped'] = blank_frame(detection_error, dig_width, dig_width)
        images['mask'] = blank_frame(detection_error, dig_width, dig_width)
        images['result'] = blank_frame(detection_error, dig_width, dig_width)
        return {'images': images, 'measurement': measurement}

    marker_count = 0 if ids is None else len(ids)
    measurement['markers'] = marker_count

    if ids is not None and settings['draw_marker']:
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)

    if ids is None:
        message = '0/4 ArUco-Marker gefunden'
        measurement['status'] = message
        images['full'] = display_frame
        images['cropped'] = blank_frame(message, dig_width, dig_width)
        images['mask'] = blank_frame(message, dig_width, dig_width)
        images['result'] = blank_frame(message, dig_width, dig_width)
        return {'images': images, 'measurement': measurement}

    selected_centroids = []
    for marker_id, marker_corners in zip(ids.flatten(), corners):
        if int(marker_id) in {0, 1, 2, 3}:
            selected_centroids.append(np.asarray(marker_corners, dtype=np.float32).reshape(4, 2).mean(axis=0))

    if len(selected_centroids) != 4:
        message = f'{len(selected_centroids)}/4 ArUco-Marker 0-3 gefunden'
        measurement['status'] = message
        images['full'] = display_frame
        images['cropped'] = blank_frame(message, dig_width, dig_width)
        images['mask'] = blank_frame(message, dig_width, dig_width)
        images['result'] = blank_frame(message, dig_width, dig_width)
        return {'images': images, 'measurement': measurement}

    centroids = np.asarray(selected_centroids, dtype=np.float32)

    sorted_points = order_points(centroids)

    if settings['draw_bound']:
        cv2.polylines(display_frame, [sorted_points.astype(np.int32)], True, (0, 255, 255), 2)

    destination = np.float32([
        [0, 0],
        [dig_width - 1, 0],
        [dig_width - 1, dig_width - 1],
        [0, dig_width - 1],
    ])
    transform = cv2.getPerspectiveTransform(sorted_points, destination)
    cropped = cv2.warpPerspective(frame, transform, (dig_width, dig_width), flags=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    lower = np.array(settings['lower_hsv'], dtype=np.uint8)
    upper = np.array(settings['upper_hsv'], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    result = cv2.bitwise_and(cropped, cropped, mask=opened_mask)
    mask_preview = cv2.cvtColor(opened_mask, cv2.COLOR_GRAY2BGR)

    contours = find_external_contours(opened_mask)

    if not contours:
        message = 'Kein Blatt erkannt'
        cv2.putText(result, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        measurement['status'] = message
        images.update({
            'full': display_frame,
            'cropped': cropped,
            'mask': mask_preview,
            'result': result,
        })
        return {'images': images, 'measurement': measurement}

    contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)

    hull_mask = np.zeros_like(opened_mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
    green_mask = cv2.bitwise_and(opened_mask, hull_mask)
    damage_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(green_mask))

    damage_kernel = np.ones((max(1, kernel_size), max(1, kernel_size)), np.uint8)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, damage_kernel)

    green_pixels = cv2.countNonZero(green_mask)
    hull_pixels = cv2.countNonZero(hull_mask)
    damage_pixels = cv2.countNonZero(damage_mask)

    area = round(green_pixels * pixel_area, 3)
    convex_area = round(hull_pixels * pixel_area, 3)
    damage_area = round(damage_pixels * pixel_area, 3)
    damage_percent = round((damage_pixels / hull_pixels) * 100, 1) if hull_pixels else 0.0

    mask_preview = cv2.cvtColor(damage_mask, cv2.COLOR_GRAY2BGR)
    result[damage_mask > 0] = (0, 0, 255)

    if settings['draw_convex']:
        cv2.drawContours(result, [hull], -1, (255, 0, 0), 5)
    if settings['draw_contours']:
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 5)

    cv2.putText(result, f'Green area: {area:.3f} cm2', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(result, f'Area convex hull: {convex_area:.3f} cm2', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(result, f'Damage: {damage_area:.3f} cm2 ({damage_percent:.1f}%)', (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    measurement.update({
        'status': 'Messung ok',
        'area': area,
        'convex_area': convex_area,
        'damage_area': damage_area,
        'damage_percent': damage_percent,
    })
    images.update({
        'full': display_frame,
        'cropped': cropped,
        'mask': mask_preview,
        'result': result,
    })
    return {'images': images, 'measurement': measurement}


def cache_key(settings: dict[str, Any]) -> tuple[Any, ...]:
    return (
        settings['mode_camera'],
        settings['phys_width'],
        settings['phys_height'],
        settings['dig_width'],
        settings['kernel_size'],
        settings['lower_hsv'],
        settings['upper_hsv'],
        settings['draw_marker'],
        settings['draw_bound'],
        settings['draw_contours'],
        settings['draw_convex'],
        settings['freeze_enabled'],
        settings['input_revision'],
    )


def get_processing_lock() -> asyncio.Lock:
    global processing_lock
    if processing_lock is None:
        processing_lock = asyncio.Lock()
    return processing_lock


async def read_source_frame(use_camera: bool) -> np.ndarray | None:
    if use_camera:
        if freeze_enabled and frozen_frame is not None:
            return frozen_frame.copy()
        if video_capture is None or not video_capture.isOpened():
            return None
        success, frame = await run.io_bound(video_capture.read)
        return frame if success and frame is not None else None

    if uploaded_image is None:
        return None
    return uploaded_image.copy()


async def get_processed_result() -> dict[str, Any] | None:
    global last_measurement, processed_cache

    settings = snapshot_settings()
    key = cache_key(settings)
    now = time.monotonic()

    if processed_cache.get('key') == key and now - processed_cache.get('time', 0) < 0.15:
        return processed_cache['result']

    async with get_processing_lock():
        now = time.monotonic()
        if processed_cache.get('key') == key and now - processed_cache.get('time', 0) < 0.15:
            return processed_cache['result']

        frame = await read_source_frame(settings['mode_camera'])
        if frame is None:
            last_measurement = {
                'status': 'Keine Kamera oder kein Bild',
                'area': None,
                'convex_area': None,
                'damage_area': None,
                'damage_percent': None,
                'markers': 0,
            }
            return None

        result = await run.io_bound(process_frame, frame, settings)
        last_measurement = result['measurement']
        processed_cache = {
            'key': key,
            'time': time.monotonic(),
            'result': result,
        }
        return result


@app.get('/video/{view}')
async def grab_video_frame(view: str) -> Response:
    if view not in {'full', 'cropped', 'mask', 'result'}:
        return placeholder

    processed = await get_processed_result()
    if processed is None:
        return placeholder

    frame = processed['images'].get(view)
    if frame is None:
        return placeholder

    jpeg = await run.io_bound(convert, frame)
    if not jpeg:
        return placeholder
    return Response(content=jpeg, media_type='image/jpeg')


def setup() -> None:
    global video_capture
    video_capture = cv2.VideoCapture(0)


@ui.page('/')
def page() -> None:
    dark = ui.dark_mode()

    with ui.row().classes('gap-4 items-start'):
        with ui.column().classes('w-200 items-stretch'):
            with ui.card().props('flat bordered').classes('w-200 items-stretch'):
                with ui.grid(columns=2).classes('w-full gap-2'):
                    with ui.row().classes('col-span-full items-center justify-between'):
                        ui.label('Fullframe')
                        freeze_button = ui.button('Freeze')
                    full_image = ui.interactive_image('/video/full').classes('border-none w-full col-span-full')

                    async def handle_freeze_click() -> None:
                        await toggle_freeze(freeze_button, full_image)

                    freeze_button.on('click', handle_freeze_click)

                    ui.label('Cropped')
                    ui.label('Result')
                    cropped_image = ui.interactive_image('/video/cropped').classes('border-none w-full')
                    result_image = ui.interactive_image('/video/result').classes('border-none w-full')

                    ui.label('Damage Mask').classes('col-span-full')
                    masked_image = ui.interactive_image('/video/mask').classes('border-none w-full col-span-full')

                for image in (full_image, cropped_image, result_image, masked_image):
                    ui.timer(interval=0.2, callback=image.force_reload)

        with ui.column().classes('w-100 items-stretch'):
            with ui.card().props('flat bordered'):
                with ui.row():
                    ui.select(langlist, label=text('select_language', 'Sprache'), on_change=load_language, value=sellang)
                    ui.switch(text('dark_mode_switch', 'Dunkelmodus')).bind_value(dark)

            with ui.card().props('flat bordered').classes('items-stretch'):
                ui.label('Messwerte')
                area_label = ui.label('Gruene Flaeche: -')
                convex_label = ui.label('Convex Hull: -')
                damage_label = ui.label('Schaden: -')
                damage_percent_label = ui.label('Schaden: - %')
                status_label = ui.label('Status: -')
                ui.timer(
                    interval=0.5,
                    callback=lambda: update_measurement_labels(
                        area_label,
                        convex_label,
                        damage_label,
                        damage_percent_label,
                        status_label,
                    ),
                )

            with ui.card().props('flat bordered'):
                ui.label(text('label_settings', 'Einstellungen'))

                with ui.card().props('flat bordered'):
                    with ui.expansion(text('label_basic_settings', 'Grundeinstellungen')).classes('w-80'):
                        mode = ui.checkbox(
                            text('mode_checkbox', 'Modus Kamera/Bild'),
                            value=app_settings.mode_camera,
                        )
                        mode.tooltip(text('mode_checkbox_tooltip', 'Modus zwischen Kamera und Bild wechseln'))
                        mode.bind_value(app_settings, 'mode_camera')

                        ui.upload(
                            label=text('image_upload', 'Bild'),
                            max_files=1,
                            on_upload=handle_upload,
                            on_rejected=lambda _: ui.notify(text('warn_file_upload_fail', 'Fehler beim Hochladen der Datei')),
                        ).classes('w-70').props('flat bordered').tooltip(text('image_upload_tooltip', 'Bild von Festplatte auswaehlen'))

                        physwidth = ui.number(
                            text('physwidth_input', 'Physische Breite'),
                            value=app_settings.phys_width,
                            min=0.001,
                            step=0.1,
                        )
                        physwidth.tooltip(text('physwidth_input_tooltip', 'Physische Breite zwischen den Markern'))
                        physwidth.bind_value(app_settings, 'phys_width')

                        physheight = ui.number(
                            text('physheight_input', 'Physische Hoehe'),
                            value=app_settings.phys_height,
                            min=0.001,
                            step=0.1,
                        )
                        physheight.tooltip(text('physheight_input_tooltip', 'Physische Hoehe zwischen den Markern'))
                        physheight.bind_value(app_settings, 'phys_height')

                        digwidth = ui.number(
                            text('digwidth_input', 'Digitale Aufloesung'),
                            value=app_settings.dig_width,
                            min=100,
                            max=2500,
                            step=50,
                        )
                        digwidth.tooltip(text('digwidth_input_tooltip', 'Digitale Aufloesung des Zuschnitts'))
                        digwidth.bind_value(app_settings, 'dig_width')

                with ui.card().props('flat bordered'):
                    with ui.expansion(text('label_filter_settings', 'Filtereinstellungen')).classes('w-80'):
                        kernelsize = ui.number(
                            label=text('kernelsize_input', 'Kernelgroesse'),
                            value=app_settings.kernel_size,
                            min=1,
                            max=100,
                            step=1,
                        )
                        kernelsize.tooltip(text('kernelsize_input_tooltip', 'Groesse der Matrix fuer die Rauschfilterung'))
                        kernelsize.bind_value(app_settings, 'kernel_size')

                        ui.color_input(
                            label=text('lower_input', 'Untere Farbgrenze'),
                            value=hsv_to_hex(app_settings.lower_hsv),
                            on_change=lambda event: update_hsv_setting('lower_hsv', event.value),
                        ).tooltip(text('lower_input_tooltip', 'Untere Farbgrenze fuer den Filter'))

                        ui.color_input(
                            label=text('upper_input', 'Obere Farbgrenze'),
                            value=hsv_to_hex(app_settings.upper_hsv),
                            on_change=lambda event: update_hsv_setting('upper_hsv', event.value),
                        ).tooltip(text('upper_input_tooltip', 'Obere Farbgrenze fuer den Filter'))

                with ui.card().props('flat bordered').classes('items-stretch'):
                    with ui.expansion(text('label_debug_settings', 'Debug Einstellungen')).classes('w-80'):
                        markers = ui.checkbox(text('marker_checkbox', 'Marker anzeigen'), value=app_settings.draw_marker)
                        markers.tooltip(text('marker_checkbox_tooltip', 'Erkannte Marker anzeigen'))
                        markers.bind_value(app_settings, 'draw_marker')

                        bounds = ui.checkbox(text('bound_checkbox', 'Flaeche markieren'), value=app_settings.draw_bound)
                        bounds.tooltip(text('bound_checkbox_tooltip', 'Flaeche zwischen den Markern anzeigen'))
                        bounds.bind_value(app_settings, 'draw_bound')

                        contours = ui.checkbox(text('contours_checkbox', 'Umrandung'), value=app_settings.draw_contours)
                        contours.tooltip(text('contours_checkbox_tooltip', 'Erkannte Kanten anzeigen'))
                        contours.bind_value(app_settings, 'draw_contours')

                        convex = ui.checkbox(text('convex_checkbox', 'Convex Hull'), value=app_settings.draw_convex)
                        convex.tooltip(text('convex_checkbox_tooltip', 'Convex Hull anzeigen'))
                        convex.bind_value(app_settings, 'draw_convex')


def update_measurement_labels(
    area_label: ui.label,
    convex_label: ui.label,
    damage_label: ui.label,
    damage_percent_label: ui.label,
    status_label: ui.label,
) -> None:
    area = last_measurement.get('area')
    convex_area = last_measurement.get('convex_area')
    damage_area = last_measurement.get('damage_area')
    damage_percent = last_measurement.get('damage_percent')
    area_label.set_text(f'Gruene Flaeche: {area:.3f} cm2' if area is not None else 'Gruene Flaeche: -')
    convex_label.set_text(f'Convex Hull: {convex_area:.3f} cm2' if convex_area is not None else 'Convex Hull: -')
    damage_label.set_text(f'Schaden: {damage_area:.3f} cm2' if damage_area is not None else 'Schaden: -')
    damage_percent_label.set_text(
        f'Schaden: {damage_percent:.1f} %' if damage_percent is not None else 'Schaden: - %'
    )
    status_label.set_text(f"Status: {last_measurement.get('status', '-')}")


async def toggle_freeze(button: ui.button, full_image: ui.interactive_image) -> None:
    global freeze_enabled, frozen_frame, input_revision, processed_cache

    async with get_processing_lock():
        if freeze_enabled:
            freeze_enabled = False
            frozen_frame = None
            button.set_text('Freeze')
            input_revision += 1
            processed_cache = {}
            full_image.force_reload()
            ui.notify('Livebild aktiv')
            return

        if app_settings.mode_camera:
            if video_capture is None or not video_capture.isOpened():
                ui.notify('Kamera nicht verfuegbar')
                return
            success, frame = await run.io_bound(video_capture.read)
            if not success or frame is None:
                ui.notify('Kein Kamerabild empfangen')
                return
            frozen_frame = frame.copy()
        elif uploaded_image is not None:
            frozen_frame = uploaded_image.copy()
        else:
            ui.notify('Kein Bild zum Einfrieren')
            return

        freeze_enabled = True
        button.set_text('Live')
        input_revision += 1
        processed_cache = {}
        full_image.force_reload()
        ui.notify('Frame eingefroren')


def load_language(event: Any) -> None:
    global language, sellang
    sellang = event.value
    language = read_language(event.value)
    ui.run_javascript('location.reload();')


async def handle_upload(event: Any) -> None:
    global uploaded_image, input_revision, processed_cache

    content = await event.file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)

    if image is None:
        ui.notify(text('warn_file_upload_fail', 'Fehler beim Hochladen der Datei'))
        return

    uploaded_image = image
    input_revision += 1
    processed_cache = {}
    app_settings.mode_camera = False
    ui.notify('Bild geladen')


async def disconnect() -> None:
    for client_id in list(Client.instances):
        await core.sio.disconnect(client_id)


def handle_sigint(signum: int, frame: Any) -> None:
    ui.timer(0.1, disconnect, once=True)
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)


async def cleanup() -> None:
    await disconnect()
    if video_capture is not None:
        video_capture.release()


app.on_startup(setup)
app.on_shutdown(cleanup)
signal.signal(signal.SIGINT, handle_sigint)

if __name__ in {'__main__', '__mp_main__'}:
    ui.run()
