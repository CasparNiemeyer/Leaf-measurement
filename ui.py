import asyncio
import base64
import csv
import itertools
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
from fastapi import Request, Response
from nicegui import Client, app, core, run, ui


BASE_DIR = Path(__file__).resolve().parent
LANG_DIR = BASE_DIR / 'lang'
ARCHIVE_DIR = BASE_DIR / 'archive'
ARCHIVE_IMAGE_DIR = ARCHIVE_DIR / 'images'
ARCHIVE_CSV = ARCHIVE_DIR / 'measurements.csv'
ARCHIVE_LOCK = asyncio.Lock()

ARCHIVE_CSV_FIELDS = [
    'timestamp',
    'session_id',
    'status',
    'markers',
    'green_area_cm2',
    'convex_hull_cm2',
    'damage_area_cm2',
    'damage_percent',
    'mode',
    'frozen',
    'phys_width_cm',
    'phys_height_cm',
    'dig_width_px',
    'kernel_size',
    'lower_hsv',
    'upper_hsv',
    'manual_damage_enabled',
    'auto_edge_damage_enabled',
    'manual_limit_to_leaf',
    'manual_leaf_shrink_px',
    'show_auto_damage_on_cropped',
    'full_image',
    'cropped_image',
    'result_image',
    'mask_image',
]

# NiceGUI creates a ProcessPoolExecutor on startup for run.cpu_bound.
# This app only uses run.io_bound, and some Windows setups block multiprocessing pipes.
run.setup = lambda: None


@dataclass
class MeasurementSettings:
    mode_camera: bool = True
    phys_width: float = 13.4
    phys_height: float = 13.4
    dig_width: int = 700
    kernel_size: int = 6
    lower_hsv: tuple[int, int, int] = (0, 30, 40)
    upper_hsv: tuple[int, int, int] = (179, 255, 255)
    draw_marker: bool = True
    draw_bound: bool = True
    draw_contours: bool = True
    draw_convex: bool = True


@dataclass
class SessionState:
    settings: MeasurementSettings
    browser_frame: np.ndarray | None = None
    browser_frame_jpeg: bytes | None = None
    uploaded_image: np.ndarray | None = None
    frozen_frame: np.ndarray | None = None
    manual_damage_mask: np.ndarray | None = None
    manual_correct_mask: np.ndarray | None = None
    manual_exclude_mask: np.ndarray | None = None
    manual_damage_enabled: bool = True
    manual_limit_to_leaf: bool = False
    manual_leaf_shrink_px: int = 0
    show_auto_damage_on_cropped: bool = False
    auto_edge_damage_enabled: bool = True
    manual_damage_revision: int = 0
    manual_brush_size: int = 18
    freeze_enabled: bool = False
    input_revision: int = 0
    processed_cache: dict[str, Any] | None = None
    processing_lock: asyncio.Lock | None = None
    last_measurement: dict[str, Any] | None = None
    last_seen: float = 0.0


def default_measurement() -> dict[str, Any]:
    return {
        'status': 'Noch kein Bild verarbeitet',
        'area': None,
        'convex_area': None,
        'damage_area': None,
        'damage_percent': None,
        'markers': 0,
    }


sessions: dict[str, SessionState] = {}

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
        setattr(default_session().settings, attr, hsv)


def default_session() -> SessionState:
    state = sessions.get('default')
    if state is None:
        state = create_session('default')
    return state


def create_session(session_id: str) -> SessionState:
    state = SessionState(
        settings=MeasurementSettings(),
        processed_cache={},
        last_measurement=default_measurement(),
        last_seen=time.monotonic(),
    )
    sessions[session_id] = state
    return state


def get_session(session_id: str) -> SessionState:
    state = sessions.get(session_id)
    if state is None:
        state = create_session(session_id)
    state.last_seen = time.monotonic()
    return state


def cleanup_sessions(max_age_seconds: int = 600) -> None:
    now = time.monotonic()
    for session_id, state in list(sessions.items()):
        if session_id != 'default' and now - state.last_seen > max_age_seconds:
            del sessions[session_id]


def update_hsv_setting_for(state: SessionState, attr: str, value: str | None) -> None:
    hsv = hex_to_hsv(value)
    if hsv is not None:
        setattr(state.settings, attr, hsv)
        state.processed_cache = {}
        state.input_revision += 1


def update_hsv_range(state: SessionState, index: int, value: Any) -> None:
    try:
        if isinstance(value, dict):
            lower_value = value.get('min')
            upper_value = value.get('max')
        else:
            lower_value, upper_value = value
        lower_numeric = int(float(lower_value))
        upper_numeric = int(float(upper_value))
    except (TypeError, ValueError):
        return

    max_value = 179 if index == 0 else 255
    lower_numeric = max(0, min(max_value, lower_numeric))
    upper_numeric = max(0, min(max_value, upper_numeric))
    if lower_numeric > upper_numeric:
        lower_numeric, upper_numeric = upper_numeric, lower_numeric

    lower = list(state.settings.lower_hsv)
    upper = list(state.settings.upper_hsv)
    lower[index] = lower_numeric
    upper[index] = upper_numeric

    state.settings.lower_hsv = tuple(lower)
    state.settings.upper_hsv = tuple(upper)
    state.processed_cache = {}
    state.input_revision += 1


def snapshot_settings(state: SessionState | None = None) -> dict[str, Any]:
    state = state or default_session()
    app_settings = state.settings
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
        'freeze_enabled': state.freeze_enabled,
        'input_revision': state.input_revision,
        'manual_damage_enabled': state.manual_damage_enabled,
        'manual_limit_to_leaf': state.manual_limit_to_leaf,
        'manual_leaf_shrink_px': state.manual_leaf_shrink_px,
        'show_auto_damage_on_cropped': state.show_auto_damage_on_cropped,
        'auto_edge_damage_enabled': state.auto_edge_damage_enabled,
        'manual_damage_revision': state.manual_damage_revision,
    }


def convert(frame: np.ndarray) -> bytes:
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    success, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes() if success else b''


def write_archive_entry(
    session_id: str,
    settings: dict[str, Any],
    measurement: dict[str, Any],
    images: dict[str, np.ndarray],
) -> dict[str, str]:
    ARCHIVE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    basename = f'{timestamp}_{session_id[:8]}'

    image_paths: dict[str, str] = {}
    for view in ('full', 'cropped', 'result', 'mask'):
        frame = images.get(view)
        if frame is None:
            continue
        path = ARCHIVE_IMAGE_DIR / f'{basename}_{view}.jpg'
        cv2.imwrite(str(path), frame)
        image_paths[f'{view}_image'] = str(path.relative_to(BASE_DIR))

    row = {
        'timestamp': timestamp,
        'session_id': session_id,
        'status': measurement.get('status', ''),
        'markers': measurement.get('markers', ''),
        'green_area_cm2': measurement.get('area', ''),
        'convex_hull_cm2': measurement.get('convex_area', ''),
        'damage_area_cm2': measurement.get('damage_area', ''),
        'damage_percent': measurement.get('damage_percent', ''),
        'mode': 'camera' if settings.get('mode_camera') else 'image',
        'frozen': settings.get('freeze_enabled', False),
        'phys_width_cm': settings.get('phys_width', ''),
        'phys_height_cm': settings.get('phys_height', ''),
        'dig_width_px': settings.get('dig_width', ''),
        'kernel_size': settings.get('kernel_size', ''),
        'lower_hsv': ';'.join(map(str, settings.get('lower_hsv', ()))),
        'upper_hsv': ';'.join(map(str, settings.get('upper_hsv', ()))),
        'manual_damage_enabled': settings.get('manual_damage_enabled', ''),
        'auto_edge_damage_enabled': settings.get('auto_edge_damage_enabled', ''),
        'manual_limit_to_leaf': settings.get('manual_limit_to_leaf', ''),
        'manual_leaf_shrink_px': settings.get('manual_leaf_shrink_px', ''),
        'show_auto_damage_on_cropped': settings.get('show_auto_damage_on_cropped', ''),
        'full_image': image_paths.get('full_image', ''),
        'cropped_image': image_paths.get('cropped_image', ''),
        'result_image': image_paths.get('result_image', ''),
        'mask_image': image_paths.get('mask_image', ''),
    }

    csv_exists = ARCHIVE_CSV.exists()
    with open(ARCHIVE_CSV, 'a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=ARCHIVE_CSV_FIELDS)
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)

    return image_paths


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


def is_reasonable_quad(points: np.ndarray, image_shape: tuple[int, int]) -> bool:
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False

    height, width = image_shape
    margin_x = width * 0.08
    margin_y = height * 0.08
    if (
        np.any(points[:, 0] < -margin_x)
        or np.any(points[:, 0] > width + margin_x)
        or np.any(points[:, 1] < -margin_y)
        or np.any(points[:, 1] > height + margin_y)
    ):
        return False

    polygon_area = abs(cv2.contourArea(points.astype(np.float32)))
    frame_area = float(width * height)
    if polygon_area < frame_area * 0.04 or polygon_area > frame_area * 0.95:
        return False

    side_lengths = [
        float(np.linalg.norm(points[1] - points[0])),
        float(np.linalg.norm(points[2] - points[1])),
        float(np.linalg.norm(points[2] - points[3])),
        float(np.linalg.norm(points[3] - points[0])),
    ]
    if min(side_lengths) < min(width, height) * 0.08:
        return False

    longest = max(side_lengths)
    shortest = min(side_lengths)
    if longest / max(1.0, shortest) > 4.0:
        return False

    opposite_ratio_a = side_lengths[0] / max(1.0, side_lengths[2])
    opposite_ratio_b = side_lengths[1] / max(1.0, side_lengths[3])
    if not (0.35 <= opposite_ratio_a <= 2.85 and 0.35 <= opposite_ratio_b <= 2.85):
        return False

    contour = points.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour.astype(np.int32)):
        return False

    return True


def aruco_parameters() -> Any:
    parameters = (
        cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, 'DetectorParameters')
        else cv2.aruco.DetectorParameters_create()
    )
    parameter_updates = {
        'adaptiveThreshWinSizeMin': 3,
        'adaptiveThreshWinSizeMax': 53,
        'adaptiveThreshWinSizeStep': 4,
        'adaptiveThreshConstant': 7,
        'minMarkerPerimeterRate': 0.015,
        'maxMarkerPerimeterRate': 4.0,
        'polygonalApproxAccuracyRate': 0.04,
        'minCornerDistanceRate': 0.03,
        'minDistanceToBorder': 2,
        'cornerRefinementMethod': getattr(cv2.aruco, 'CORNER_REFINE_SUBPIX', 1),
        'cornerRefinementWinSize': 5,
        'cornerRefinementMaxIterations': 50,
        'cornerRefinementMinAccuracy': 0.01,
        'errorCorrectionRate': 0.8,
    }
    for name, value in parameter_updates.items():
        if hasattr(parameters, name):
            setattr(parameters, name, value)
    return parameters


def fallback_aruco_parameters() -> Any:
    parameters = aruco_parameters()
    fallback_updates = {
        'adaptiveThreshWinSizeMax': 153,
        'minMarkerPerimeterRate': 0.005,
        'polygonalApproxAccuracyRate': 0.08,
        'minCornerDistanceRate': 0.01,
        'errorCorrectionRate': 1.0,
    }
    for name, value in fallback_updates.items():
        if hasattr(parameters, name):
            setattr(parameters, name, value)
    return parameters


def run_aruco_detector(
    gray: np.ndarray,
    aruco_dict: Any,
    parameters: Any,
) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
    if hasattr(cv2.aruco, 'ArucoDetector'):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected

    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    return corners, ids, rejected


def scale_corners(corners: list[np.ndarray], scale: float) -> list[np.ndarray]:
    if scale == 1.0:
        return corners
    return [(np.asarray(corner, dtype=np.float32) / scale).astype(np.float32) for corner in corners]


def marker_area(corner: np.ndarray) -> float:
    points = np.asarray(corner, dtype=np.float32).reshape(4, 2)
    return abs(cv2.contourArea(points))


def marker_center(corner: np.ndarray) -> np.ndarray:
    return np.asarray(corner, dtype=np.float32).reshape(4, 2).mean(axis=0)


def marker_shape_is_plausible(corner: np.ndarray, image_shape: tuple[int, int]) -> bool:
    points = np.asarray(corner, dtype=np.float32).reshape(4, 2)
    height, width = image_shape
    frame_area = float(width * height)
    area = marker_area(points)
    if area < max(80.0, frame_area * 0.00008) or area > frame_area * 0.08:
        return False

    x, y, box_width, box_height = cv2.boundingRect(points.astype(np.int32))
    if box_width < 8 or box_height < 8:
        return False

    aspect = box_width / max(1, box_height)
    if not (0.35 <= aspect <= 2.85):
        return False

    side_lengths = [
        float(np.linalg.norm(points[1] - points[0])),
        float(np.linalg.norm(points[2] - points[1])),
        float(np.linalg.norm(points[3] - points[2])),
        float(np.linalg.norm(points[0] - points[3])),
    ]
    if min(side_lengths) < 6:
        return False
    if max(side_lengths) / max(1.0, min(side_lengths)) > 3.2:
        return False

    margin_x = width * 0.015
    margin_y = height * 0.015
    if (
        np.any(points[:, 0] < -margin_x)
        or np.any(points[:, 0] > width + margin_x)
        or np.any(points[:, 1] < -margin_y)
        or np.any(points[:, 1] > height + margin_y)
    ):
        return False

    return True


def select_wanted_markers(
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    image_shape: tuple[int, int],
) -> tuple[list[np.ndarray], np.ndarray | None]:
    if ids is None:
        return [], None

    wanted_ids = {0, 1, 2, 3}
    by_id: dict[int, list[np.ndarray]] = {marker_id: [] for marker_id in wanted_ids}
    for marker_id, corner in zip(ids.flatten(), corners):
        marker_id_int = int(marker_id)
        if marker_id_int not in wanted_ids:
            continue
        if marker_shape_is_plausible(corner, image_shape):
            by_id[marker_id_int].append(np.asarray(corner, dtype=np.float32))

    if any(not by_id[marker_id] for marker_id in wanted_ids):
        return [], None

    selected: list[np.ndarray] = []
    for marker_id in sorted(wanted_ids):
        options = by_id[marker_id]
        selected.append(max(options, key=marker_area))

    areas = np.asarray([marker_area(corner) for corner in selected], dtype=np.float32)
    median_area = float(np.median(areas))
    if median_area <= 0:
        return [], None
    if float(np.max(areas) / max(1.0, np.min(areas))) > 6.5:
        return [], None

    centers = np.asarray([marker_center(corner) for corner in selected], dtype=np.float32)
    ordered_centers = order_points(centers)
    if not is_reasonable_quad(ordered_centers, image_shape):
        return [], None

    center = ordered_centers.mean(axis=0)
    distances = np.linalg.norm(ordered_centers - center, axis=1)
    if float(np.max(distances) / max(1.0, np.median(distances))) > 2.25:
        return [], None

    ordered_corners: list[np.ndarray] = []
    for ordered_center in ordered_centers:
        index = int(np.argmin(np.linalg.norm(centers - ordered_center, axis=1)))
        ordered_corners.append(selected[index].astype(np.float32))

    synthetic_ids = np.asarray([[0], [1], [2], [3]], dtype=np.int32)
    return ordered_corners, synthetic_ids


def synthesize_missing_marker_from_decoded(
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    image_shape: tuple[int, int],
) -> tuple[list[np.ndarray], np.ndarray | None]:
    if ids is None:
        return [], None

    wanted_ids = {0, 1, 2, 3}
    by_id: dict[int, np.ndarray] = {}
    for marker_id, corner in zip(ids.flatten(), corners):
        marker_id_int = int(marker_id)
        if marker_id_int not in wanted_ids or not marker_shape_is_plausible(corner, image_shape):
            continue
        current = by_id.get(marker_id_int)
        if current is None or marker_area(corner) > marker_area(current):
            by_id[marker_id_int] = np.asarray(corner, dtype=np.float32)

    if len(by_id) != 3:
        return [], None

    missing_index = next(marker_id for marker_id in sorted(wanted_ids) if marker_id not in by_id)
    centers_by_id = {marker_id: marker_center(corner) for marker_id, corner in by_id.items()}
    if missing_index == 0:
        center = centers_by_id[1] + centers_by_id[3] - centers_by_id[2]
    elif missing_index == 1:
        center = centers_by_id[0] + centers_by_id[2] - centers_by_id[3]
    elif missing_index == 2:
        center = centers_by_id[1] + centers_by_id[3] - centers_by_id[0]
    else:
        center = centers_by_id[0] + centers_by_id[2] - centers_by_id[1]

    sizes = [np.sqrt(marker_area(corner)) for corner in by_id.values()]
    half = max(6.0, float(np.median(sizes)) * 0.35)
    synthetic = np.asarray([[
        [center[0] - half, center[1] - half],
        [center[0] + half, center[1] - half],
        [center[0] + half, center[1] + half],
        [center[0] - half, center[1] + half],
    ]], dtype=np.float32)

    completed = [by_id.get(marker_id, synthetic).astype(np.float32) for marker_id in sorted(wanted_ids)]
    centers = np.asarray([marker_center(corner) for corner in completed], dtype=np.float32)
    if not is_reasonable_quad(order_points(centers), image_shape):
        return [], None

    return completed, np.asarray([[0], [1], [2], [3]], dtype=np.int32)


def fallback_marker_candidates(
    image_shape: tuple[int, int],
    candidates: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray | None]:
    height, width = image_shape
    frame_area = float(width * height)
    min_area = max(250.0, frame_area * 0.00045)
    max_area = frame_area * 0.08
    usable: list[dict[str, Any]] = []

    for candidate in candidates:
        points = np.asarray(candidate, dtype=np.float32).reshape(4, 2)
        area = abs(cv2.contourArea(points))
        if not (min_area <= area <= max_area):
            continue

        x, y, box_width, box_height = cv2.boundingRect(points.astype(np.int32))
        if box_width < 20 or box_height < 20:
            continue

        aspect = box_width / max(1, box_height)
        if not (0.25 <= aspect <= 4.0):
            continue

        usable.append({
            'points': points.reshape(1, 4, 2),
            'center': points.mean(axis=0),
            'area': area,
        })

    diagonal = float(np.hypot(width, height))
    clustered: list[dict[str, Any]] = []
    for item in sorted(usable, key=lambda candidate: candidate['area'], reverse=True):
        if any(np.linalg.norm(item['center'] - other['center']) < diagonal * 0.055 for other in clustered):
            continue
        clustered.append(item)

    if len(clustered) < 3:
        return [], None

    if len(clustered) >= 4:
        best_group: tuple[float, tuple[dict[str, Any], ...]] | None = None
        search_pool = sorted(clustered, key=lambda candidate: candidate['area'], reverse=True)[:24]
        for group in itertools.combinations(search_pool, 4):
            centers = np.asarray([item['center'] for item in group], dtype=np.float32)
            ordered_centers = order_points(centers)
            if not is_reasonable_quad(ordered_centers, image_shape):
                continue

            areas = np.asarray([item['area'] for item in group], dtype=np.float32)
            area_ratio = float(np.max(areas) / max(1.0, np.min(areas)))
            if area_ratio > 8.0:
                continue

            side_lengths = [
                float(np.linalg.norm(ordered_centers[1] - ordered_centers[0])),
                float(np.linalg.norm(ordered_centers[2] - ordered_centers[1])),
                float(np.linalg.norm(ordered_centers[2] - ordered_centers[3])),
                float(np.linalg.norm(ordered_centers[3] - ordered_centers[0])),
            ]
            side_ratio = max(side_lengths) / max(1.0, min(side_lengths))
            opposite_ratio_a = side_lengths[0] / max(1.0, side_lengths[2])
            opposite_ratio_b = side_lengths[1] / max(1.0, side_lengths[3])
            polygon_area = abs(cv2.contourArea(ordered_centers))
            score = (
                np.log(area_ratio)
                + abs(np.log(max(0.001, opposite_ratio_a)))
                + abs(np.log(max(0.001, opposite_ratio_b)))
                + max(0.0, side_ratio - 2.0) * 0.25
                - (polygon_area / max(1.0, frame_area)) * 0.15
            )
            if best_group is None or score < best_group[0]:
                best_group = (float(score), group)

        if best_group is not None:
            group = best_group[1]
            centers = np.asarray([item['center'] for item in group], dtype=np.float32)
            ordered_centers = order_points(centers)
            ordered_items: list[dict[str, Any]] = []
            used: set[int] = set()
            for ordered_center in ordered_centers:
                distances = np.linalg.norm(centers - ordered_center, axis=1)
                for index in np.argsort(distances):
                    index_int = int(index)
                    if index_int not in used:
                        used.add(index_int)
                        ordered_items.append(group[index_int])
                        break

            synthetic_ids = np.asarray([[0], [1], [2], [3]], dtype=np.int32)
            return [item['points'].astype(np.float32) for item in ordered_items], synthetic_ids

    targets = np.asarray([
        [0.0, 0.0],
        [float(width), 0.0],
        [float(width), float(height)],
        [0.0, float(height)],
    ], dtype=np.float32)

    slots: list[dict[str, Any] | None] = [None, None, None, None]
    slot_scores = [float('inf')] * 4
    for item in clustered:
        distances = np.linalg.norm(targets - item['center'], axis=1) / max(1.0, diagonal)
        target_index = int(np.argmin(distances))
        score = float(distances[target_index]) - min(0.08, item['area'] / frame_area * 8.0)
        if score < slot_scores[target_index]:
            slots[target_index] = item
            slot_scores[target_index] = score

    missing = [index for index, item in enumerate(slots) if item is None]
    if len(missing) > 1:
        return [], None

    if len(missing) == 1:
        present_centers = [item['center'] for item in slots if item is not None]
        average_size = max(12.0, np.mean([
            np.sqrt(item['area']) for item in slots if item is not None
        ]) * 0.35)
        missing_index = missing[0]
        if missing_index == 0 and slots[1] is not None and slots[2] is not None and slots[3] is not None:
            center = slots[1]['center'] + slots[3]['center'] - slots[2]['center']
        elif missing_index == 1 and slots[0] is not None and slots[2] is not None and slots[3] is not None:
            center = slots[0]['center'] + slots[2]['center'] - slots[3]['center']
        elif missing_index == 2 and slots[0] is not None and slots[1] is not None and slots[3] is not None:
            center = slots[1]['center'] + slots[3]['center'] - slots[0]['center']
        elif missing_index == 3 and slots[0] is not None and slots[1] is not None and slots[2] is not None:
            center = slots[0]['center'] + slots[2]['center'] - slots[1]['center']
        else:
            center = np.mean(np.asarray(present_centers, dtype=np.float32), axis=0)

        half = average_size / 2.0
        points = np.asarray([[
            [center[0] - half, center[1] - half],
            [center[0] + half, center[1] - half],
            [center[0] + half, center[1] + half],
            [center[0] - half, center[1] + half],
        ]], dtype=np.float32)
        slots[missing_index] = {
            'points': points,
            'center': np.asarray(center, dtype=np.float32),
            'area': average_size * average_size,
        }

    if any(item is None for item in slots):
        return [], None

    selected_corners = [item['points'].astype(np.float32) for item in slots if item is not None]
    centers = np.asarray([corner.reshape(4, 2).mean(axis=0) for corner in selected_corners], dtype=np.float32)
    if not is_reasonable_quad(order_points(centers), image_shape):
        return [], None

    synthetic_ids = np.asarray([[0], [1], [2], [3]], dtype=np.int32)
    return selected_corners, synthetic_ids


def detect_aruco(frame: np.ndarray) -> tuple[list[np.ndarray], np.ndarray | None, str | None]:
    if not hasattr(cv2, 'aruco'):
        return [], None, 'OpenCV ArUco Modul nicht gefunden'

    original_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    max_detection_side = 1500
    max_side = max(original_gray.shape[:2])
    detection_scale = min(1.0, max_detection_side / max(1, max_side))
    if detection_scale < 1.0:
        base_gray = cv2.resize(
            original_gray,
            (round(original_gray.shape[1] * detection_scale), round(original_gray.shape[0] * detection_scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        base_gray = original_gray

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = aruco_parameters()
    fallback_parameters = fallback_aruco_parameters()

    variants: list[tuple[np.ndarray, float]] = [(base_gray, 1.0)]
    equalized = cv2.equalizeHist(base_gray)
    variants.append((equalized, 1.0))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(base_gray)
    variants.append((clahe, 1.0))

    if max(base_gray.shape[:2]) < 1200:
        variants.append((cv2.resize(base_gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC), 1.5))
        variants.append((cv2.resize(clahe, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC), 1.5))

    best_corners: list[np.ndarray] = []
    best_ids: np.ndarray | None = None
    best_score = -1
    wanted_ids = {0, 1, 2, 3}
    decoded_candidates: list[tuple[list[np.ndarray], np.ndarray]] = []
    fallback_candidates: list[np.ndarray] = []

    for gray, scale in variants:
        corners, ids, rejected = run_aruco_detector(gray, aruco_dict, parameters)
        total_scale = scale * detection_scale
        scaled_corners = scale_corners(corners, total_scale)
        fallback_candidates.extend(scale_corners(list(corners) + list(rejected), total_scale))
        if ids is None:
            score = 0
        else:
            found_wanted = {int(marker_id) for marker_id in ids.flatten() if int(marker_id) in wanted_ids}
            score = len(found_wanted) * 10 + len(ids)
            decoded_candidates.append((scaled_corners, ids))

        if score > best_score:
            best_score = score
            best_corners = scaled_corners
            best_ids = ids

        if ids is not None and wanted_ids.issubset({int(marker_id) for marker_id in ids.flatten()}):
            selected_corners, selected_ids = select_wanted_markers(scaled_corners, ids, original_gray.shape[:2])
            if selected_ids is not None:
                return selected_corners, selected_ids, None

    for gray, scale in variants[:1]:
        corners, ids, rejected = run_aruco_detector(gray, aruco_dict, fallback_parameters)
        total_scale = scale * detection_scale
        fallback_candidates.extend(scale_corners(list(corners) + list(rejected), total_scale))
        if ids is not None:
            decoded_candidates.append((scale_corners(corners, total_scale), ids))

    for corners, ids in decoded_candidates:
        selected_corners, selected_ids = select_wanted_markers(corners, ids, original_gray.shape[:2])
        if selected_ids is not None:
            return selected_corners, selected_ids, None

    for corners, ids in decoded_candidates:
        fallback_corners, fallback_ids = synthesize_missing_marker_from_decoded(corners, ids, original_gray.shape[:2])
        if fallback_ids is not None:
            return fallback_corners, fallback_ids, None

    fallback_corners, fallback_ids = fallback_marker_candidates(original_gray.shape[:2], fallback_candidates)
    if fallback_ids is not None:
        return fallback_corners, fallback_ids, None

    return [], None, None


def process_frame(
    frame: np.ndarray,
    settings: dict[str, Any],
    manual_damage_mask: np.ndarray | None = None,
    manual_correct_mask: np.ndarray | None = None,
    manual_exclude_mask: np.ndarray | None = None,
) -> dict[str, Any]:
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
    if not is_reasonable_quad(sorted_points, frame.shape[:2]):
        message = 'Markerpunkte unplausibel - bitte Marker vollstaendig sichtbar halten'
        measurement['status'] = message
        images['full'] = display_frame
        images['cropped'] = blank_frame(message, dig_width, dig_width)
        images['mask'] = blank_frame(message, dig_width, dig_width)
        images['result'] = blank_frame(message, dig_width, dig_width)
        return {'images': images, 'measurement': measurement}

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

    contour_mask = np.zeros_like(opened_mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    hull_mask = np.zeros_like(opened_mask)
    cv2.drawContours(hull_mask, [hull], -1, 255, thickness=-1)
    green_mask = cv2.bitwise_and(opened_mask, hull_mask)

    inner_damage_mask = cv2.bitwise_and(contour_mask, cv2.bitwise_not(green_mask))
    edge_damage_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(contour_mask))

    damage_kernel = np.ones((max(1, kernel_size), max(1, kernel_size)), np.uint8)
    inner_damage_mask = cv2.morphologyEx(inner_damage_mask, cv2.MORPH_OPEN, damage_kernel)
    edge_damage_mask = cv2.morphologyEx(edge_damage_mask, cv2.MORPH_OPEN, damage_kernel)

    damage_mask = inner_damage_mask.copy()
    if settings.get('auto_edge_damage_enabled', True):
        damage_mask = cv2.bitwise_or(damage_mask, edge_damage_mask)
    auto_damage_mask = damage_mask.copy()

    cropped_preview = cropped.copy()
    if settings.get('show_auto_damage_on_cropped'):
        red_overlay = np.zeros_like(cropped_preview)
        red_overlay[auto_damage_mask > 0] = (0, 0, 255)
        cropped_preview = cv2.addWeighted(cropped_preview, 1.0, red_overlay, 0.45, 0)

    manual_limit_mask = contour_mask.copy()
    shrink_px = max(0, min(250, int(settings.get('manual_leaf_shrink_px') or 0)))
    if shrink_px > 0:
        shrink_kernel = np.ones((shrink_px * 2 + 1, shrink_px * 2 + 1), np.uint8)
        manual_limit_mask = cv2.erode(manual_limit_mask, shrink_kernel, iterations=1)

    manual_mask_for_display = np.zeros_like(damage_mask)
    correct_mask_for_display = np.zeros_like(damage_mask)
    exclude_mask_for_display = np.zeros_like(damage_mask)
    if settings.get('manual_damage_enabled') and manual_damage_mask is not None:
        if manual_damage_mask.shape != damage_mask.shape:
            manual_damage_mask = cv2.resize(
                manual_damage_mask,
                (damage_mask.shape[1], damage_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        manual_mask_for_display = manual_damage_mask.copy()
        if settings.get('manual_limit_to_leaf'):
            manual_mask_for_display = cv2.bitwise_and(manual_mask_for_display, manual_limit_mask)
        damage_mask = cv2.bitwise_or(damage_mask, manual_mask_for_display)

    if settings.get('manual_damage_enabled') and manual_correct_mask is not None:
        if manual_correct_mask.shape != damage_mask.shape:
            manual_correct_mask = cv2.resize(
                manual_correct_mask,
                (damage_mask.shape[1], damage_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        correct_mask_for_display = manual_correct_mask.copy()
        if settings.get('manual_limit_to_leaf'):
            correct_mask_for_display = cv2.bitwise_and(correct_mask_for_display, manual_limit_mask)
        damage_mask = cv2.bitwise_and(damage_mask, cv2.bitwise_not(correct_mask_for_display))

    if settings.get('manual_damage_enabled') and manual_exclude_mask is not None:
        if manual_exclude_mask.shape != damage_mask.shape:
            manual_exclude_mask = cv2.resize(
                manual_exclude_mask,
                (damage_mask.shape[1], damage_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        exclude_mask_for_display = manual_exclude_mask.copy()
        if settings.get('manual_limit_to_leaf'):
            exclude_mask_for_display = cv2.bitwise_and(exclude_mask_for_display, manual_limit_mask)
        damage_mask = cv2.bitwise_and(damage_mask, cv2.bitwise_not(exclude_mask_for_display))

    measured_green_mask = cv2.bitwise_or(green_mask, correct_mask_for_display)
    measured_green_mask = cv2.bitwise_and(measured_green_mask, cv2.bitwise_not(exclude_mask_for_display))
    measured_hull_mask = cv2.bitwise_and(hull_mask, cv2.bitwise_not(exclude_mask_for_display))
    green_pixels = cv2.countNonZero(measured_green_mask)
    hull_pixels = cv2.countNonZero(measured_hull_mask)
    damage_pixels = cv2.countNonZero(damage_mask)

    area = round(green_pixels * pixel_area, 3)
    convex_area = round(hull_pixels * pixel_area, 3)
    damage_area = round(damage_pixels * pixel_area, 3)
    damage_percent = round((damage_pixels / hull_pixels) * 100, 1) if hull_pixels else 0.0

    mask_preview = cv2.cvtColor(damage_mask, cv2.COLOR_GRAY2BGR)
    result[damage_mask > 0] = (0, 0, 255)
    result[manual_mask_for_display > 0] = (0, 128, 255)
    result[correct_mask_for_display > 0] = (0, 255, 0)
    result[exclude_mask_for_display > 0] = (255, 0, 255)

    if settings['draw_convex']:
        cv2.drawContours(result, [hull], -1, (255, 0, 0), 2)
    if settings['draw_contours']:
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

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
        'cropped': cropped_preview,
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
        settings['manual_damage_enabled'],
        settings['manual_limit_to_leaf'],
        settings['manual_leaf_shrink_px'],
        settings['show_auto_damage_on_cropped'],
        settings['auto_edge_damage_enabled'],
        settings['manual_damage_revision'],
    )


def get_processing_lock(state: SessionState) -> asyncio.Lock:
    if state.processing_lock is None:
        state.processing_lock = asyncio.Lock()
    return state.processing_lock


async def read_source_frame(state: SessionState, use_camera: bool) -> np.ndarray | None:
    if use_camera:
        if state.freeze_enabled and state.frozen_frame is not None:
            return state.frozen_frame.copy()
        if state.browser_frame is None:
            return None
        return state.browser_frame.copy()

    if state.uploaded_image is None:
        return None
    return state.uploaded_image.copy()


async def get_processed_result(session_id: str) -> dict[str, Any] | None:
    state = get_session(session_id)
    if state.processed_cache is None:
        state.processed_cache = {}

    settings = snapshot_settings(state)
    key = cache_key(settings)
    now = time.monotonic()
    min_cache_age = 0.8 if settings['mode_camera'] and not settings['freeze_enabled'] else 0.12

    if state.processed_cache.get('key') == key and now - state.processed_cache.get('time', 0) < min_cache_age:
        return state.processed_cache['result']

    async with get_processing_lock(state):
        now = time.monotonic()
        if state.processed_cache.get('key') == key and now - state.processed_cache.get('time', 0) < min_cache_age:
            return state.processed_cache['result']

        frame = await read_source_frame(state, settings['mode_camera'])
        if frame is None:
            state.last_measurement = {
                'status': 'Keine Kamera oder kein Bild',
                'area': None,
                'convex_area': None,
                'damage_area': None,
                'damage_percent': None,
                'markers': 0,
            }
            return None

        manual_damage_mask = state.manual_damage_mask.copy() if state.manual_damage_mask is not None else None
        manual_correct_mask = state.manual_correct_mask.copy() if state.manual_correct_mask is not None else None
        manual_exclude_mask = state.manual_exclude_mask.copy() if state.manual_exclude_mask is not None else None
        result = await run.io_bound(
            process_frame,
            frame,
            settings,
            manual_damage_mask,
            manual_correct_mask,
            manual_exclude_mask,
        )
        state.last_measurement = result['measurement']
        state.processed_cache = {
            'key': key,
            'time': time.monotonic(),
            'result': result,
        }
        return result


@app.get('/video/{session_id}/{view}')
async def grab_video_frame(session_id: str, view: str) -> Response:
    if view not in {'full', 'cropped', 'mask', 'result'}:
        return placeholder

    state = get_session(session_id)
    if view == 'full':
        if (
            state.settings.mode_camera
            and not state.freeze_enabled
            and state.browser_frame_jpeg is not None
        ):
            return Response(content=state.browser_frame_jpeg, media_type='image/jpeg')
        frame = await read_source_frame(state, state.settings.mode_camera)
        if frame is None:
            return placeholder
        jpeg = await run.io_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg') if jpeg else placeholder

    processed = await get_processed_result(session_id)
    if processed is None:
        return placeholder

    frame = processed['images'].get(view)
    if frame is None:
        return placeholder

    jpeg = await run.io_bound(convert, frame)
    if not jpeg:
        return placeholder
    return Response(content=jpeg, media_type='image/jpeg')


@app.get('/archive/measurements.csv')
async def download_archive_csv() -> Response:
    if not ARCHIVE_CSV.exists():
        content = ','.join(ARCHIVE_CSV_FIELDS) + '\n'
    else:
        content = ARCHIVE_CSV.read_text(encoding='utf-8')
    return Response(
        content=content,
        media_type='text/csv; charset=utf-8',
        headers={'Content-Disposition': 'attachment; filename="leaf_measurements.csv"'},
    )


@app.post('/camera/frame/{session_id}')
async def receive_browser_frame(session_id: str, request: Request) -> Response:
    state = get_session(session_id)

    content = await request.body()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
    if frame is None:
        return Response(status_code=400)

    if not state.freeze_enabled:
        state.browser_frame = frame
        state.browser_frame_jpeg = bytes(content)
    return Response(status_code=204)


def ensure_manual_damage_mask(state: SessionState, width: int, height: int) -> np.ndarray:
    if state.manual_damage_mask is None or state.manual_damage_mask.shape != (height, width):
        state.manual_damage_mask = np.zeros((height, width), dtype=np.uint8)
    return state.manual_damage_mask


def ensure_manual_correct_mask(state: SessionState, width: int, height: int) -> np.ndarray:
    if state.manual_correct_mask is None or state.manual_correct_mask.shape != (height, width):
        state.manual_correct_mask = np.zeros((height, width), dtype=np.uint8)
    return state.manual_correct_mask


def ensure_manual_exclude_mask(state: SessionState, width: int, height: int) -> np.ndarray:
    if state.manual_exclude_mask is None or state.manual_exclude_mask.shape != (height, width):
        state.manual_exclude_mask = np.zeros((height, width), dtype=np.uint8)
    return state.manual_exclude_mask


@app.post('/manual-damage/{session_id}')
async def receive_manual_damage(session_id: str, request: Request) -> Response:
    state = get_session(session_id)
    try:
        payload = await request.json()
    except Exception:
        return Response(status_code=400)

    tool = payload.get('tool')
    points = payload.get('points') or []
    if tool not in {'brush', 'damage', 'correct', 'exclude', 'eraser'} or len(points) == 0:
        return Response(status_code=400)

    dig_width = max(100, min(2500, int(state.settings.dig_width or 700)))
    canvas_width = max(1, float(payload.get('canvas_width') or dig_width))
    canvas_height = max(1, float(payload.get('canvas_height') or dig_width))
    brush_size = max(1, min(250, int(payload.get('brush_size') or state.manual_brush_size)))
    scaled_brush = max(1, int(round(brush_size * dig_width / max(canvas_width, canvas_height))))

    damage_mask = ensure_manual_damage_mask(state, dig_width, dig_width)
    correct_mask = ensure_manual_correct_mask(state, dig_width, dig_width)
    exclude_mask = ensure_manual_exclude_mask(state, dig_width, dig_width)

    scaled_points = []
    for point in points:
        try:
            x = int(round(float(point['x']) * dig_width / canvas_width))
            y = int(round(float(point['y']) * dig_width / canvas_height))
        except (KeyError, TypeError, ValueError):
            continue
        x = max(0, min(dig_width - 1, x))
        y = max(0, min(dig_width - 1, y))
        scaled_points.append((x, y))

    if not scaled_points:
        return Response(status_code=400)

    def draw(mask: np.ndarray, value: int) -> None:
        if len(scaled_points) == 1:
            cv2.circle(mask, scaled_points[0], max(1, scaled_brush // 2), value, thickness=-1)
        else:
            for start, end in zip(scaled_points, scaled_points[1:]):
                cv2.line(mask, start, end, value, scaled_brush, lineType=cv2.LINE_8)

    if tool in {'brush', 'damage'}:
        draw(damage_mask, 255)
        draw(correct_mask, 0)
        draw(exclude_mask, 0)
    elif tool == 'correct':
        draw(correct_mask, 255)
        draw(damage_mask, 0)
        draw(exclude_mask, 0)
    elif tool == 'exclude':
        draw(exclude_mask, 255)
        draw(damage_mask, 0)
        draw(correct_mask, 0)
    else:
        draw(damage_mask, 0)
        draw(correct_mask, 0)
        draw(exclude_mask, 0)

    state.manual_damage_revision += 1
    state.processed_cache = {}
    return Response(status_code=204)


@app.post('/manual-damage/{session_id}/clear')
async def clear_manual_damage(session_id: str) -> Response:
    state = get_session(session_id)
    if state.manual_damage_mask is not None:
        state.manual_damage_mask.fill(0)
    if state.manual_correct_mask is not None:
        state.manual_correct_mask.fill(0)
    if state.manual_exclude_mask is not None:
        state.manual_exclude_mask.fill(0)
    state.manual_damage_revision += 1
    state.processed_cache = {}
    return Response(status_code=204)


def setup() -> None:
    pass


@ui.page('/')
def page() -> None:
    dark = ui.dark_mode()
    session_id = uuid4().hex
    state = create_session(session_id)
    app_settings = state.settings
    camera_post_url = f'/camera/frame/{session_id}'
    manual_damage_url = f'/manual-damage/{session_id}'
    manual_clear_url = f'/manual-damage/{session_id}/clear'

    async def handle_session_upload(event: Any) -> None:
        await handle_upload(state, event)

    translatable_text: list[tuple[Any, str, str]] = []
    translatable_label: list[tuple[Any, str, str]] = []

    def tr(key: str, fallback: str) -> str:
        return text(key, fallback)

    def remember_text(element: Any, key: str, fallback: str) -> Any:
        translatable_text.append((element, key, fallback))
        return element

    def remember_label(element: Any, key: str, fallback: str) -> Any:
        translatable_label.append((element, key, fallback))
        return element

    def label_t(key: str, fallback: str) -> Any:
        return remember_text(ui.label(tr(key, fallback)), key, fallback)

    def button_t(key: str, fallback: str, **kwargs: Any) -> Any:
        return remember_text(ui.button(tr(key, fallback), **kwargs), key, fallback)

    def icon_button_t(icon: str, key: str, fallback: str, **kwargs: Any) -> Any:
        button = ui.button(icon=icon, **kwargs).props('dense round flat')
        button.classes('leaf-tool-material-button')
        button.tooltip(tr(key, fallback))
        return button

    def svg_tool_button_t(svg: str, key: str, fallback: str, js: str) -> Any:
        title = tr(key, fallback).replace('&', '&amp;').replace('"', '&quot;')
        onclick = js.replace('&', '&amp;').replace('"', '&quot;')
        return ui.html(
            f'<button class="leaf-tool-icon-button" type="button" title="{title}" '
            f'onclick="{onclick}">{svg}</button>'
        )

    def switch_t(key: str, fallback: str, **kwargs: Any) -> Any:
        return remember_text(ui.switch(tr(key, fallback), **kwargs), key, fallback)

    def checkbox_t(key: str, fallback: str, **kwargs: Any) -> Any:
        return remember_text(ui.checkbox(tr(key, fallback), **kwargs), key, fallback)

    def select_t(key: str, fallback: str, *args: Any, **kwargs: Any) -> Any:
        element = ui.select(*args, label=tr(key, fallback), **kwargs)
        return remember_label(element, key, fallback)

    def number_t(key: str, fallback: str, **kwargs: Any) -> Any:
        element = ui.number(label=tr(key, fallback), **kwargs)
        return remember_label(element, key, fallback)

    def apply_translations() -> None:
        for element, key, fallback in translatable_text:
            if hasattr(element, 'set_text'):
                element.set_text(tr(key, fallback))
        for element, key, fallback in translatable_label:
            if hasattr(element, 'set_label'):
                element.set_label(tr(key, fallback))
        try:
            if state.freeze_enabled:
                freeze_button.set_text(tr('live_button', 'Live'))
        except NameError:
            pass
        fullscreen_text = json.dumps(tr('fullscreen_button', 'Vollbild'))
        fullscreen_close_text = json.dumps(tr('fullscreen_close', 'Schließen'))
        ui.run_javascript(
            f"const btn = document.getElementById('cropped-fullscreen-button-{session_id}');"
            f"const drawState = window.leafManualDamage?.['{session_id}'];"
            f"if (btn) btn.textContent = drawState?.fullscreen ? {fullscreen_close_text} : {fullscreen_text};"
            f"window.leafMeasurementCamera?.['{session_id}']?.updateTexts({camera_texts_json()});"
        )

    def handle_language_change(event: Any) -> None:
        load_language(event, reload_page=False)
        apply_translations()

    def camera_texts_json() -> str:
        return json.dumps({
            'prefix': tr('camera_status_prefix', 'Kamera'),
            'bereit': tr('camera_state_ready', 'bereit'),
            'gestoppt': tr('camera_state_stopped', 'gestoppt'),
            'startet': tr('camera_state_starting', 'startet'),
            'aktiv': tr('camera_state_active', 'aktiv'),
            'wartet': tr('camera_state_waiting', 'wartet'),
            'unterbrochen': tr('camera_state_interrupted', 'unterbrochen'),
            'haengt': tr('camera_state_hanging', 'hängt'),
            'Fehler': tr('camera_state_error', 'Fehler'),
            'nicht verfuegbar': tr('camera_state_unavailable', 'nicht verfügbar'),
            'hole Kameranamen': tr('camera_detail_getting_names', 'hole Kameranamen'),
            'Kameranamen noch gesperrt': tr('camera_detail_names_locked', 'Kameranamen noch gesperrt'),
            'Browser blockiert getUserMedia': tr('camera_detail_get_user_media_blocked', 'Browser blockiert getUserMedia'),
            'Video-Track beendet': tr('camera_detail_track_ended', 'Video-Track beendet'),
            'Video-Track liefert gerade keine Frames': tr('camera_detail_no_frames', 'Video-Track liefert gerade keine Frames'),
            'Stream nicht live': tr('camera_detail_stream_not_live', 'Stream nicht live'),
            'letzter Frame-Upload hing, versuche weiter': tr('camera_detail_upload_stuck', 'letzter Frame-Upload hing, versuche weiter'),
            'Stream beendet, starte neu': tr('camera_detail_stream_restart', 'Stream beendet, starte neu'),
            'Watchdog startet Kamera neu': tr('camera_detail_watchdog_restart', 'Watchdog startet Kamera neu'),
            'Standardkamera': tr('camera_default', 'Standardkamera'),
            'Kamera': tr('camera_fallback_name', 'Kamera'),
            'nicht verfuegbar_device_scan': tr('camera_device_scan_unavailable', 'enumerateDevices: nicht verfügbar'),
            'Device-Scan Fehler': tr('camera_device_scan_error', 'Device-Scan Fehler'),
            'Videogeraete': tr('camera_video_devices', 'Videogeräte'),
        }, ensure_ascii=False)

    ui.colors(primary='#16a34a')

    ui.add_head_html("""
        <style>
            :root {
                --leaf-accent: #16a34a;
                --leaf-accent-soft: rgba(22, 163, 74, 0.14);
                --leaf-accent-hover: rgba(22, 163, 74, 0.2);
            }
            .body--dark {
                --leaf-accent: #4ade80;
                --leaf-accent-soft: rgba(74, 222, 128, 0.16);
                --leaf-accent-hover: rgba(74, 222, 128, 0.23);
            }
            .q-btn.text-primary,
            .q-icon.text-primary {
                color: var(--leaf-accent) !important;
            }
            .q-btn.bg-primary {
                background: var(--leaf-accent) !important;
            }
            .q-toggle__inner--truthy,
            .q-slider__track,
            .q-slider__selection,
            .q-slider__thumb {
                color: var(--leaf-accent) !important;
            }
            .leaf-shell {
                width: min(100%, 1540px);
                margin: 0 auto;
                align-items: flex-start;
            }
            .leaf-preview-panel {
                flex: 1 1 1080px;
                min-width: 0;
            }
            .leaf-settings-panel {
                flex: 0 0 380px;
                max-width: 380px;
            }
            .leaf-preview-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(420px, 1fr));
                gap: 12px;
                width: 100%;
            }
            .leaf-span-full {
                grid-column: 1 / -1;
            }
            .leaf-preview-title {
                font-weight: 600;
                margin-top: 4px;
            }
            .leaf-tool-icon-button {
                width: 34px;
                height: 34px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                border: 0;
                border-radius: 999px;
                background: transparent;
                color: var(--leaf-accent);
                cursor: pointer;
            }
            .leaf-tool-icon-button:hover {
                background: var(--leaf-accent-hover);
            }
            .leaf-tool-icon-button svg {
                width: 24px;
                height: 24px;
                display: block;
            }
            .leaf-tool-material-button {
                color: var(--leaf-accent) !important;
            }
            .leaf-tool-material-button:hover {
                background: var(--leaf-accent-hover) !important;
            }
            .leaf-cropped-draw-wrap {
                position: relative;
                width: 100%;
                aspect-ratio: 1 / 1;
                overflow: hidden;
                box-sizing: border-box;
                touch-action: none;
                overscroll-behavior: contain;
                -webkit-user-select: none;
                user-select: none;
            }
            .leaf-cropped-image,
            .leaf-manual-damage-canvas {
                position: absolute;
                inset: 0;
                display: block;
                width: 100%;
                height: 100%;
                box-sizing: border-box;
                user-select: none;
                -webkit-user-select: none;
            }
            .leaf-cropped-image {
                object-fit: contain;
                -webkit-touch-callout: none;
                border: 2px solid #f59e0b;
                background: #111;
            }
            .leaf-manual-damage-canvas {
                cursor: crosshair;
                touch-action: none;
                overscroll-behavior: contain;
                border: 2px dashed rgba(245, 158, 11, 0.7);
            }
            @media (max-width: 980px) {
                .leaf-shell {
                    display: flex;
                    flex-direction: column;
                }
                .leaf-preview-panel,
                .leaf-settings-panel {
                    width: 100%;
                    max-width: none;
                    flex-basis: auto;
                }
                .leaf-preview-grid {
                    grid-template-columns: minmax(0, 1fr);
                }
            }
            body.leaf-draw-fullscreen-active {
                overflow: hidden;
            }
            .leaf-cropped-fullscreen-button {
                position: absolute;
                top: 8px;
                right: 8px;
                z-index: 4;
                padding: 6px 10px;
                border: 1px solid rgba(255, 255, 255, 0.7);
                border-radius: 4px;
                background: rgba(17, 17, 17, 0.82);
                color: white;
                font-size: 13px;
                line-height: 1.2;
            }
            .leaf-draw-fullscreen {
                position: fixed !important;
                inset: 0 !important;
                z-index: 5000;
                width: 100vw !important;
                height: 100dvh !important;
                aspect-ratio: auto !important;
                background: #111;
                padding: 12px;
                box-sizing: border-box;
                touch-action: none;
                overscroll-behavior: contain;
            }
            .leaf-draw-fullscreen img,
            .leaf-draw-fullscreen canvas {
                inset: auto !important;
                left: 50% !important;
                top: 50% !important;
                width: min(calc(100vw - 24px), calc(100dvh - 24px)) !important;
                height: min(calc(100vw - 24px), calc(100dvh - 24px)) !important;
                transform: translate(-50%, -50%);
                box-sizing: border-box !important;
                border-width: 0 !important;
            }
            .leaf-draw-fullscreen .leaf-cropped-fullscreen-button {
                top: 18px;
                right: 18px;
            }
        </style>
    """)
    ui.add_body_html("""
        <script>
        (() => {
            const cookieMatch = document.cookie.match(/(?:^|; )leafMeasurementDarkMode=(true|false)/);
            const stored = localStorage.getItem('leafMeasurementDarkMode') ?? cookieMatch?.[1] ?? null;
            if (stored !== null) {
                const enabled = stored === 'true';
                document.documentElement.classList.toggle('dark', enabled);
                document.body.classList.toggle('body--dark', enabled);
                const apply = () => {
                    if (window.Quasar?.Dark) {
                        window.Quasar.Dark.set(enabled);
                    }
                };
                apply();
                window.setTimeout(apply, 250);
            }
            window.leafMeasurementSetDarkMode = enabled => {
                localStorage.setItem('leafMeasurementDarkMode', enabled ? 'true' : 'false');
                document.cookie = `leafMeasurementDarkMode=${enabled ? 'true' : 'false'}; max-age=31536000; path=/; SameSite=Lax`;
                if (window.Quasar?.Dark) {
                    window.Quasar.Dark.set(enabled);
                }
            };
        })();
        </script>
    """)

    camera_script = """
        <video id="browser-camera-video-__SESSION_ID__" autoplay playsinline muted style="position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;left:-9999px;top:-9999px"></video>
        <canvas id="browser-camera-canvas-__SESSION_ID__" style="display:none"></canvas>
        <script>
        (() => {
            const video = document.getElementById('browser-camera-video-__SESSION_ID__');
            const canvas = document.getElementById('browser-camera-canvas-__SESSION_ID__');
            const ctx = canvas.getContext('2d');
            const sessionId = '__SESSION_ID__';
            let cameraTexts = __CAMERA_TEXTS__;
            let stream = null;
            let started = false;
            let startingPromise = null;
            let sending = false;
            let sendingStartedAt = 0;
            let lastFrameSentAt = 0;
            let lastWatchdogRestartAt = 0;
            let consecutiveFrameErrors = 0;
            let deviceRefreshRevision = 0;
            let initialDeviceRefreshStarted = false;
            let lastCameraStatus = 'bereit';
            let lastCameraError = '';
            const frameIntervalMs = 350;
            const sendTimeoutMs = 2500;

            window.leafMeasurementCamera = window.leafMeasurementCamera || {};
            const api = window.leafMeasurementCamera[sessionId] = {
                start: startCamera,
                stop: stopCamera,
                refreshDevices,
                freezeStream: freezeCameraStream,
                resumeAfterFreeze: resumeCameraAfterFreeze,
                updateTexts: texts => {
                    cameraTexts = { ...cameraTexts, ...texts };
                    renderCameraStatus();
                },
            };

            function cameraText(value) {
                return cameraTexts[value] || value;
            }

            function renderCameraStatus() {
                const label = document.getElementById(`camera-status-${sessionId}`);
                if (!label) {
                    return;
                }
                const prefix = cameraTexts.prefix || 'Kamera';
                const status = cameraText(lastCameraStatus);
                const error = cameraText(lastCameraError);
                label.textContent = error ? `${prefix}: ${status} (${error})` : `${prefix}: ${status}`;
            }

            function setCameraStatus(status, error = '') {
                lastCameraStatus = status;
                lastCameraError = error;
                renderCameraStatus();
            }

            function setCameraDebug(lines) {
                const debug = document.getElementById(`camera-debug-${sessionId}`);
                if (debug) {
                    debug.textContent = Array.isArray(lines) ? lines.join('\\n') : String(lines || '');
                }
            }

            function describeError(error) {
                return [error?.name, error?.message].filter(Boolean).join(': ') || 'unbekannt';
            }

            function stopCurrentStream() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
                video.pause();
                video.srcObject = null;
                sending = false;
            }

            function freezeCameraStream() {
                started = false;
                stopCurrentStream();
                setCameraStatus('gestoppt');
            }

            async function resumeCameraAfterFreeze() {
                if (started || startingPromise) {
                    return startingPromise;
                }
                return startCamera();
            }

            function stopCamera() {
                started = false;
                stopCurrentStream();
                setCameraStatus('gestoppt');
            }

            async function unlockDeviceLabelsIfNeeded(videoDevices) {
                if (started || !navigator.mediaDevices?.getUserMedia || !videoDevices.length) {
                    return false;
                }
                if (videoDevices.some(device => device.label)) {
                    return false;
                }
                try {
                    setCameraStatus('bereit', 'hole Kameranamen');
                    const probeStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                    probeStream.getTracks().forEach(track => track.stop());
                    return true;
                } catch (error) {
                    setCameraDebug([
                        `URL: ${window.location.href}`,
                        `Secure Context: ${window.isSecureContext}`,
                        `getUserMedia: ${Boolean(navigator.mediaDevices?.getUserMedia)}`,
                        `Kameranamen noch gesperrt: ${describeError(error)}`,
                    ]);
                    return false;
                }
            }

            async function refreshDevices(unlockLabels = true) {
                const revision = ++deviceRefreshRevision;
                const select = document.getElementById(`camera-device-${sessionId}`);
                const lines = [
                    `URL: ${window.location.href}`,
                    `Secure Context: ${window.isSecureContext}`,
                    `getUserMedia: ${Boolean(navigator.mediaDevices?.getUserMedia)}`,
                ];
                if (!select) {
                    setCameraDebug(lines);
                    return [];
                }

                const previousValue = select.value;
                if (!navigator.mediaDevices?.enumerateDevices) {
                    lines.push('enumerateDevices: nicht verfuegbar');
                    const defaultOption = document.createElement('option');
                    defaultOption.value = '';
                    defaultOption.textContent = cameraText('Standardkamera');
                    select.replaceChildren(defaultOption);
                    setCameraDebug(lines);
                    return [];
                }

                try {
                    let devices = await navigator.mediaDevices.enumerateDevices();
                    let videoDevices = devices.filter(device => device.kind === 'videoinput');
                    if (unlockLabels && await unlockDeviceLabelsIfNeeded(videoDevices)) {
                        devices = await navigator.mediaDevices.enumerateDevices();
                        videoDevices = devices.filter(device => device.kind === 'videoinput');
                    }
                    if (revision !== deviceRefreshRevision) {
                        return [];
                    }

                    const seen = new Set();
                    const uniqueVideoDevices = [];
                    videoDevices.forEach((device, index) => {
                        const key = device.deviceId || device.label || `camera-${index}`;
                        if (seen.has(key)) {
                            return;
                        }
                        seen.add(key);
                        uniqueVideoDevices.push(device);
                    });

                    const defaultOption = document.createElement('option');
                    defaultOption.value = '';
                    defaultOption.textContent = cameraText('Standardkamera');
                    const options = [defaultOption];
                    lines.push(`${cameraText('Videogeraete')}: ${uniqueVideoDevices.length}`);
                    uniqueVideoDevices.forEach((device, index) => {
                        const option = document.createElement('option');
                        option.value = device.deviceId;
                        option.textContent = device.label || `${cameraText('Kamera')} ${index + 1}`;
                        options.push(option);
                        lines.push(`- ${option.textContent}`);
                    });
                    select.replaceChildren(...options);
                    if ([...select.options].some(option => option.value === previousValue)) {
                        select.value = previousValue;
                    }
                    setCameraDebug(lines);
                    return uniqueVideoDevices;
                } catch (error) {
                    if (revision !== deviceRefreshRevision) {
                        return [];
                    }
                    lines.push(`Device-Scan Fehler: ${describeError(error)}`);
                    setCameraDebug(lines);
                    return [];
                }
            }

            function refreshDevicesWhenUiIsReady(attempt = 0) {
                if (initialDeviceRefreshStarted) {
                    return;
                }
                const select = document.getElementById(`camera-device-${sessionId}`);
                if (select) {
                    initialDeviceRefreshStarted = true;
                    refreshDevices(true);
                    return;
                }
                if (attempt < 50) {
                    window.setTimeout(() => refreshDevicesWhenUiIsReady(attempt + 1), 100);
                }
            }

            function waitForMetadata(timeoutMs = 2500) {
                if (video.readyState >= 1 && video.videoWidth > 0) {
                    return Promise.resolve();
                }
                return new Promise(resolve => {
                    const done = () => {
                        video.removeEventListener('loadedmetadata', done);
                        window.clearTimeout(timer);
                        resolve();
                    };
                    const timer = window.setTimeout(done, timeoutMs);
                    video.addEventListener('loadedmetadata', done, { once: true });
                });
            }

            async function requestCameraStream() {
                const select = document.getElementById(`camera-device-${sessionId}`);
                const selectedDeviceId = select?.value || '';
                const attempts = [];

                if (selectedDeviceId) {
                    attempts.push({
                        label: 'Ausgewaehltes Geraet',
                        video: { deviceId: { exact: selectedDeviceId } },
                    });
                }

                attempts.push(
                    { label: 'Standardkamera', video: true },
                    { label: '640x480', video: { width: { ideal: 640 }, height: { ideal: 480 } } },
                    { label: 'Rueckkamera', video: { facingMode: { ideal: 'environment' } } },
                );

                const debugLines = [`Startversuche: ${attempts.length}`];
                let lastError = null;
                for (const attempt of attempts) {
                    try {
                        debugLines.push(`Versuche: ${attempt.label}`);
                        const nextStream = await navigator.mediaDevices.getUserMedia({ video: attempt.video, audio: false });
                        debugLines.push(`OK: ${attempt.label}`);
                        setCameraDebug(debugLines);
                        return nextStream;
                    } catch (error) {
                        lastError = error;
                        debugLines.push(`Fehler ${attempt.label}: ${describeError(error)}`);
                    }
                }
                setCameraDebug(debugLines);
                throw lastError || new Error('Keine Kamera gefunden');
            }

            async function startCamera() {
                if (startingPromise) {
                    return startingPromise;
                }
                if (started && video.readyState >= 2) {
                    return;
                }
                if (!navigator.mediaDevices?.getUserMedia) {
                    setCameraStatus('nicht verfuegbar', 'Browser blockiert getUserMedia');
                    return;
                }
                startingPromise = (async () => {
                    try {
                        setCameraStatus('startet');
                        stopCurrentStream();
                        await refreshDevices(false);
                        stream = await requestCameraStream();
                        stream.getVideoTracks().forEach(track => {
                            track.addEventListener('ended', () => {
                                started = false;
                                setCameraStatus('unterbrochen', 'Video-Track beendet');
                            });
                            track.addEventListener('mute', () => {
                                setCameraStatus('wartet', 'Video-Track liefert gerade keine Frames');
                            });
                            track.addEventListener('unmute', () => {
                                if (started) {
                                    setCameraStatus('aktiv');
                                }
                            });
                        });
                        video.srcObject = stream;
                        video.muted = true;
                        video.playsInline = true;
                        await waitForMetadata();
                        try {
                            await video.play();
                        } catch (playError) {
                            if (video.readyState < 2 || video.videoWidth === 0) {
                                throw playError;
                            }
                        }
                        started = true;
                        lastFrameSentAt = Date.now();
                        consecutiveFrameErrors = 0;
                        setCameraStatus('aktiv');
                    } catch (error) {
                        started = false;
                        stopCurrentStream();
                        setCameraStatus('Fehler', describeError(error));
                        console.error('Browser camera could not be started:', error);
                    } finally {
                        startingPromise = null;
                    }
                })();
                return startingPromise;
            }

            function streamIsLive() {
                return stream?.getVideoTracks?.().some(track => track.readyState === 'live') || false;
            }

            function canvasToJpegBlob(timeoutMs = 1200) {
                return new Promise((resolve, reject) => {
                    const timer = window.setTimeout(() => reject(new Error('canvas.toBlob timeout')), timeoutMs);
                    try {
                        canvas.toBlob(blob => {
                            window.clearTimeout(timer);
                            if (blob) {
                                resolve(blob);
                            } else {
                                reject(new Error('canvas.toBlob returned null'));
                            }
                        }, 'image/jpeg', 0.55);
                    } catch (error) {
                        window.clearTimeout(timer);
                        reject(error);
                    }
                });
            }

            async function postFrame(blob) {
                const controller = new AbortController();
                const timer = window.setTimeout(() => controller.abort(), sendTimeoutMs);
                try {
                    await fetch('__CAMERA_POST_URL__', {
                        method: 'POST',
                        headers: { 'Content-Type': 'image/jpeg' },
                        body: blob,
                        signal: controller.signal,
                        keepalive: false,
                    });
                } finally {
                    window.clearTimeout(timer);
                }
            }

            async function sendFrame() {
                if (!started) {
                    return;
                }
                if (!streamIsLive()) {
                    started = false;
                    setCameraStatus('unterbrochen', 'Stream nicht live');
                    return;
                }
                if (sending && Date.now() - sendingStartedAt < sendTimeoutMs) {
                    return;
                }
                if (sending && Date.now() - sendingStartedAt >= sendTimeoutMs) {
                    sending = false;
                    consecutiveFrameErrors += 1;
                    setCameraStatus('aktiv', 'letzter Frame-Upload hing, versuche weiter');
                }
                if (video.readyState < 2 || video.videoWidth === 0) {
                    return;
                }

                sending = true;
                sendingStartedAt = Date.now();
                try {
                    const maxWidth = 960;
                    const scale = Math.min(1, maxWidth / video.videoWidth);
                    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
                    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const blob = await canvasToJpegBlob();
                    await postFrame(blob);
                    lastFrameSentAt = Date.now();
                    consecutiveFrameErrors = 0;
                    if (started && consecutiveFrameErrors > 0) {
                        setCameraStatus('aktiv');
                    }
                } catch (error) {
                    consecutiveFrameErrors += 1;
                    console.error('Browser camera frame could not be sent:', error);
                    if (consecutiveFrameErrors >= 3) {
                        setCameraStatus('aktiv', `Frame-Upload Problem: ${describeError(error)}`);
                    }
                } finally {
                    sending = false;
                }
            }

            async function cameraWatchdog() {
                if (!started || startingPromise) {
                    return;
                }
                const now = Date.now();
                if (!streamIsLive()) {
                    started = false;
                    setCameraStatus('unterbrochen', 'Stream beendet, starte neu');
                    await startCamera();
                    return;
                }
                if (lastFrameSentAt > 0 && now - lastFrameSentAt > 5000 && now - lastWatchdogRestartAt > 8000) {
                    lastWatchdogRestartAt = now;
                    started = false;
                    setCameraStatus('haengt', 'Watchdog startet Kamera neu');
                    await startCamera();
                }
            }

            setCameraStatus('bereit');
            refreshDevicesWhenUiIsReady();
            if (document.readyState === 'loading') {
                window.addEventListener('load', () => refreshDevicesWhenUiIsReady(), { once: true });
            }
            window.addEventListener('beforeunload', stopCamera);
            document.addEventListener('visibilitychange', () => {
                if (document.hidden) {
                    setCameraStatus(started ? 'aktiv' : 'bereit');
                } else {
                    refreshDevices(true);
                    if (started && video.paused) {
                        video.play().catch(error => console.warn('Could not resume camera video:', error));
                    }
                }
            });
            window.leafMeasurementCameraTimer = window.setInterval(sendFrame, frameIntervalMs);
            window.leafMeasurementCameraWatchdogTimer = window.setInterval(cameraWatchdog, 1000);
        })();
        </script>
    """
    ui.add_body_html(
        camera_script
        .replace('__SESSION_ID__', session_id)
        .replace('__CAMERA_POST_URL__', camera_post_url)
        .replace('__CAMERA_TEXTS__', camera_texts_json())
    )
    drawing_script = """
        <script>
        (() => {
            const sessionId = '__SESSION_ID__';
            const imageUrl = '__CROPPED_URL__';
            const damageUrl = '__MANUAL_DAMAGE_URL__';
            const clearUrl = '__MANUAL_CLEAR_URL__';
            window.leafManualDamage = window.leafManualDamage || {};
            const state = window.leafManualDamage[sessionId] = {
                tool: 'damage',
                brushSize: 18,
                enabled: true,
                fullscreen: false,
                toggleFullscreen: () => {},
                clear: async () => {
                    await fetch(clearUrl, { method: 'POST' });
                    const canvas = document.getElementById(`manual-damage-canvas-${sessionId}`);
                    const ctx = canvas?.getContext('2d');
                    if (canvas && ctx) {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                    }
                },
            };

            function init() {
                const wrap = document.getElementById(`cropped-draw-wrap-${sessionId}`);
                const img = document.getElementById(`cropped-image-${sessionId}`);
                const canvas = document.getElementById(`manual-damage-canvas-${sessionId}`);
                const fullscreenButton = document.getElementById(`cropped-fullscreen-button-${sessionId}`);
                if (!wrap || !img || !canvas) {
                    window.setTimeout(init, 100);
                    return;
                }
                const ctx = canvas.getContext('2d');
                let drawing = false;
                let points = [];
                let activePointerId = null;
                const supportsPointerEvents = window.PointerEvent !== undefined;

                function syncCanvasSize() {
                    const rect = canvas.getBoundingClientRect();
                    if (rect.width < 2 || rect.height < 2) {
                        return;
                    }
                    const old = document.createElement('canvas');
                    old.width = canvas.width;
                    old.height = canvas.height;
                    old.getContext('2d').drawImage(canvas, 0, 0);
                    canvas.width = Math.max(1, Math.round(rect.width));
                    canvas.height = Math.max(1, Math.round(rect.height));
                    ctx.drawImage(old, 0, 0, canvas.width, canvas.height);
                }

                function toggleFullscreen(force) {
                    state.fullscreen = typeof force === 'boolean' ? force : !state.fullscreen;
                    wrap.classList.toggle('leaf-draw-fullscreen', state.fullscreen);
                    document.body.classList.toggle('leaf-draw-fullscreen-active', state.fullscreen);
                    if (fullscreenButton) {
                        fullscreenButton.textContent = state.fullscreen ? '__FULLSCREEN_CLOSE__' : '__FULLSCREEN_BUTTON__';
                    }
                    window.setTimeout(syncCanvasSize, 60);
                }

                function canvasPoint(event) {
                    const source = event.touches?.[0] || event.changedTouches?.[0] || event;
                    const rect = canvas.getBoundingClientRect();
                    const x = source.clientX - rect.left;
                    const y = source.clientY - rect.top;
                    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
                        return null;
                    }
                    return {
                        x: Math.max(0, Math.min(canvas.width, x)),
                        y: Math.max(0, Math.min(canvas.height, y)),
                    };
                }

                function drawLocalLine(a, b) {
                    ctx.strokeStyle = state.tool === 'eraser'
                        ? 'rgba(255,255,255,0.85)'
                        : state.tool === 'correct'
                            ? 'rgba(34,197,94,0.75)'
                            : state.tool === 'exclude'
                                ? 'rgba(168,85,247,0.75)'
                                : 'rgba(255,128,0,0.75)';
                    ctx.lineWidth = state.brushSize;
                    ctx.lineCap = 'round';
                    ctx.lineJoin = 'round';
                    ctx.globalCompositeOperation = state.tool === 'eraser' ? 'destination-out' : 'source-over';
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                    ctx.globalCompositeOperation = 'source-over';
                }

                async function sendStroke() {
                    if (!points.length) {
                        return;
                    }
                    await fetch(damageUrl, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            tool: state.tool,
                            points,
                            brush_size: state.brushSize,
                            canvas_width: canvas.width,
                            canvas_height: canvas.height,
                        }),
                    });
                }

                function startStroke(event) {
                    syncCanvasSize();
                    const point = canvasPoint(event);
                    if (!point) {
                        return;
                    }
                    if (event.pointerId !== undefined) {
                        activePointerId = event.pointerId;
                        canvas.setPointerCapture?.(event.pointerId);
                    }
                    drawing = true;
                    points = [point];
                    event.preventDefault();
                }

                function moveStroke(event) {
                    if (!drawing) {
                        return;
                    }
                    if (event.pointerId !== undefined && activePointerId !== null && event.pointerId !== activePointerId) {
                        return;
                    }
                    const point = canvasPoint(event);
                    if (!point) {
                        finishStroke(event);
                        return;
                    }
                    drawLocalLine(points[points.length - 1], point);
                    points.push(point);
                    event.preventDefault();
                }

                async function finishStroke(event) {
                    if (!drawing) {
                        return;
                    }
                    if (event.pointerId !== undefined && activePointerId !== null && event.pointerId !== activePointerId) {
                        return;
                    }
                    drawing = false;
                    activePointerId = null;
                    if (points.length === 1) {
                        drawLocalLine(points[0], points[0]);
                    }
                    await sendStroke();
                    points = [];
                    event.preventDefault();
                }

                if (supportsPointerEvents) {
                    canvas.addEventListener('pointerdown', startStroke, { passive: false });
                    canvas.addEventListener('pointermove', moveStroke, { passive: false });
                    canvas.addEventListener('pointerup', finishStroke, { passive: false });
                    canvas.addEventListener('pointercancel', finishStroke, { passive: false });
                } else {
                    canvas.addEventListener('touchstart', startStroke, { passive: false });
                    canvas.addEventListener('touchmove', moveStroke, { passive: false });
                    canvas.addEventListener('touchend', finishStroke, { passive: false });
                    canvas.addEventListener('touchcancel', finishStroke, { passive: false });
                }

                state.toggleFullscreen = toggleFullscreen;
                fullscreenButton?.addEventListener('click', event => {
                    event.preventDefault();
                    event.stopPropagation();
                    toggleFullscreen();
                });
                img.addEventListener('load', syncCanvasSize);
                window.addEventListener('resize', syncCanvasSize);
                window.addEventListener('keydown', event => {
                    if (event.key === 'Escape' && state.fullscreen) {
                        toggleFullscreen(false);
                    }
                });
                window.setInterval(() => {
                    img.src = `${imageUrl}?t=${Date.now()}`;
                    syncCanvasSize();
                }, 700);
            }

            init();
        })();
        </script>
    """
    ui.add_body_html(
        drawing_script
        .replace('__SESSION_ID__', session_id)
        .replace('__CROPPED_URL__', f'/video/{session_id}/cropped')
        .replace('__MANUAL_DAMAGE_URL__', manual_damage_url)
        .replace('__MANUAL_CLEAR_URL__', manual_clear_url)
        .replace('__FULLSCREEN_BUTTON__', tr('fullscreen_button', 'Vollbild'))
        .replace('__FULLSCREEN_CLOSE__', tr('fullscreen_close', 'Schließen'))
    )

    with ui.row().classes('leaf-shell gap-4'):
        with ui.column().classes('leaf-preview-panel items-stretch'):
            with ui.card().props('flat bordered').classes('w-full items-stretch'):
                with ui.element('div').classes('leaf-preview-grid'):
                    with ui.row().classes('leaf-span-full items-center justify-between'):
                        label_t('preview_fullframe', 'Fullframe')
                        with ui.row().classes('items-center gap-2'):
                            label_t('camera_status_ready', 'Kamera: bereit').props(f'id=camera-status-{session_id}').classes('text-xs text-gray-500')
                            button_t(
                                'camera_start',
                                'Kamera starten',
                                on_click=lambda: ui.run_javascript(
                                    f"window.leafMeasurementCamera?.['{session_id}']?.start()"
                                ),
                            ).props('dense')
                            button_t(
                                'camera_stop',
                                'Stop',
                                on_click=lambda: ui.run_javascript(
                                    f"window.leafMeasurementCamera?.['{session_id}']?.stop()"
                                ),
                            ).props('dense')
                    with ui.row().classes('leaf-span-full items-center gap-2'):
                        ui.html(
                            f'<select id="camera-device-{session_id}" '
                            'style="min-width:260px;max-width:100%;padding:4px 8px;border:1px solid #999;border-radius:4px;">'
                            '<option value="">Standardkamera</option></select>'
                        )
                        button_t(
                            'camera_search',
                            'Kameras suchen',
                            on_click=lambda: ui.run_javascript(
                                f"window.leafMeasurementCamera?.['{session_id}']?.refreshDevices(true)"
                            ),
                        ).props('dense')
                    full_image = ui.interactive_image(f'/video/{session_id}/full').classes('border-none w-full leaf-span-full')

                    with ui.row().classes('leaf-span-full items-center gap-2'):
                        label_t('manual_draw_title', 'Manuell auf Cropped zeichnen').classes('font-bold')
                        manual_switch = switch_t(
                            'manual_include',
                            'Einrechnen',
                            value=state.manual_damage_enabled,
                            on_change=lambda event: set_manual_damage_enabled(state, session_id, event.value),
                        )
                        svg_tool_button_t(
                            '<svg viewBox="0 0 24 24" aria-hidden="true">'
                            '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                            '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                            '<circle cx="14.5" cy="10.5" r="2" fill="var(--q-page, #fff)" stroke="currentColor" stroke-width="1.8"/>'
                            '</svg>',
                            'tool_damage',
                            'Schaden',
                            f"window.leafManualDamage?.['{session_id}'] && "
                            f"(window.leafManualDamage['{session_id}'].tool = 'damage')",
                        )
                        svg_tool_button_t(
                            '<svg viewBox="0 0 24 24" aria-hidden="true">'
                            '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                            '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                            '</svg>',
                            'tool_correct',
                            'Korrekt',
                            f"window.leafManualDamage?.['{session_id}'] && "
                            f"(window.leafManualDamage['{session_id}'].tool = 'correct')",
                        )
                        icon_button_t(
                            'block',
                            'tool_exclude',
                            'Entfernen',
                            on_click=lambda: ui.run_javascript(
                                f"window.leafManualDamage?.['{session_id}'] && "
                                f"(window.leafManualDamage['{session_id}'].tool = 'exclude')"
                            ),
                        )
                        svg_tool_button_t(
                            '<svg viewBox="0 0 24 24" aria-hidden="true">'
                            '<path d="M21 21H8a2 2 0 0 1-1.42-.587l-3.994-3.999a2 2 0 0 1 0-2.828l10-10a2 2 0 0 1 2.829 0l5.999 6a2 2 0 0 1 0 2.828L12.834 21" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                            '<path d="m5.082 11.09 8.828 8.828" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
                            '</svg>',
                            'tool_eraser',
                            'Radierer',
                            f"window.leafManualDamage?.['{session_id}'] && "
                            f"(window.leafManualDamage['{session_id}'].tool = 'eraser')",
                        )
                        icon_button_t(
                            'delete_sweep',
                            'tool_clear',
                            'Alles löschen',
                            on_click=lambda: ui.run_javascript(
                                f"window.leafManualDamage?.['{session_id}']?.clear()"
                            ),
                        )
                        brush_size = number_t(
                            'brush_size',
                            'Größe',
                            value=state.manual_brush_size,
                            min=2,
                            max=120,
                            step=2,
                            on_change=lambda event: set_manual_brush_size(state, session_id, event.value),
                        ).classes('w-24')
                        manual_switch.tooltip(tr('manual_include_tooltip', 'Schaltet die manuelle Maske in Berechnung und Ergebnisanzeige ein oder aus'))
                        brush_size.tooltip(tr('brush_size_tooltip', 'Breite von Pinsel und Radierer'))
                        switch_t(
                            'auto_edge_damage',
                            'Convex-Randschäden',
                            value=state.auto_edge_damage_enabled,
                            on_change=lambda event: set_auto_edge_damage_enabled(state, event.value),
                        ).tooltip(tr('auto_edge_damage_tooltip', 'Schaltet nur den Bereich zwischen Blattkontur und Convex Hull ein oder aus'))

                    with ui.row().classes('leaf-span-full items-center gap-2'):
                        switch_t(
                            'manual_limit_to_leaf',
                            'Zeichnen auf Blattmaske begrenzen',
                            value=state.manual_limit_to_leaf,
                            on_change=lambda event: set_manual_limit_to_leaf(state, event.value),
                        ).tooltip(tr('manual_limit_to_leaf_tooltip', 'Zählt manuelle Korrekturen und Schäden nur innerhalb der erkannten Blattfläche'))
                        number_t(
                            'manual_shrink_mask',
                            'Maske schrumpfen',
                            value=state.manual_leaf_shrink_px,
                            min=0,
                            max=250,
                            step=2,
                            on_change=lambda event: set_manual_leaf_shrink_px(state, event.value),
                        ).classes('w-36').tooltip(tr('manual_shrink_mask_tooltip', 'Schrumpft die erkannte Blattmaske vor der Begrenzung in Pixeln'))
                        switch_t(
                            'show_auto_damage_on_cropped',
                            'Auto-Schäden auf Cropped',
                            value=state.show_auto_damage_on_cropped,
                            on_change=lambda event: set_show_auto_damage_on_cropped(state, event.value),
                        ).tooltip(tr('show_auto_damage_on_cropped_tooltip', 'Zeigt automatisch erkannte Schäden rot direkt auf dem Cropped-Bild'))

                    label_t('preview_cropped_draw', 'Cropped - hier malen').classes('leaf-preview-title')
                    label_t('preview_result', 'Result').classes('leaf-preview-title')
                    ui.html(f'''
                        <div id="cropped-draw-wrap-{session_id}" class="leaf-cropped-draw-wrap">
                            <img id="cropped-image-{session_id}" class="leaf-cropped-image" src="/video/{session_id}/cropped" draggable="false">
                            <canvas id="manual-damage-canvas-{session_id}" class="leaf-manual-damage-canvas"></canvas>
                            <button id="cropped-fullscreen-button-{session_id}" class="leaf-cropped-fullscreen-button" type="button">{tr('fullscreen_button', 'Vollbild')}</button>
                        </div>
                    ''').classes('border-none w-full')
                    result_image = ui.interactive_image(f'/video/{session_id}/result').classes('border-none w-full')

                    with ui.row().classes('items-center gap-2'):
                        freeze_button = button_t('freeze_button', 'Freeze').props('dense')

                    async def handle_freeze_click() -> None:
                        await toggle_freeze(session_id, state, freeze_button, full_image)

                    freeze_button.on('click', handle_freeze_click)

                    label_t('preview_damage_mask', 'Damage Mask').classes('leaf-span-full leaf-preview-title')
                    masked_image = ui.interactive_image(f'/video/{session_id}/mask').classes('border-none w-full leaf-span-full')

                ui.timer(interval=0.25, callback=full_image.force_reload)
                for image in (result_image, masked_image):
                    ui.timer(interval=0.8, callback=image.force_reload)

        with ui.column().classes('leaf-settings-panel items-stretch'):
            with ui.card().props('flat bordered').classes('items-stretch'):
                label_t('measurements_title', 'Messwerte')
                area_label = ui.label()
                convex_label = ui.label()
                damage_label = ui.label()
                damage_percent_label = ui.label()
                status_label = ui.label()
                archive_status_label = ui.label('').classes('text-xs text-gray-500')

                async def handle_archive_click() -> None:
                    await archive_current_measurement(session_id, state, archive_status_label)

                ui.timer(
                    interval=0.5,
                    callback=lambda: update_measurement_labels(
                        state,
                        area_label,
                        convex_label,
                        damage_label,
                        damage_percent_label,
                        status_label,
                    ),
                )
                with ui.row().classes('items-center gap-2'):
                    button_t(
                        'archive_button',
                        'Archivieren',
                        on_click=handle_archive_click,
                    ).props('dense')
                    button_t(
                        'download_csv_button',
                        'CSV herunterladen',
                        on_click=lambda: ui.run_javascript(
                            "window.open('/archive/measurements.csv?t=' + Date.now(), '_blank')"
                        ),
                    ).props('dense')

            with ui.card().props('flat bordered'):
                label_t('label_settings', 'Einstellungen')
                with ui.row().classes('items-center gap-2'):
                    select_t('select_language', 'Sprache', langlist, on_change=handle_language_change, value=sellang).classes('min-w-36')
                    dark_switch = switch_t(
                        'dark_mode_switch',
                        'Dunkelmodus',
                        on_change=lambda event: ui.run_javascript(
                            f"window.leafMeasurementSetDarkMode?.({str(bool(event.value)).lower()})"
                        ),
                    )
                    dark_switch.bind_value(dark)

                with ui.card().props('flat bordered'):
                    with remember_text(ui.expansion(tr('label_basic_settings', 'Grundeinstellungen')), 'label_basic_settings', 'Grundeinstellungen').classes('w-80'):
                        mode = checkbox_t(
                            'mode_checkbox',
                            'Modus Kamera/Bild',
                            value=app_settings.mode_camera,
                        )
                        mode.tooltip(tr('mode_checkbox_tooltip', 'Modus zwischen Kamera und Bild wechseln'))
                        mode.bind_value(app_settings, 'mode_camera')

                        upload = remember_label(ui.upload(
                            label=tr('image_upload', 'Bild'),
                            max_files=1,
                            on_upload=handle_session_upload,
                            on_rejected=lambda _: ui.notify(tr('warn_file_upload_fail', 'Fehler beim Hochladen der Datei')),
                        ), 'image_upload', 'Bild')
                        upload.classes('w-70').props('flat bordered').tooltip(tr('image_upload_tooltip', 'Bild von Festplatte auswählen'))

                        physwidth = number_t(
                            'physwidth_input',
                            'Physische Breite',
                            value=app_settings.phys_width,
                            min=0.001,
                            step=0.1,
                        )
                        physwidth.tooltip(tr('physwidth_input_tooltip', 'Physische Breite zwischen den Markern'))
                        physwidth.bind_value(app_settings, 'phys_width')

                        physheight = number_t(
                            'physheight_input',
                            'Physische Höhe',
                            value=app_settings.phys_height,
                            min=0.001,
                            step=0.1,
                        )
                        physheight.tooltip(tr('physheight_input_tooltip', 'Physische Höhe zwischen den Markern'))
                        physheight.bind_value(app_settings, 'phys_height')

                        digwidth = number_t(
                            'digwidth_input',
                            'Digitale Auflösung',
                            value=app_settings.dig_width,
                            min=100,
                            max=2500,
                            step=50,
                        )
                        digwidth.tooltip(tr('digwidth_input_tooltip', 'Digitale Auflösung des Zuschnitts'))
                        digwidth.bind_value(app_settings, 'dig_width')

                with ui.card().props('flat bordered'):
                    with remember_text(ui.expansion(tr('label_filter_settings', 'Filtereinstellungen')), 'label_filter_settings', 'Filtereinstellungen').classes('w-80'):
                        kernelsize = number_t(
                            'kernelsize_input',
                            'Kernelgröße',
                            value=app_settings.kernel_size,
                            min=1,
                            max=100,
                            step=1,
                        )
                        kernelsize.tooltip(tr('kernelsize_input_tooltip', 'Größe der Matrix für die Rauschfilterung'))
                        kernelsize.bind_value(app_settings, 'kernel_size')

                        label_t('hsv_live_limits', 'HSV Live-Grenzen').classes('font-medium mt-3')
                        label_t('hsv_hue', 'Hue / Farbton').classes('text-sm text-gray-500')
                        ui.range(
                            min=0,
                            max=179,
                            step=1,
                            value={'min': app_settings.lower_hsv[0], 'max': app_settings.upper_hsv[0]},
                            on_change=lambda event: update_hsv_range(state, 0, event.value),
                        ).props('label-always').classes('w-full')

                        label_t('hsv_saturation', 'Saturation / Sättigung').classes('text-sm text-gray-500')
                        ui.range(
                            min=0,
                            max=255,
                            step=1,
                            value={'min': app_settings.lower_hsv[1], 'max': app_settings.upper_hsv[1]},
                            on_change=lambda event: update_hsv_range(state, 1, event.value),
                        ).props('label-always').classes('w-full')

                        label_t('hsv_value', 'Value / Helligkeit').classes('text-sm text-gray-500')
                        ui.range(
                            min=0,
                            max=255,
                            step=1,
                            value={'min': app_settings.lower_hsv[2], 'max': app_settings.upper_hsv[2]},
                            on_change=lambda event: update_hsv_range(state, 2, event.value),
                        ).props('label-always').classes('w-full')

def update_measurement_labels(
    state: SessionState,
    area_label: ui.label,
    convex_label: ui.label,
    damage_label: ui.label,
    damage_percent_label: ui.label,
    status_label: ui.label,
) -> None:
    last_measurement = state.last_measurement or default_measurement()
    area = last_measurement.get('area')
    convex_area = last_measurement.get('convex_area')
    damage_area = last_measurement.get('damage_area')
    damage_percent = last_measurement.get('damage_percent')
    green_label = text('metric_green_area', 'Grüne Fläche')
    convex_text = text('metric_convex_hull', 'Convex Hull')
    damage_text = text('metric_damage', 'Schaden')
    status_text = text('metric_status', 'Status')
    area_label.set_text(f'{green_label}: {area:.3f} cm2' if area is not None else f'{green_label}: -')
    convex_label.set_text(f'{convex_text}: {convex_area:.3f} cm2' if convex_area is not None else f'{convex_text}: -')
    damage_label.set_text(f'{damage_text}: {damage_area:.3f} cm2' if damage_area is not None else f'{damage_text}: -')
    damage_percent_label.set_text(
        f'{damage_text}: {damage_percent:.1f} %' if damage_percent is not None else f'{damage_text}: - %'
    )
    status_label.set_text(f"{status_text}: {last_measurement.get('status', '-')}")


async def archive_current_measurement(session_id: str, state: SessionState, status_label: ui.label | None = None) -> None:
    processed = await get_processed_result(session_id)
    if processed is None:
        message = text('archive_no_image', 'Kein Bild zum Archivieren vorhanden')
        if status_label is not None:
            status_label.set_text(message)
        ui.notify(message)
        return

    measurement = processed.get('measurement', {})
    settings = snapshot_settings(state)
    async with ARCHIVE_LOCK:
        image_paths = await run.io_bound(
            write_archive_entry,
            session_id,
            settings,
            measurement,
            processed.get('images', {}),
        )

    count = len(image_paths)
    message = text('archive_saved', 'Archiviert') + f': {count} ' + text('archive_images', 'Bilder')
    if status_label is not None:
        status_label.set_text(message)
    ui.notify(message)


def set_manual_damage_enabled(state: SessionState, session_id: str, enabled: bool) -> None:
    state.manual_damage_enabled = bool(enabled)
    state.manual_damage_revision += 1
    state.processed_cache = {}
    enabled_js = 'true' if state.manual_damage_enabled else 'false'
    ui.run_javascript(
        f"window.leafManualDamage?.['{session_id}'] && "
        f"(window.leafManualDamage['{session_id}'].enabled = {enabled_js})"
    )


def set_manual_brush_size(state: SessionState, session_id: str, value: Any) -> None:
    try:
        brush_size = int(float(value))
    except (TypeError, ValueError):
        return
    state.manual_brush_size = max(2, min(120, brush_size))
    ui.run_javascript(
        f"window.leafManualDamage?.['{session_id}'] && "
        f"(window.leafManualDamage['{session_id}'].brushSize = {state.manual_brush_size})"
    )


def set_auto_edge_damage_enabled(state: SessionState, enabled: bool) -> None:
    state.auto_edge_damage_enabled = bool(enabled)
    state.manual_damage_revision += 1
    state.processed_cache = {}


def set_manual_limit_to_leaf(state: SessionState, enabled: bool) -> None:
    state.manual_limit_to_leaf = bool(enabled)
    state.manual_damage_revision += 1
    state.processed_cache = {}


def set_manual_leaf_shrink_px(state: SessionState, value: Any) -> None:
    try:
        shrink_px = int(float(value))
    except (TypeError, ValueError):
        return
    state.manual_leaf_shrink_px = max(0, min(250, shrink_px))
    state.manual_damage_revision += 1
    state.processed_cache = {}


def set_show_auto_damage_on_cropped(state: SessionState, enabled: bool) -> None:
    state.show_auto_damage_on_cropped = bool(enabled)
    state.manual_damage_revision += 1
    state.processed_cache = {}


async def toggle_freeze(
    session_id: str,
    state: SessionState,
    button: ui.button,
    full_image: ui.interactive_image,
) -> None:
    async with get_processing_lock(state):
        if state.freeze_enabled:
            state.freeze_enabled = False
            state.frozen_frame = None
            button.set_text(text('freeze_button', 'Freeze'))
            state.input_revision += 1
            state.processed_cache = {}
            full_image.force_reload()
            if state.settings.mode_camera:
                ui.run_javascript(
                    f"window.leafMeasurementCamera?.['{session_id}']?.resumeAfterFreeze?.()"
                )
            ui.notify(text('notify_live_active', 'Livebild aktiv'))
            return

        if state.settings.mode_camera:
            if state.browser_frame is None:
                ui.notify(text('notify_no_camera_frame', 'Noch kein Browser-Kamerabild empfangen'))
                return
            state.frozen_frame = state.browser_frame.copy()
            state.browser_frame_jpeg = None
        elif state.uploaded_image is not None:
            state.frozen_frame = state.uploaded_image.copy()
        else:
            ui.notify(text('notify_no_image_to_freeze', 'Kein Bild zum Einfrieren'))
            return

        state.freeze_enabled = True
        button.set_text(text('live_button', 'Live'))
        state.input_revision += 1
        state.processed_cache = {}
        full_image.force_reload()
        if state.settings.mode_camera:
            ui.run_javascript(
                f"window.leafMeasurementCamera?.['{session_id}']?.freezeStream?.()"
            )
        ui.notify(text('notify_frame_frozen', 'Frame eingefroren'))


def load_language(event: Any, reload_page: bool = True) -> None:
    global language, sellang
    sellang = event.value
    language = read_language(event.value)
    if reload_page:
        ui.run_javascript('location.reload();')


async def handle_upload(state: SessionState, event: Any) -> None:
    content = await event.file.read()
    nparr = np.frombuffer(content, np.uint8)
    image = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)

    if image is None:
        ui.notify(text('warn_file_upload_fail', 'Fehler beim Hochladen der Datei'))
        return

    state.uploaded_image = image
    state.input_revision += 1
    state.processed_cache = {}
    state.settings.mode_camera = False
    ui.notify(text('notify_image_loaded', 'Bild geladen'))


async def disconnect() -> None:
    for client_id in list(Client.instances):
        await core.sio.disconnect(client_id)


def handle_sigint(signum: int, frame: Any) -> None:
    ui.timer(0.1, disconnect, once=True)
    ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)


async def cleanup() -> None:
    await disconnect()

app.on_startup(setup)
app.on_shutdown(cleanup)
signal.signal(signal.SIGINT, handle_sigint)

if __name__ == '__main__':
    ui.run(host='0.0.0.0', port=8080, show=False, reload=False)
