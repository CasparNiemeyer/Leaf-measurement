# Graph Report - .  (2026-06-16)

## Corpus Check
- Corpus is ~16,614 words - fits in a single context window. You may not need a graph.

## Summary
- 72 nodes · 182 edges · 7 communities
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_UI Session Controls|UI Session Controls]]
- [[_COMMUNITY_Vision Processing|Vision Processing]]
- [[_COMMUNITY_API Mask Endpoints|API Mask Endpoints]]
- [[_COMMUNITY_Settings Localization|Settings Localization]]
- [[_COMMUNITY_Freeze Locking|Freeze Locking]]
- [[_COMMUNITY_Legacy Camera UI|Legacy Camera UI]]

## God Nodes (most connected - your core abstractions)
1. `SessionState` - 22 edges
2. `ndarray` - 15 edges
3. `page()` - 12 edges
4. `get_processed_result()` - 9 edges
5. `get_session()` - 8 edges
6. `detect_aruco()` - 8 edges
7. `process_frame()` - 8 edges
8. `create_session()` - 7 edges
9. `snapshot_settings()` - 7 edges
10. `receive_manual_damage()` - 7 edges

## Surprising Connections (you probably didn't know these)
- `create_session()` --calls--> `MeasurementSettings`  [EXTRACTED]
  ui.py → ui.py  _Bridges community 3 → community 0_
- `ensure_manual_correct_mask()` --references--> `SessionState`  [EXTRACTED]
  ui.py → ui.py  _Bridges community 0 → community 2_
- `get_processing_lock()` --references--> `SessionState`  [EXTRACTED]
  ui.py → ui.py  _Bridges community 0 → community 4_
- `default_measurement()` --references--> `Any`  [EXTRACTED]
  ui.py → ui.py  _Bridges community 0 → community 1_
- `cache_key()` --references--> `Any`  [EXTRACTED]
  ui.py → ui.py  _Bridges community 1 → community 2_

## Import Cycles
- None detected.

## Communities (7 total, 0 thin omitted)

### Community 0 - "UI Session Controls"
Cohesion: 0.22
Nodes (17): label, archive_current_measurement(), create_session(), default_measurement(), default_session(), handle_upload(), page(), SessionState (+9 more)

### Community 1 - "Vision Processing"
Cohesion: 0.25
Nodes (16): Any, aruco_parameters(), blank_frame(), convert(), detect_aruco(), fallback_aruco_parameters(), fallback_marker_candidates(), find_external_contours() (+8 more)

### Community 2 - "API Mask Endpoints"
Cohesion: 0.21
Nodes (14): Request, Response, cache_key(), clear_manual_damage(), download_archive_csv(), ensure_manual_correct_mask(), ensure_manual_damage_mask(), ensure_manual_exclude_mask() (+6 more)

### Community 3 - "Settings Localization"
Cohesion: 0.21
Nodes (11): clamp_hsv(), cleanup(), disconnect(), hex_to_hsv(), hsv_to_hex(), load_language(), MeasurementSettings, read_language() (+3 more)

### Community 4 - "Freeze Locking"
Cohesion: 0.40
Nodes (5): button, interactive_image, Lock, get_processing_lock(), toggle_freeze()

### Community 5 - "Legacy Camera UI"
Cohesion: 0.40
Nodes (3): convert(), ndarray, Converts a frame from OpenCV to a JPEG image.      This is a free function (no

## Knowledge Gaps
- **3 isolated node(s):** `ndarray`, `Lock`, `button`
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SessionState` connect `UI Session Controls` to `API Mask Endpoints`, `Settings Localization`, `Freeze Locking`?**
  _High betweenness centrality (0.049) - this node is a cross-community bridge._
- **Why does `toggle_freeze()` connect `Freeze Locking` to `UI Session Controls`, `Settings Localization`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `page()` connect `UI Session Controls` to `Settings Localization`, `Freeze Locking`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **What connects `ndarray`, `Converts a frame from OpenCV to a JPEG image.      This is a free function (no`, `Lock` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._