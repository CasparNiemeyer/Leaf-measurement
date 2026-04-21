from nicegui import Client, app, core, run, ui
import cv2
import numpy as np
import os
import json
from fastapi import Response
import base64
import signal

global language, sellang
langlist = [i.name for i in os.scandir('lang')]

black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')

def convert(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image.

    This is a free function (not in a class or inner-function),
    to allow run.cpu_bound to pickle it and send it to a separate process.
    """
    _, imencode_image = cv2.imencode('.jpg', frame)
    return imencode_image.tobytes()


def setup() -> None:
    global language, sellang
    sellang= 'de.json'
    language = json.loads(open('lang/de.json',encoding='utf-8').read())

    video_capture = cv2.VideoCapture(0)

    @app.get('/video/frame')
    # Thanks to FastAPI's `app.get` it is easy to create a web route which always provides the latest image from OpenCV.
    async def grab_video_frame() -> Response:
        if not video_capture.isOpened():
            return placeholder
        # The `video_capture.read` call is a blocking function.
        # So we run it in a separate thread (default executor) to avoid blocking the event loop.
        _, frame = await run.io_bound(video_capture.read)
        if frame is None:
            return placeholder
        # `convert` is a CPU-intensive function, so we run it in a separate process to avoid blocking the event loop and GIL.
        jpeg = await run.cpu_bound(convert, frame)
        return Response(content=jpeg, media_type='image/jpeg')

    
@ui.page('/')
def page():
    global matte
    dark = ui.dark_mode()
    with ui.row().classes('gap-16'):
        with ui.column().classes('w-100 items-stretch'):
            ui.label("insert camera feed here")
            # For non-flickering image updates and automatic bandwidth adaptation an interactive image is much better than `ui.image()`.
            video_image = ui.interactive_image('/video/frame').classes('w-full h-full')
            # A timer constantly updates the source of the image.
            ui.timer(interval=0.1, callback=video_image.force_reload)
        with ui.column().classes('w-100 items-stretch'):
            with ui.card().props('flat bordered'):
                with ui.row():
                    ui.select(langlist,label=language['select_language'],on_change=load_language,value=sellang)
                    ui.switch(language['dark_mode_switch']).bind_value(dark)
                
            with ui.card().props('flat bordered'):
                ui.label(language['label_settings'])

                with ui.card().props('flat bordered'):
                    with ui.expansion(language['label_basic_settings']).classes("w-80"):
                        mode = ui.checkbox(language['mode_checkbox']).tooltip(language['mode_checkbox_tooltip'])
                        ui.upload(label=language['image_upload'],max_files=1,on_upload=handle_upload,on_rejected=ui.notify(language['warn_file_upload_fail'])).classes("w-70").props('flat bordered').tooltip(language['image_upload_tooltip'])
                        physwidth = ui.number(language['physwidth_input']).tooltip(language['physwidth_input_tooltip'])
                        physheight = ui.number(language['physheight_input']).tooltip(language['physheight_input_tooltip'])
                        digwidth = ui.number(language['digwidth_input'],value=700).tooltip(language['digwidth_input_tooltip'])

                with ui.card().props('flat bordered'):
                    with ui.expansion(language['label_filter_settings']).classes("w-80"):
                        kernelsize = ui.number(label=language['kernelsize_input'],value=6).tooltip(language['kernelsize_input_tooltip'])
                        lower = ui.color_input(label=language['lower_input']).tooltip(language['lower_input_tooltip'])
                        upper = ui.color_input(label=language['upper_input']).tooltip(language['upper_input_tooltip'])

                with ui.card().props('flat bordered').classes('items-stretch'):
                    with ui.expansion(language['label_debug_settings']).classes("w-80"):
                        markers = ui.checkbox(language['marker_checkbox']).tooltip(language['marker_checkbox_tooltip'])
                        bounds = ui.checkbox(language['bound_checkbox']).tooltip(language['bound_checkbox_tooltip'])
                        contours = ui.checkbox(language['contours_checkbox']).tooltip(language['contours_checkbox_tooltip'])
                        convex = ui.checkbox(language['convex_checkbox']).tooltip(language['convex_checkbox_tooltip'])



    async def disconnect() -> None:
        """Disconnect all clients from current running server."""
        for client_id in Client.instances:
            await core.sio.disconnect(client_id)

    def handle_sigint(signum, frame) -> None:
        # `disconnect` is async, so it must be called from the event loop; we use `ui.timer` to do so.
        ui.timer(0.1, disconnect, once=True)
        # Delay the default handler to allow the disconnect to complete.
        ui.timer(1, lambda: signal.default_int_handler(signum, frame), once=True)

    async def cleanup() -> None:
        # This prevents ugly stack traces when auto-reloading on code change,
        # because otherwise disconnected clients try to reconnect to the newly started server.
        await disconnect()
        # Release the webcam hardware so it can be used by other applications again.
        video_capture.release()

    app.on_shutdown(cleanup)
    # We also need to disconnect clients when the app is stopped with Ctrl+C,
    # because otherwise they will keep requesting images which lead to unfinished subprocesses blocking the shutdown.
    signal.signal(signal.SIGINT, handle_sigint)

def load_language(event):
    global language, sellang
    sellang= event.value
    language = json.loads(open(f'lang/{event.value}',encoding='utf-8').read())
    ui.run_javascript('location.reload();')
    

async def handle_upload(event):
    content = await event.file.read()
    nparr = np.frombuffer(content, np.uint8)
    img_np = cv2.imdecode(nparr, flags=1)
    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    cv2.imshow("test",img_np)
    cv2.waitKey()

app.on_startup(setup)

ui.run()