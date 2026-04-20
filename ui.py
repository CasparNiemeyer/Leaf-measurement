from nicegui import events,Client, app, core, run, ui
import cv2 as cv
import os
import json

os.environ['PYTHONTRACEMALLOC']='1'

global language, sellang
langlist = [i.name for i in os.scandir('lang')]

def startup():
    global language, sellang
    sellang= 'de.json'
    language = json.loads(open('lang/de.json',encoding='utf-8').read())


def root():
    ui.sub_pages({
        '/': main
    }).classes('w_full')

def main():
    global matte
    with ui.row().classes('gap-16'):
        with ui.column().classes('w-100 items-stretch'):
            langselect = ui.select(langlist,on_change=load_language,value=sellang)
        with ui.column().classes('w-100 items-stretch'):
            with ui.card().props('flat bordered'):
                ui.label(language['label_settings'])

                with ui.card().props('flat bordered'):
                    with ui.expansion(language['label_basic_settings']).classes("w-80"):
                        mode = ui.checkbox(language['mode_checkbox']).tooltip(language['mode_checkbox_tooltip'])
                        ui.upload(label=language['image_upload'],max_files=1,on_rejected=ui.notify(language['warn_file_upload_fail'])).classes("w-3/4").tooltip(language['image_upload_tooltip'])
                        physwidth = ui.number(language['physwidth_input']).tooltip(language['physwidth_input_tooltip'])
                        physheight = ui.number(language['physheight_input']).tooltip(language['physheight_input_tooltip'])
                        digwidth = ui.number(language['digwidth_input']).tooltip(language['digwidth_input_tooltip'])

                with ui.card().props('flat bordered'):
                    with ui.expansion(language['label_filter_settings']).classes("w-80"):
                        kernelsize = ui.number(label=language['kernelsize_input']).tooltip(language['kernelsize_input_tooltip'])
                        lower = ui.color_input(label=language['lower_input']).tooltip(language['lower_input_tooltip'])
                        upper = ui.color_input(label=language['upper_input']).tooltip(language['upper_input_tooltip'])

                with ui.card().props('flat bordered').classes('items-stretch'):
                    with ui.expansion(language['label_debug_settings']).classes("w-80"):
                        markers = ui.checkbox(language['marker_checkbox']).tooltip(language['marker_checkbox_tooltip'])
                        bounds = ui.checkbox(language['bound_checkbox']).tooltip(language['bound_checkbox_tooltip'])
                        contours = ui.checkbox(language['contours_checkbox']).tooltip(language['contours_checkbox_tooltip'])
                        convex = ui.checkbox(language['convex_checkbox']).tooltip(language['convex_checkbox_tooltip'])

def load_language(event):
    global language, sellang
    sellang= event.value
    language = json.loads(open(f'lang/{event.value}',encoding='utf-8').read())
    ui.run_javascript('location.reload();')
    

def handle_upload(e: events.UploadEventArguments):
    e.file.save('tmp/images/test.jpeg')

def run():
    print(matte.value)

app.on_startup(startup)

ui.run(root)