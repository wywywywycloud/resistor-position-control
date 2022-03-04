import PySimpleGUI as sg
import cv2

from a import process_photo

sg.theme('Reddit')   # Add a touch of color

trs = []

for i in range(12):
    trs.append([sg.vtop(sg.Image('', size=(65, 350), key='trs'+str(i)))])

layout = [[sg.Column([[sg.Input(default_text='D:/documents/coursach/resistor-position-control/resistor-position-control/Resources/video_test.mp4', size=(80, 1), key='link'), sg.Button('enter')],
          [sg.Image('', size=(700, 400), key='board')],
          [sg.Image('', size=(700, 400), key='video')]]),
          sg.Column(trs),
          sg.Column([[sg.vtop(sg.Text('',size=(70, 350), key='dsc'))]])]]

window = sg.Window('Window Title', layout, size=(1440, 810), resizable=True)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break

    if event == 'enter':
        cap = cv2.VideoCapture( values['link'] )
        ret, frame = cap.read()

        frames = 0
        while ret:
            ret, frame = cap.read()

            if frames != 10:
                frames += 1
                continue

            frames = 0

            if frame.shape[0] == 0:
                break

            brd = process_photo(frame)
            frame = cv2.resize(frame, (700, 400))

            count = 0
            for tr in brd[1]:
                tr_bytes = cv2.imencode('.png', cv2.resize(tr, (350, 65)))[1].tobytes()
                window['trs'+str(count)].update(data=tr_bytes)
                count += 1

            des = ''
            for key, d in brd[2].items():
                des = des + str(d[0]) + ' (Вероятность ' + str(d[1]) + ')' + '\n\n\n\n\n'

            window['dsc'].update(des)

            brd_bytes = cv2.imencode('.png', cv2.resize(brd[0], (700,400)))[1].tobytes()
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['video'].update(data=imgbytes)
            window['board'].update(data=brd_bytes)
            window.refresh()

window.close()