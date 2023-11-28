import PySimpleGUI as sg
import numpy as np
import cv2
import tkinter as tk
import datetime , os, time
# global 
FPS = 10
WIDTH = 640
HEIGHT = 480
RECT_COLOR = (0,0,255) # BGR
RECOGNIZE_RANGE_COLOR = (0,255,0) #BGR
SENSITIVITY = 0.8


def main():
    
    # Webカメラを取得
    # camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FPS,FPS)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    # 1つ前のフレーム
    prev_frame = None
    ## 録画用パラメータ
    # 検知フラグ
    recognize = False
    # 撮影中フラグ
    shooting = False
    # マージンタイム(前後の余白)
    margin_time = 5
    # ビデオ
    video = None
    video_setting = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # 保存先ディレクトリ
    save_dir = None
    
    # 1. 描画エリア
    ret, frame = camera.read()
    height, width, channels = frame.shape[:3]
    canvas = sg.Graph(
        (width,height),
        (0,height),
        (width,0),
        background_color='#555555',
        pad=(5,5),
        key='CANVAS'
    )

    layout = [
        [   
            sg.Menu(
                [
                    [
                        'ファイル(&F)', 
                        [
                            '新規作成 (&N)::MENU_NEW::', 
                            '開く (&O)::MENU_OPEN::', 
                            '保存 (&S)::MENU_SAVE::', 
                            '名前を付けて保存 (&A)::MENU_SAVEAS::', 
                            '終了 (&X)::MENU_EXIT::', 
                        ], 
                    ]
                ], 
            ),
        ],
        [
            sg.Text('保存先: '),
            sg.Input('',readonly=True,key='SAVE_DIR'),
            sg.Button('選択',size=(10,1),key='SELECT_DIR')
        ],
        [
            sg.Checkbox('縦横比固定', True, pad=(0, 0), key='ENABLE_ASPECT'), 
            sg.Combo(['画面サイズ', '1:1', '3:2', '4:3', '16:9', '2:3', '3:4', '9:16', '指定比率'], '画面サイズ', size=(12, 1), readonly=True, enable_events=True, key='ASPECT_MODE'), 
            sg.Text('', size=(2, 1), pad=(0, 0)), 
            sg.Column(
                [
                    [
                        sg.Input('', size=(5, 1), pad=(0, 0), key='ASPECT_X'), 
                        sg.Text(' : ', pad=(0, 0)), 
                        sg.Input('', size=(5, 1), pad=(0, 0), key='ASPECT_Y'), 
                    ]
                ], 
                visible=False, 
                key='COLUMN_ASPECT', 
            ),
        ],
        [
            # table_source, 
            canvas, 
            sg.Column(
                [
                    [
                        sg.Text('ステータス'),
                        sg.Input('設定中',size=(10,1),readonly=True,key='STATUS'),
                    ],
                    [
                        sg.Text('検知範囲 (左上が原点)')
                    ],
                    [
                        sg.Text('左上 (x:y)'),
                        sg.Spin(list(range(width+1)), 0, size=(5, 1), pad=(0, 0), key='START_X'), 
                        sg.Text('px : ', pad=(0, 0)), 
                        sg.Spin(list(range(height+1)), 0, size=(5, 1), pad=(0, 0), key='START_Y'), 
                        sg.Text('px', pad=(0, 0)), 
                    ],
                    [
                        sg.Text('右下 (x:y)'),
                        sg.Spin(list(range(width+1)), width, size=(5, 1), pad=(0, 0), key='END_X'), 
                        sg.Text('px : ', pad=(0, 0)), 
                        sg.Spin(list(range(height+1)), height, size=(5, 1), pad=(0, 0), key='END_Y'), 
                        sg.Text('px', pad=(0, 0)), 
                    ],
                    [
                        sg.Button('範囲設定',size=(10,1),key='SET_RANGE'),
                        sg.Button('範囲リセット',size=(10,1),key='RESET_RANGE')
                    ],
                    [
                        sg.Button('実行', size=(10, 1), key='RECOGNIZE'),
                        sg.Button('実行停止', size=(10, 1), key='STOP')
                    ]
                ]
            ),
        ],
        [
            sg.Multiline(default_text='',size=(0,5),key='LOG',disabled=True,do_not_clear=True)
        ]
    ]
    
    # 2. ウィンドウの生成
    window = sg.Window(
        title='動体検知録画',
        layout=layout, 
        resizable=False, 
        # size=(800, 600), 
        margins=(0, 0), 
    )
    window.finalize()
    window['LOG'].expand(expand_x=True)
    
    # canvas.bind('<MouseWheel>', '__SCROLL') # 表示画像のスクロール変更
    canvas.bind('<ButtonPress-1>', '__LEFT_PRESS') # 範囲選択開始
    canvas.bind('<Button1-Motion>', '__DRAG') # ドラッグで範囲選択
    canvas.bind('<Button1-ButtonPress-3>', '__DRAG_CANCEL') # ドラッグ中止（ドラッグ中に右クリック）
    canvas.bind('<ButtonRelease-1>', '__LEFT_RELEASE') # ドラッグ範囲確定
    canvas.bind('<Double-ButtonPress-1>', '__DOUBLE_LEFT') # 選択範囲解除

    canvas.drag_from = None # ドラッグ開始位置
    canvas.current = None # カーソル現在位置
    canvas.selection = None # 選択範囲
    canvas.selection_figure = None # 選択範囲の描画ID

    start_x, start_y = 0, 0
    end_x, end_y = width, height
        
    # 3. GUI処理
    while True:
        event, values = window.read(timeout=20)
        # 終了
        if event is None or '::MENU_EXIT::' in event:
            break
        # 縦横比選択
        if event == 'ASPECT_MODE':
            if values['ASPECT_MODE'] == '指定比率':
                aspect_visible = True
            else:
                aspect_visible = False
            window['COLUMN_ASPECT'].update(aspect_visible)
        # 保存先ディレクトリ設定
        if event == 'SELECT_DIR':
            save_dir = tk.filedialog.askdirectory(initialdir=os.path.abspath(os.path.dirname(__file__)))
            window['SAVE_DIR'].update(save_dir)
        # 画像を1フレームずつ取得する。
        ret, frame = camera.read()
        cv2.putText(frame, datetime.datetime.now().strftime('%Y,%b,%d %H:%M:%S'), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,255), 1, cv2.LINE_AA)
        # 画像があれば処理
        if ret is True:
            # アス比取得
            if values['ENABLE_ASPECT']:
                if ':' in values['ASPECT_MODE']:
                    (x, y) = values['ASPECT_MODE'].split(':')
                    aspect = np.array((
                        int(x), 
                        int(y), 
                    ))
                elif values['ASPECT_MODE'] == '指定比率':
                    try:
                        aspect = np.array((int(values['ASPECT_X']), int(values['ASPECT_Y'])))
                    except ValueError:
                        aspect = None
                # get_size()で表示エリアサイズを測定
                elif values['ASPECT_MODE'] == '画面サイズ':
                    aspect = np.array(canvas.get_size())
            else:
                aspect = None
            # 矩形選択開始
            if event == 'CANVAS__LEFT_PRESS':
                canvas.drag_from = np.array((canvas.user_bind_event.x, canvas.user_bind_event.y))
                canvas.current = np.array((canvas.user_bind_event.x, canvas.user_bind_event.y))
            # ドラッグ処理
            if event == 'CANVAS__DRAG' and canvas.drag_from is not None:
                canvas.current = np.array((canvas.user_bind_event.x, canvas.user_bind_event.y))
                canvas.selection = np.array((canvas.drag_from, canvas.current))
                canvas.selection = np.array((canvas.selection.min(axis=0), canvas.selection.max(axis=0))) # ((左上), (右下))の順に並び替える
                # アスペクト比の適用
                if aspect is not None:
                    selection_size = (canvas.selection[1] - canvas.selection[0])
                    aspected = (aspect[0]/aspect[1]*selection_size[1], aspect[1]/aspect[0]*selection_size[0]) + canvas.selection[0]
                    canvas.selection = np.vstack([canvas.selection, [aspected]]) # アス比適応時と合体させる
                canvas.selection = np.array((canvas.selection.min(axis=0), canvas.selection.max(axis=0))).clip((0, 0), (width,height)) # アス比適応、上下限適応
            # 矩形選択キャンセル
            if event == 'CANVAS__DRAG_CANCEL':
                canvas.selection = None
                canvas.drag_from = None
            # 矩形選択完了
            if event == 'CANVAS__LEFT_RELEASE' and canvas.selection is not None:
                # 面積0の選択範囲はスキップ
                if (canvas.selection[1] - canvas.selection[0]).min() >= 1:
                    canvas.selection = canvas.selection.astype(int)
                    window['START_X'].update(canvas.selection[0][0])
                    window['START_Y'].update(canvas.selection[0][1])
                    window['END_X'].update(canvas.selection[1][0])
                    window['END_Y'].update(canvas.selection[1][1])
                    start_x = canvas.selection[0][0]
                    start_y = canvas.selection[0][1]
                    end_x = canvas.selection[1][0]
                    end_y = canvas.selection[1][1]
                    # 選択範囲が変わったら1つ前のフレームをリセット
                    prev_frame = None
                # 範囲を記録したらリセット
                canvas.selection = None
                canvas.drag_from = None
            
            # 範囲選択
            if event == 'SET_RANGE':
                start_x = values['START_X']
                start_y = values['START_Y']
                end_x = values['END_X']
                end_y = values['END_Y']
                # 選択範囲が変わったら1つ前のフレームをリセット
                prev_frame = None
            # 範囲リセット
            if event == 'RESET_RANGE':
                start_x, start_y = 0, 0
                end_x, end_y = width, height
                window['START_X'].update(0)
                window['START_Y'].update(0)
                window['END_X'].update(width)
                window['END_Y'].update(height)
                # 選択範囲が変わったら1つ前のフレームをリセット
                prev_frame = None
            
            if event == 'RECOGNIZE':
                if save_dir is None or save_dir == () or os.path.isdir(save_dir) is False:
                    recognize = False
                    if save_dir is None:
                        sg.popup('保存先を選択してください。')
                    else:
                        sg.popup('保存先が存在しません。')
                else :
                    recognize = True
                    window['LOG'].print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 検知開始")
                    window['STATUS'].update('検知中')
            if event == 'STOP':
                recognize = False
                window['LOG'].print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 検知停止")
                window['STATUS'].update('実行停止')

            # 動体検知ロジック
            original_frame = np.copy(frame)
            frame, prev_frame, movement = move_recognize(frame, prev_frame, start_x, start_y,end_x,end_y)
            # 実行中
            if recognize is True :
                if movement is True:
                    last_movement_time = int(time.time())
                    if shooting is False:
                        window['LOG'].print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 録画開始")
                        window['STATUS'].update('録画中')
                        shooting = True
                        file_name = f"{save_dir}/{datetime.datetime.fromtimestamp(last_movement_time).strftime('%Y-%m-%d-%H_%M_%S')}.mp4"
                        window['LOG'].print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]  {file_name}")
                        video = cv2.VideoWriter(file_name, video_setting, FPS, (width,height))
                if video is not None:
                    if shooting is True :
                        video.write(original_frame)
                        
                    if (time.time() - last_movement_time > margin_time):
                        window['LOG'].print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 録画終了")
                        window['STATUS'].update('検知中')
                        video.release()
                        video = None
                        shooting = False
            else :
                if video is not None:
                    video.release()
                    video = None
                    shooting = False
            canvas.erase()
            canvas.draw_image_plus(frame)
    # 選択範囲表示
        # if canvas.selection_figure is not None:
        #     canvas.delete_figure(canvas.selection_figure)
        if canvas.selection is not None:
            canvas.selection_figure = canvas.draw_rectangle(
                list(canvas.selection[0]), 
                list(canvas.selection[1]), 
                line_color='#FF0000', 
                line_width=2
            )
    camera.release()
    cv2.destroyAllWindows()
    window.close()
    
# 画像を表示（sg.Graph インスタンスメソッド）
def draw_image_plus(self, img, location=(0,0)):
    if type(img) == np.ndarray:
        img = cv2.imencode('.png', img)[1].tobytes()
    id_ = self.draw_image(data=img, location=location)
    return id_
sg.Graph.draw_image_plus = draw_image_plus

# 動体検知
def move_recognize(frame, prev, start_x=0, start_y=0, end_x=WIDTH,end_y=HEIGHT):
    # 画像をトリミング
    trim = frame[start_y:end_y, start_x:end_x]
    # グレースケールに変換
    trim_gray= cv2.cvtColor(trim,cv2.COLOR_BGR2GRAY)
    movement = False
    if prev is None:
        prev = trim_gray.copy().astype("float")
        return frame , prev, movement
    # 現在のフレームと移動平均との差を計算
    cv2.accumulateWeighted(trim_gray, prev, SENSITIVITY)
    frameDelta = cv2.absdiff(trim_gray, cv2.convertScaleAbs(prev))

    # デルタ画像を閾値処理を行う
    thresh = cv2.threshold(frameDelta, 3, 255, cv2.THRESH_BINARY)[1]
    
    #輪郭のデータを得る
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # 差分があった点を画面に描く
    for target in contours:
        x, y, w, h = cv2.boundingRect(target)
        x = x + start_x
        y = y + start_y
        if w < 30 or h < 30: continue # 小さな変更点は無視
        if movement is False:
            movement = True
        # break
        cv2.rectangle(frame, (x, y), (x+w, y+h), RECT_COLOR, 2)
    # 認識範囲を表示
    cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),RECOGNIZE_RANGE_COLOR,2)
    return frame, prev, movement

###########################################################
if __name__ == '__main__':
    main()
