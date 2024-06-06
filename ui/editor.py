# SPDX-FileCopyrightText: 2024-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: MIT
import sys
from pathlib import Path

import cv2
from PySide6.QtGui import QAction, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QPushButton, QSlider, QWidget, QVBoxLayout,
                               QHBoxLayout, QFileDialog, QSpacerItem, QLineEdit, QGroupBox, QCheckBox, QRadioButton,
                               QListWidgetItem, QListWidget, QLabel)
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QUrl, Qt, QTime

from detector import Detector
from detector.de_id import DeId


class VideoEditor(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        # self.detector = Detector()
        self.setWindowTitle("VideoMasker")
        self.resize(1280, 800)
        self.selected_video_path = ''

        open_action = QAction('&Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open video')
        open_action.triggered.connect(self.get_file_path)

        close_action = QAction('&Close', self)
        close_action.setStatusTip('Close video')
        close_action.triggered.connect(self.close_video)

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        menubar.setStyleSheet("font-weight: bold;"
                              "text-decoration: underline;")
        file_menu = menubar.addMenu('File')
        file_menu.setStyleSheet("font-weight: normal;"
                                "text-decoration: none;")
        file_menu.addAction(open_action)
        file_menu.addAction(close_action)
        menubar.addSeparator()

        main_layout = QHBoxLayout()
        # --------------player_layout start--------------
        player_layout = QVBoxLayout()
        player_layout.setSpacing(10)

        video_widget = QVideoWidget()
        video_widget.setAutoFillBackground(True)

        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)
        self.slider.setEnabled(False)

        self.current_time = QLineEdit('00:00:00:00')
        self.current_time.setReadOnly(True)
        self.current_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_time.setFixedWidth(100)
        self.current_time.setUpdatesEnabled(True)
        self.current_time.selectionChanged.connect(lambda: self.current_time.setSelection(0, 0))

        self.total_time = QLineEdit('00:00:00:00')
        self.total_time.setReadOnly(True)
        self.total_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.total_time.setFixedWidth(100)
        self.total_time.setUpdatesEnabled(True)
        self.total_time.selectionChanged.connect(lambda: self.total_time.setSelection(0, 0))

        slider_layout.addWidget(self.current_time)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.total_time)

        self.button_group = QGroupBox()
        self.button_group.setFlat(True)
        self.button_group.setFixedHeight(45)
        self.analyze_button = QPushButton("Analyze")
        self.analyze_button.setFixedSize(80, 35)
        self.analyze_button.setStyleSheet("background-color: rgb(225, 110, 50);")
        self.analyze_button.clicked.connect(self.analyze)
        self.backward_button = QPushButton("◀◀")
        self.backward_button.setFixedSize(80, 35)
        self.backward_button.setStyleSheet("background-color: rgb(225, 110, 50);")
        self.backward_button.clicked.connect(self.backward)
        self.play_button = QPushButton("▶")
        self.play_button.setFixedSize(80, 35)
        self.play_button.setStyleSheet("background-color: rgb(225, 110, 50);")
        self.play_button.clicked.connect(self.play_pause)
        self.forward_button = QPushButton("▶▶")
        self.forward_button.setFixedSize(80, 35)
        self.forward_button.setStyleSheet("background-color: rgb(225, 110, 50);")
        self.forward_button.clicked.connect(self.forward)
        self.deid_button = QPushButton("De-Id")
        self.deid_button.setFixedSize(80, 35)
        self.deid_button.setStyleSheet("background-color: rgb(225, 110, 50);")
        self.deid_button.clicked.connect(self.deid)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.analyze_button)
        button_layout.addWidget(self.backward_button)
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.forward_button)
        button_layout.addWidget(self.deid_button)
        self.button_group.setLayout(button_layout)
        self.button_group.setEnabled(False)

        function_layout = QHBoxLayout()
        function_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        function_layout.setSpacing(30)

        target_groupbox = QGroupBox('Targets')
        target_groupbox.setFixedSize(160, 100)
        self.face_checkbox = QCheckBox('Face')
        self.face_checkbox.setChecked(True)
        self.plate_checkbox = QCheckBox('Plate')
        target_layout = QVBoxLayout()
        target_layout.addWidget(self.face_checkbox)
        target_layout.addWidget(self.plate_checkbox)
        target_groupbox.setLayout(target_layout)

        method_groupbox = QGroupBox('De-Identification')
        method_groupbox.setFixedSize(160, 100)
        self.blur = QRadioButton('Blur')
        self.blur.setChecked(True)
        self.pixelate = QRadioButton('Pixelate')
        self.emoji = QRadioButton('Emoji')
        method_box = QVBoxLayout()
        method_box.addWidget(self.blur)
        method_box.addWidget(self.pixelate)
        method_box.addWidget(self.emoji)
        method_groupbox.setLayout(method_box)

        function_layout.addWidget(target_groupbox)
        function_layout.addWidget(method_groupbox)

        player_layout.addWidget(video_widget)
        player_layout.addLayout(slider_layout)
        player_layout.addWidget(self.button_group)
        player_layout.addLayout(function_layout)

        main_layout.addLayout(player_layout)
        # --------------player_layout end--------------
        # --------------image_layout start--------------
        image_layout = QVBoxLayout()
        self.image_list = QListWidget()
        self.image_list.setFixedWidth(200)
        image_layout.addWidget(self.image_list)
        main_layout.addLayout(image_layout)
        # --------------image_layout end--------------
        container = QWidget()
        container.setLayout(main_layout)

        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(video_widget)

        self.media_player.positionChanged.connect(self.position_changed)
        self.media_player.durationChanged.connect(self.duration_changed)

        self.setCentralWidget(container)

    def open_video(self, file_name=None):
        if file_name is None:
            self.media_player.setSource(QUrl.fromLocalFile(self.selected_video_path))
        else:
            self.media_player.setSource(QUrl.fromLocalFile(file_name))
        self.media_player.play()
        self.media_player.pause()
        self.button_group.setEnabled(True)
        self.deid_button.setEnabled(False)
        self.slider.setEnabled(True)

    def get_file_path(self):
        selected_file = QFileDialog.getOpenFileName(self, caption='Open file',
                                                    dir='../',
                                                    filter='Video (*.mp4 *.avi *.mov)')
        self.close_video()
        self.selected_video_path = selected_file[0]
        self.open_video()

    def close_video(self, hot_reload=False):
        if not hot_reload:
            self.image_list.clear()
            self.selected_video_path = ''
        self.media_player.setSource('')
        self.slider.setValue(0)
        self.current_time.setText('00:00:00:00')
        self.total_time.setText('00:00:00:00')
        self.play_button.setText("▶")
        self.button_group.setEnabled(False)
        self.slider.setEnabled(False)

    def analyze(self):
        self.image_list.clear()
        if self.media_player.isPlaying():
            self.play_button.setText("▶")
            self.media_player.pause()
        cap = cv2.VideoCapture(self.selected_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        face_detector = Detector(fps=fps, width=width, height=height)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections, cropped_image_list = face_detector(frame=frame)
            if cropped_image_list:
                for img in cropped_image_list:
                    self.add_image(img)
            img = face_detector.visualize(original_image=frame, detections=detections)
            cv2.imshow('Analyzing...', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()
        del face_detector
        self.deid_button.setEnabled(True)

    def add_image(self, image_path):
        file_name = Path(image_path).stem
        face_id = file_name.split('_')[0]
        item = QListWidgetItem()

        container = QWidget()
        main_layout = QHBoxLayout()

        # Create QLabel with HTML content
        label = QLabel()
        html = f"""
        <div style="text-align:center;">
            <img src="{image_path}" height="100" style="vertical-align:middle;"><br/>
            <span style="font-family: Arial; font-size: 10pt;">face {face_id}</span>
        </div>
        """
        label.setText(html)
        label.setObjectName(face_id)

        # Create QCheckBox
        checkbox = QCheckBox()
        main_layout.addWidget(checkbox)
        main_layout.addWidget(label)

        container.setLayout(main_layout)
        item.setSizeHint(container.sizeHint())
        self.image_list.addItem(item)
        self.image_list.setItemWidget(item, container)

    def get_selected_items(self):
        selected_labels = []
        for index in range(self.image_list.count()):
            item = self.image_list.item(index)
            widget = self.image_list.itemWidget(item)
            checkbox = widget.findChild(QCheckBox)
            if checkbox.isChecked():
                label = widget.findChild(QLabel)
                selected_labels.append(int(label.objectName()))
        return selected_labels

    def deid(self):
        if self.media_player.isPlaying():
            self.play_button.setText("▶")
            self.media_player.pause()
        selected_items = self.get_selected_items()
        if self.blur.isChecked():
            method = 'blur'
        elif self.pixelate.isChecked():
            method = 'pixelate'
        else:
            method = 'emoji'

        cap = cv2.VideoCapture(self.selected_video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = Path("output")
        output_path.mkdir(exist_ok=True)
        vid_name = Path(self.selected_video_path).stem
        output_vid_name = f"{output_path}/{vid_name}_{method}.mp4"
        writer = cv2.VideoWriter(
            output_vid_name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        face_detector = Detector(fps=fps, width=width, height=height)
        de_id = DeId(method=method, exclude_face_ids=selected_items)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = face_detector(frame=frame, mode='de-id')
            de_id(detections, frame, strength_factor=1)
            writer.write(frame)
            cv2.imshow('Processing...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
        cap.release()
        writer.release()
        del face_detector
        # self.close_video(hot_reload=True)
        # TODO: [mpeg4 @ 0x3300ef4d0] Failed setup for format videotoolbox_vld: hwaccel initialisation returned error.
        # self.open_video(Path(output_vid_name).absolute().as_posix())

    def play_pause(self):
        if self.media_player.mediaStatus() != QMediaPlayer.MediaStatus.NoMedia:
            if self.media_player.isPlaying():
                self.play_button.setText("▶")
                self.media_player.pause()
            else:
                self.play_button.setText("❚❚")
                self.media_player.play()

    def backward(self):
        self.media_player.setPosition(self.media_player.position() - 10000)

    def forward(self):
        self.media_player.setPosition(self.media_player.position() + 10000)

    def set_position(self, position):
        self.media_player.setPosition(position)

    def position_changed(self, position):
        self.slider.setValue(position)
        mtime = QTime(0, 0, 0, 0)
        mtime = mtime.addMSecs(self.media_player.position())
        text = f"{mtime.hour():02}:{mtime.minute():02}:{mtime.second():02}:{mtime.msec() % 100:02}"
        self.current_time.setText(text)

    def duration_changed(self, duration):
        self.slider.setRange(0, duration)
        mtime = QTime(0, 0, 0, 0)
        mtime = mtime.addMSecs(self.media_player.duration())
        text = f"{mtime.hour():02}:{mtime.minute():02}:{mtime.second():02}:{mtime.msec() % 100:02}"
        self.total_time.setText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_player = VideoEditor()
    video_player.show()
    sys.exit(app.exec())
