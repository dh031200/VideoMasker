# SPDX-FileCopyrightText: 2024-present dh031200 <imbird0312@gmail.com>
#
# SPDX-License-Identifier: MIT
import sys
from PySide6.QtWidgets import QApplication
from ui import VideoEditor

if __name__ == "__main__":
    app = QApplication(sys.argv)
    video_player = VideoEditor()
    video_player.show()
    sys.exit(app.exec())
