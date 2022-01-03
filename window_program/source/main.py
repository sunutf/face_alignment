import os, sys
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QEventLoop
from PyQt5.QtWidgets import QWidget, QApplication,\
                            QLabel, QFileDialog, QGroupBox,\
                            QPushButton, QLineEdit, QMessageBox,\
                            QTextBrowser, QVBoxLayout, QHBoxLayout,\
                            QCheckBox

from stdout import StdoutRedirect
from main_summary import main_job


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        # member variable
        self._stdout = StdoutRedirect()
        self._stdout.start()
        self._stdout.printOccur.connect(lambda x : self._append_text(x))

        self.findInputPathBtn.clicked.connect(self.findInputPathBtn_func)
        self.inputPathEditor.textChanged.connect(self.on_changed_input_path_func)
        self.actionBtn.clicked.connect(self.actionBtn_func)

        self.checkbox_cnt = 0
        self.checked_photo_type = []
           
    def init_ui(self):
        self.set_window_in_center()

        self.setWindowTitle("Photo Therapist")
        self.setWindowIcon(QtGui.QIcon('sources/icon.png'))

        groupbox1 = QGroupBox('Image Preparation')
        groupbox2 = QGroupBox('Therapy')

        # Text
        self.text1 = QLabel("Image Folder     ", self)
        # text1.move(20, 20)

        # Find Button
        self.findInputPathBtn = QPushButton("Browse", self)
        self.findInputPathBtn.setFont(QtGui.QFont("Arial", 9))

        # Input Path Editor
        self.inputPathEditor = QLineEdit(self)
        self.inputPathEditor.setFont(QtGui.QFont("Arial", 9))
        self.inputPathEditor.resize(320, 30)

        # Photo Type Check Box
        self.cb1 = QCheckBox('Landscape', self)
        self.cb2 = QCheckBox('Zoom In', self)
        self.cb3 = QCheckBox('Zoom Out', self)

        self.cb1.toggled.connect(self.on_changed_checkbox_func)
        self.cb2.toggled.connect(self.on_changed_checkbox_func)
        self.cb3.toggled.connect(self.on_changed_checkbox_func)

        # Action Button
        self.actionBtn = QPushButton("Action", self)
        self.actionBtn.setIcon(QtGui.QIcon('sources/action.png'))
        self.set_actionBtn_disabled()

        # Log Browser
        self.logBrowser = QTextBrowser(self)

        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        layout = QVBoxLayout()

        hbox1.addWidget(self.text1)
        hbox1.addWidget(self.findInputPathBtn)
        hbox2.addWidget(self.cb1)
        hbox2.addWidget(self.cb2)
        hbox2.addWidget(self.cb3)
        vbox1.addLayout(hbox1)
        vbox1.addWidget(self.inputPathEditor)
        vbox1.addLayout(hbox2)

        groupbox1.setLayout(vbox1)

        vbox2.addWidget(self.actionBtn)
        vbox2.addWidget(self.logBrowser)

        groupbox2.setLayout(vbox2)

        layout.addWidget(groupbox1, 4)
        layout.addStretch(1)
        layout.addWidget(groupbox2,20)
        self.setLayout(layout)

    def set_window_in_center(self):
        self.setFixedSize(480, 530)
        frameGm = self.frameGeometry()
        screen = QApplication.desktop().screenNumber(
                        QApplication.desktop().cursor().pos())
        centerPoint = QApplication.desktop().screenGeometry(screen).center()
        frameGm.moveCenter(centerPoint)
        self.move(frameGm.topLeft())

    def findInputPathBtn_func(self):
        dialog = QFileDialog()
        folder_path = dialog.getExistingDirectory(None, "Select the Image Folder")
        self.inputPathEditor.setText(folder_path)

    def on_changed_input_path_func(self):
        self.path = self.inputPathEditor.text()
        self.check_actionBtn_available()

    def on_changed_checkbox_func(self):
        self.checked_photo_type = []

        if self.cb1.isChecked():
            self.checked_photo_type.append("landscape")
        if self.cb2.isChecked():
            self.checked_photo_type.append("zoom_in")
        if self.cb3.isChecked():
            self.checked_photo_type.append("zoom_out")

        self.checkbox_cnt = len(self.checked_photo_type)
        self.check_actionBtn_available()
        
    def check_actionBtn_available(self):
        if (self.path or self.path != "") and (self.checkbox_cnt > 0):
            self.set_actionBtn_enabled()
        else:
            self.set_actionBtn_disabled()

    def set_actionBtn_enabled(self):
        self.actionBtn.setEnabled(True)
        self.actionBtn.setStyleSheet("color: white; background-color: rgb(58, 134, 255);")

    def set_actionBtn_disabled(self):
        self.actionBtn.setEnabled(False)
        self.actionBtn.setStyleSheet("background-color: #BBB;")

    def actionBtn_func(self):
        input_path = self.inputPathEditor.text()
        result = main_job(input_path=input_path, checked_photo_type=self.checked_photo_type)
        
        if result == None:
            QMessageBox.warning(self, "알림", "폴더 내에 이미지가 존재하지 않습니다.\n(jpg, jpeg 확장자 지원)")
        elif result == True:
            QMessageBox.information(self, "알림", "작업이 완료 되었습니다!")
            os.startfile(os.path.join(input_path, "output"))
        else:
            QMessageBox.warning(self, "오류", result)

    def _append_text(self, msg):
        self.logBrowser.moveCursor(QtGui.QTextCursor.End)
        self.logBrowser.insertPlainText(msg)
        # refresh textedit show
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()

    sys.exit(app.exec())