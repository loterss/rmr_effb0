import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
import torch

from predict import getPredictions


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Rock RMR Predicted Application")
        self.setGeometry(150, 150, 500, 500)
        self.setFixedSize(534, 616)

        self.mainLayout = QVBoxLayout()
        self.setStyleSheet("QWidget{font-size: 10pt;}")

        # === 1V
        self.LayoutV01 = QHBoxLayout()
        self.top = QLabel("Upload the picture ...")
        self.LayoutV01.addWidget(self.top)

        # === 2V
        self.LayoutV02 = QHBoxLayout()
        self.url = QLineEdit()
        self.selectButton = QPushButton("...")
        self.selectButton.clicked.connect(self.getUrl)

        self.LayoutV02.addWidget(self.url)
        self.LayoutV02.addWidget(self.selectButton)

        # === 3V
        self.LayoutV03 = QHBoxLayout()
        self.image = QLabel(self)
        self.image.setScaledContents(True)
        self.image.setFixedSize(512, 512)
        self.image.setPixmap(QPixmap('images/none.png'))

        self.LayoutV03.addStretch()
        self.LayoutV03.addWidget(self.image)
        self.LayoutV03.addStretch()

        # === 4V
        self.LayoutV04 = QHBoxLayout()
        self.processButton = QPushButton("Process")
        self.processButton.clicked.connect(self.predictClass)
        self.closeButton = QPushButton("Close")
        self.closeButton.clicked.connect(self.closeWindow)
        self.LayoutV04.addWidget(self.processButton)
        self.LayoutV04.addWidget(self.closeButton)

        # === Main Layour Structure ===
        self.mainLayout.addLayout(self.LayoutV01)
        self.mainLayout.addLayout(self.LayoutV02)
        self.mainLayout.addStretch()
        self.mainLayout.addLayout(self.LayoutV03)
        self.mainLayout.addStretch()
        self.mainLayout.addLayout(self.LayoutV04)

        self.setLayout(self.mainLayout)

    def predictClass(self):
        propability_output, class_type = getPredictions(self.url.text())
        self.resultWindow = ResultWindow(propability_output, class_type, self.url.text())
        self.resultWindow.show()
        self.close()

    def getUrl(self):
        url = QFileDialog.getOpenFileName(self, "Open a file", "", "Image Files|*.jpg;*.jpeg;*.png;*.JPG;.*JPEG;*.PNG")
        self.url.setText(url[0])
        self.image.setPixmap(QPixmap(url[0]))

    def closeWindow(self):
        self.close()


class ResultWindow(QWidget):
    def __init__(self, probability_output, class_type, image_path):
        super().__init__()
        self.setWindowTitle("Results")
        self.setGeometry(150, 150, 500, 500)
        self.setFixedSize(534, 650)
        self.class_type = class_type
        self.probability_output = probability_output
        label = ['20-30', '30-40', '40-50', '50-60', '60-70', '70-80']

        self.mainLayout = QVBoxLayout()
        self.setStyleSheet("QWidget{font-size: 10pt;}")

        # === 1V
        LayoutV01 = QHBoxLayout()
        self.image = QLabel()
        self.image.setScaledContents(True)
        self.image.setFixedSize(512, 512)
        self.image.setPixmap(QPixmap(image_path))

        LayoutV01.addStretch()
        LayoutV01.addWidget(self.image)
        LayoutV01.addStretch()

        # === 2V
        LayoutV02 = QHBoxLayout()

        LayoutV02a = QVBoxLayout()
        LayoutV02a.addWidget(QLabel("RMR 20 - 30"), alignment=Qt.AlignCenter)
        prop01Text = QTextEdit()
        prop01Text.setDisabled(True)
        self.setEditText(prop01Text, str(np.round(self.probability_output[0], 2)))
        LayoutV02a.addWidget(prop01Text)

        LayoutV02b = QVBoxLayout()
        LayoutV02b.addWidget(QLabel("RMR 30 - 40"), alignment=Qt.AlignCenter)
        prop02Text = QTextEdit()
        prop02Text.setDisabled(True)
        self.setEditText(prop02Text, str(np.round(self.probability_output[1], 2)))
        LayoutV02b.addWidget(prop02Text)

        LayoutV02c = QVBoxLayout()
        LayoutV02c.addWidget(QLabel("RMR 40 - 50"), alignment=Qt.AlignCenter)
        prop03Text = QTextEdit()
        prop03Text.setDisabled(True)
        self.setEditText(prop03Text, str(np.round(self.probability_output[2], 2)))
        LayoutV02c.addWidget(prop03Text)

        LayoutV02d = QVBoxLayout()
        LayoutV02d.addWidget(QLabel("RMR 50 - 60"), alignment=Qt.AlignCenter)
        prop04Text = QTextEdit()
        prop04Text.setDisabled(True)
        self.setEditText(prop04Text, str(np.round(self.probability_output[3], 2)))
        LayoutV02d.addWidget(prop04Text)

        LayoutV02e = QVBoxLayout()
        LayoutV02e.addWidget(QLabel("RMR 60 - 70"), alignment=Qt.AlignCenter)
        prop05Text = QTextEdit()
        prop05Text.setDisabled(True)
        self.setEditText(prop05Text, str(np.round(self.probability_output[4], 2)))
        LayoutV02e.addWidget(prop05Text)

        LayoutV02f = QVBoxLayout()
        LayoutV02f.addWidget(QLabel("RMR 70 - 80"), alignment=Qt.AlignCenter)
        prop06Text = QTextEdit()
        prop06Text.setDisabled(True)
        self.setEditText(prop06Text, str(np.round(self.probability_output[5], 2)))
        LayoutV02f.addWidget(prop06Text)

        LayoutV02.addLayout(LayoutV02a)
        LayoutV02.addLayout(LayoutV02b)
        LayoutV02.addLayout(LayoutV02c)
        LayoutV02.addLayout(LayoutV02d)
        LayoutV02.addLayout(LayoutV02e)
        LayoutV02.addLayout(LayoutV02f)

        # === 3V
        LayoutV03 = QHBoxLayout()
        self.RMRLabel = QLabel("RMR Class: ")

        self.RMRClass = QTextEdit()
        self.setEditText(self.RMRClass, label[class_type])
        self.RMRClass.setReadOnly(True)

        self.ProbLabel = QLabel("Probability: ")

        self.ProbText = QTextEdit()
        self.setEditText(self.ProbText, str(np.round(max(self.probability_output), 2)) + ' %')
        self.ProbText.setReadOnly(True)

        LayoutV03.addWidget(self.RMRLabel)
        LayoutV03.addWidget(self.RMRClass)
        LayoutV03.addWidget(self.ProbLabel)
        LayoutV03.addWidget(self.ProbText)

        # === 4V
        LayoutV04 = QHBoxLayout()
        self.backbtn = QPushButton("Back")
        self.backbtn.clicked.connect(self.funcBack)
        self.closebtn = QPushButton("Close")
        self.closebtn.clicked.connect(self.funcClose)

        LayoutV04.addWidget(self.backbtn)
        LayoutV04.addWidget(self.closebtn)

        self.mainLayout.addLayout(LayoutV01)
        self.mainLayout.addLayout(LayoutV02)
        self.mainLayout.addLayout(LayoutV03)
        self.mainLayout.addLayout(LayoutV04)

        self.setLayout(self.mainLayout)

    def funcBack(self):
        self.close()
        mainWindow.show()

    def funcClose(self):
        self.close()

    def setEditText(self, textbox, text):
        textbox.setText(text)
        textbox.setAlignment(Qt.AlignCenter)
        textbox.setMinimumHeight(26)


if __name__ == "__main__":
    App = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(App.exec_())
