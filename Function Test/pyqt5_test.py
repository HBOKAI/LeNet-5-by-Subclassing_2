from PyQt5 import QtWidgets
import sys
import threading

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('test.studio')
        self.resize(300, 200)
        self.ui()

    def ui(self):
        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(10,10,200,30)

        self.box = QtWidgets.QComboBox(self)
        self.box.addItems(['A','B','C','D'])
        self.box.setGeometry(10,50,200,30)
        # self.box.currentIndexChanged.connect(self.showMsg)

    def showMsg(self):
        text = self.box.currentText()
        num = self.box.currentIndex()
        self.label.setText(f'{num}:{text}')

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget()
    Form.show()
    sys.exit(app.exec_())