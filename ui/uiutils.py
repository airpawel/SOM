import sys
from PyQt4 import QtCore, QtGui


class ParamLineEdit(QtGui.QLineEdit):
    lineEditSignal = QtCore.pyqtSignal(int, name='lineEditSig')

    def __init__(self, parent=None):
        QtGui.QLineEdit.__init__(self, parent)
        self.editingFinished.connect(self.line_edit_changed)

    def line_edit_changed(self):
        self.lineEditSignal.emit(int(self.text()))

    def color_change(self, color):
        c = QtCore.Qt.red if color == 'red' else QtCore.Qt.black
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Text, c)
        self.setPalette(palette)


class ParametersValidator(QtGui.QIntValidator):
    colorChanged = QtCore.pyqtSignal(str, name='colorChange')

    def __init__(self, min, max, parent=None):
        QtGui.QValidator.__init__(self, min, max, parent)

        self.min = min
        self.max = max
        self.value = None

    def validate(self, s, pos):
        # state = QtGui.QIntValidator.validate(self, s, pos)
        # print(state)
        # return state
        if not s:
            self.colorChanged.emit('black')
            # print(QtGui.QValidator.Intermediate, s, pos)
            return QtGui.QValidator.Intermediate, s, pos

        try:
            val = int(s)
        except ValueError:
            self.colorChanged.emit('red')
            # print(QtGui.QValidator.Intermediate, '', 0)
            return QtGui.QValidator.Intermediate, '', 0

        if val < self.min:
            self.colorChanged.emit('red')
            # print(QtGui.QValidator.Intermediate, s, pos)
            return QtGui.QValidator.Intermediate, s, pos
        elif self.min <= val <= self.max:
            self.value = val
            self.colorChanged.emit('black')
            # print(QtGui.QValidator.Acceptable, s, pos)
            return QtGui.QValidator.Acceptable, s, pos

        # self.colorChanged.emit('red')
        # print(QtGui.QValidator.Invalid, s, pos)
        return QtGui.QValidator.Invalid, s, pos

class Window(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)

        self.value = 1

        self.editLine = QtGui.QLineEdit(self)
        self.validator = ParametersValidator(10, 50, self.editLine)

        self.editLine_2 = ParamLineEdit(self)
        self.editLine_2.setValidator(self.validator)
        self.editLine_2.lineEditSig.connect(self.change_value)
        # self.validator = QtGui.QIntValidator(0, 50, editLine)
        # self.validator = MyDoubleValidator(10,100,2,self.editLine)

        self.editLine.setValidator(self.validator)


        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(self.editLine)

        # self.editLine.editingFinished.connect(self.change_value)

    def change_value(self, v):
        self.value = v
        print(self.value, v)

if __name__ == '__main__':
    import sys

    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 500, 100)
    window.show()
    sys.exit(app.exec_())
