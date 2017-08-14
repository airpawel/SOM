import sys
import random
from PyQt4 import QtGui, QtCore
from PyQt4.Qt import *
import datetime
import numpy as np
from src.controllogic import Control


class MapFrame(QtGui.QFrame):

    nStepsLearnCall = QtCore.pyqtSignal([int, int], name='twoHundredStepsCall')
    startSimulation = QtCore.pyqtSignal(name='startSimulation')
    iterationChange = QtCore.pyqtSignal(str, name='iterationChange')
    oneStepLearnCall = QtCore.pyqtSignal(int, name='oneStepLearnCall')
    simulationSpeed = 3000

    def __init__(self, parent, path):
        super(MapFrame, self).__init__(parent)

        self.init_Board(path)

    def init_Board(self, path):
        self.timer = QtCore.QBasicTimer()
        self.isStarted = False
        self.isPaused = False
        self.curImageId = 0
        self.dim = 150
        self.label = QLabel(self)
        self.label.resize(self.dim, self.dim)
        self.label.setUpdatesEnabled(True)
        self.k = 0
        self.som_data = None
        self.steps = 1
        self.pix_edge_num = 12

        print('label ', self.label.geometry())

        # change background
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Background, QtCore.Qt.darkGray)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

    def start(self):
        if self.isPaused:
            return
        self.isStarted = True
        self.timer.start(MapFrame.simulationSpeed, self)

    def pause(self):
        print('pause')
        if not self.isStarted:
            return
        self.isPaused = not self.isPaused
        if self.isPaused:
            self.timer.stop()
            print('timer held')
        else:
            self.timer.start(MapFrame.simulationSpeed, self)
            print('timer released')
        self.update()

    def timerEvent(self, event):
        # t = datetime.datetime.now()
        # print(t.second, t.microsecond)

        if self.steps != 0:
            self.nStepsLearnCall.emit(self.k, self.steps)
            self.k += self.steps
            self.steps = 1
        else:
            self.oneStepLearnCall.emit(self.k)
            self.k += 1

        self.pixmap = self.get_image_from_raw_data()
        # self.img = QtGui.QImage('pzielony.png')
        # self.pixmap = QtGui.QPixmap.fromImage(self.img)
        self.label.setPixmap(self.pixmap)
        self.iterationChange.emit(str(self.k))
        self.label.show()

    def n_steps_button_handler(self, steps):

        self.steps = steps
        print(self.steps)

    def change_sim_speed(self, value):
        MapFrame.simulationSpeed = value
        print(MapFrame.simulationSpeed)

    def change_som_map_scaling(self, value):
        self.dim = value
        self.label.setGeometry(QtCore.QRect(0,0, value, value))

    def change_som_map_data(self, data, size):
        self.pix_edge_num = size
        self.som_data = np.array(data)

    def get_image_from_raw_data(self):
        data = self.som_data
        # data = np.random.randint(0, 256, size=(50, 50, 3)).astype(np.uint8)
        img = QtGui.QImage(self.pix_edge_num, self.pix_edge_num, QtGui.QImage.Format_RGB32)
        for x in range(self.pix_edge_num):
            for y in range(self.pix_edge_num):
                if max(data[x][y]) < 255:
                    img.setPixel(x, y, QtGui.QColor(*data[x][y]).rgb())
                    img1 = img.scaled(self.dim,self.dim,Qt.KeepAspectRatio)
                else:
                    print('overflow ', data[x][y])
                    img.setPixel(x, y, QtGui.QColor(*[255,255,255]).rgb())
                    img1 = img.scaled(self.dim, self.dim, Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(img1)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    # sender = Sender()
    sys.exit(app.exec_())
