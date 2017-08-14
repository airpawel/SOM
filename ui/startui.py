import sys
from PyQt4 import QtGui
from ui.qsom import Ui_QSom
from ui.mapframe import MapFrame
from src.controllogic import Control
from ui.uiutils import ParametersValidator
from src.learningasset import LearningAsset


class StartQT4(QtGui.QMainWindow):

    def __init__(self, path, parent=None,):
        QtGui.QMainWindow.__init__(self, parent)
        self.ui = Ui_QSom()
        self.ui.setupUi(self)
        self.setCentralWidget(self.ui.horizontalLayoutWidget)

        self.data = LearningAsset()
        self.data.loadAsset('../data/IrisDataAll.csv')
        self.control = Control(dataset=self.data)

        # widget - qframe, presenting image of results
        self.frame = MapFrame(self, path)
        self.ui.horizontalLayoutMain.addWidget(self.frame, 1)

        ######################
        # signals added here #
        # ################## #

        # somDataChanged
        self.control.somMapChanged.connect(self.frame.change_som_map_data)

        # pause
        self.ui.pauseButton.clicked.connect(self.frame.pause)
        # simulation speed
        self.ui.simSpeedHSlider.valueChanged.connect(self.frame.change_sim_speed)
        # som map scale
        self.ui.somMapScaleHSlide.valueChanged.connect(self.frame.change_som_map_scaling)
        # gamma value changed
        self.control.gammaChanged.connect(self.ui.gammaLEdit.setText)
        # sigma value changed
        self.control.sigmaChanged.connect(self.ui.sigmaLEdit.setText)

        # steps buttons
        # connect buttons to handler function
        self.ui.oneStepButton.clicked.connect(lambda: self.frame.n_steps_button_handler(1))
        self.ui.tenStepsButton.clicked.connect(lambda: self.frame.n_steps_button_handler(10))
        self.ui.oneHundredStepsButton.clicked.connect(lambda: self.frame.n_steps_button_handler(100))
        self.ui.twoHundredStepsButton.clicked.connect(lambda: self.frame.n_steps_button_handler(200))
        self.ui.fiveHundredStepsButton.clicked.connect(lambda: self.frame.n_steps_button_handler(500))
        # frame signal ---> control handler (the same for every button)
        self.frame.nStepsLearnCall.connect(self.control.i_steps_forward_handler)

        # alfa
        self.alfaValidator = ParametersValidator(1000,5000,self.ui.alfaLEdit)
        self.ui.alfaLEdit.setValidator(self.alfaValidator)
        self.alfaValidator.colorChanged.connect(self.ui.alfaLEdit.color_change)
        self.ui.alfaLEdit.lineEditSignal.connect(self.control.change_alfa)

        # gamma 0
        self.gammaZeroValidator = ParametersValidator(1, 50, self.ui.gammaZeroLEdit)
        self.ui.gammaZeroLEdit.setValidator(self.gammaZeroValidator)
        self.gammaZeroValidator.colorChanged.connect(self.ui.gammaZeroLEdit.color_change)
        self.ui.gammaZeroLEdit.lineEditSignal.connect(self.control.change_gamma_zero)

        # sigma 0
        self.sigmaZeroValidator = ParametersValidator(1, 50, self.ui.sigmaZeroLEdit)
        self.ui.sigmaZeroLEdit.setValidator(self.sigmaZeroValidator)
        self.sigmaZeroValidator.colorChanged.connect(self.ui.sigmaZeroLEdit.color_change)
        self.ui.sigmaZeroLEdit.lineEditSignal.connect(self.control.change_sigma_zero)

        # network size
        self.networkSizeValidator = ParametersValidator(4, 400, self.ui.networkSizeLEdit)
        self.ui.networkSizeLEdit.setValidator(self.networkSizeValidator)
        self.networkSizeValidator.colorChanged.connect(self.ui.networkSizeLEdit.color_change)
        self.ui.networkSizeLEdit.lineEditSignal.connect(self.control.change_network_size)

        # self.ui.sigmaZeroLEdit.editingFinished.connect(self.change_alfa)
        self.frame.iterationChange.connect(self.ui.iterationNumberLEdit.setText)
        self.frame.oneStepLearnCall.connect(self.control.one_step_learn_call_handler)
        # QtCore.QObject.connect(self.simSpeedHSlider, QtCore.SIGNAL(_fromUtf8("valueChanged(int)")), QSom.update)

        # start button
        self.ui.startButton.clicked.connect(self.frame.start)

        print(self.frame.geometry())
        self.show()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    path = '../data/IrisDataAll.csv'
    somapp = StartQT4(path)

    somapp.show()
    sys.exit(app.exec_())

