from PyQt5 import QtWidgets, uic
import sys

application = QtWidgets.QApplication([])


def main():
    # specify the location of our .ui file
    window = uic.loadUi("MainWindow.ui")
    # show the window
    window.show()
    # kill the process, when the application is closed
    sys.exit(application.exec_())
