from PyQt5 import QtWidgets, uic, QtGui
import sys


def main():
    # Create our application window
    application = QtWidgets.QApplication([])
    # set the icon
    application.setWindowIcon(QtGui.QIcon("icon.svg"))
    # set the style
    application.setStyle("fusion")

    # specify the location of our .ui file
    window = uic.loadUi("MainWindow.ui")
    # show the window
    window.show()
    # We used sys.exit(app.exec()) instead of using app.exec() directly to send the correct status code
    # to the parent process or the calling process, once it closes.
    # If we used app.exec() instead, the application would return zero, which means success,
    # and this will happen even if the application crashed.
    sys.exit(application.exec())
