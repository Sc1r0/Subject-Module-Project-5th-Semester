# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

# import sys (used for PyQt5) and webbrowser (used to open our GitHub Project Repository when the button 'Open' is
# pressed in our "About" MessageBox.
import sys
import webbrowser

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QMessageBox, QListWidgetItem

# get data from X_test to show in List view
from excel_sheet_data import X, X_test, X_train, y, y_test, y_train

# get our KNN function
from KNN import KNNFromScratch as kNN


# Our GUI Class
class Ui_MainWindow(object):
    def setupUI(self, MainWindow):
        # Main Window set up
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(841, 592)
        MainWindow.setFixedSize(841, 592)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        MainWindow.setFont(font)
        MainWindow.setAcceptDrops(True)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setWindowIcon(QtGui.QIcon("icon.svg"))

        # Create Widget within the Window Frame
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(True)
        self.centralwidget.setObjectName("centralwidget")

        # Add Instruction text to the window
        self.Instructions = QtWidgets.QLabel(self.centralwidget)
        self.Instructions.setGeometry(QtCore.QRect(20, 0, 561, 121))
        self.Instructions.setWordWrap(True)
        self.Instructions.setObjectName("Instructions")

        # Add Group Box containing the Map Graphic
        self.MapGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.MapGroupBox.setGeometry(QtCore.QRect(20, 120, 571, 281))
        self.MapGroupBox.setToolTip("")
        self.MapGroupBox.setAutoFillBackground(False)
        self.MapGroupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.MapGroupBox.setObjectName("MapGroupBox")

        # Add the Map Graphic
        self.MapGraphic = QtWidgets.QGraphicsView(self.MapGroupBox)
        self.MapGraphic.setGeometry(QtCore.QRect(20, 20, 531, 251))
        self.MapGraphic.viewport().setProperty("cursor", QtGui.QCursor(QtCore.Qt.ForbiddenCursor))
        self.MapGraphic.setObjectName("MapGraphic")

        # Add Group Box for the RSSI related UI components
        self.RSSIValuesGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.RSSIValuesGroupBox.setGeometry(QtCore.QRect(20, 410, 321, 131))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.RSSIValuesGroupBox.setFont(font)
        self.RSSIValuesGroupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.RSSIValuesGroupBox.setObjectName("RSSIValuesGroupBox")

        # Add Input Fields for all 4 beacons
        # accept only integers - also makes it possible to enter positive values: we don't want this!
        only_integers = QRegExp("[-]?[1-9]\\d{1}")

        # Beacon 1 Label:
        self.B2Label = QtWidgets.QLabel(self.RSSIValuesGroupBox)
        self.B2Label.setGeometry(QtCore.QRect(10, 30, 71, 16))
        self.B2Label.setCursor(QtGui.QCursor(QtCore.Qt.WhatsThisCursor))
        self.B2Label.setObjectName("B2Label")

        # Beacon 1 input field:
        self.B2Input = QtWidgets.QLineEdit(self.RSSIValuesGroupBox)
        self.B2Input.setGeometry(QtCore.QRect(10, 49, 61, 20))
        self.B2Input.setObjectName("B2Input")
        self.B2Input.setPlaceholderText("-39")
        self.B2Input.setValidator(QRegExpValidator(only_integers))

        # Beacon 2 Label:
        self.B3Label = QtWidgets.QLabel(self.RSSIValuesGroupBox)
        self.B3Label.setGeometry(QtCore.QRect(90, 30, 71, 16))
        self.B3Label.setCursor(QtGui.QCursor(QtCore.Qt.WhatsThisCursor))
        self.B3Label.setObjectName("B3Label")

        # Beacon 2 Input field:
        self.B3Input = QtWidgets.QLineEdit(self.RSSIValuesGroupBox)
        self.B3Input.setGeometry(QtCore.QRect(90, 49, 61, 20))
        self.B3Input.setObjectName("B3Input")
        self.B3Input.setPlaceholderText("-42")
        self.B3Input.setValidator(QRegExpValidator(only_integers))

        # Beacon 3 Label:
        self.B4Label = QtWidgets.QLabel(self.RSSIValuesGroupBox)
        self.B4Label.setGeometry(QtCore.QRect(170, 30, 71, 16))
        self.B4Label.setCursor(QtGui.QCursor(QtCore.Qt.WhatsThisCursor))
        self.B4Label.setObjectName("B4Label")

        # Beacon 3 Input field:
        self.B4Input = QtWidgets.QLineEdit(self.RSSIValuesGroupBox)
        self.B4Input.setGeometry(QtCore.QRect(170, 49, 61, 20))
        self.B4Input.setObjectName("B4Input")
        self.B4Input.setPlaceholderText("-46")
        self.B4Input.setValidator(QRegExpValidator(only_integers))

        # Beacon 4 Label:
        self.B5Label = QtWidgets.QLabel(self.RSSIValuesGroupBox)
        self.B5Label.setGeometry(QtCore.QRect(250, 30, 71, 16))
        self.B5Label.setCursor(QtGui.QCursor(QtCore.Qt.WhatsThisCursor))
        self.B5Label.setObjectName("B5Label")

        # Beacon 4 Input field:
        self.B5Input = QtWidgets.QLineEdit(self.RSSIValuesGroupBox)
        self.B5Input.setGeometry(QtCore.QRect(250, 49, 61, 20))
        self.B5Input.setObjectName("B5Input")
        self.B5Input.setPlaceholderText("-36")
        self.B5Input.setValidator(QRegExpValidator(only_integers))

        # Button to initiate calculation of the users position:
        self.FindMeBtn = QtWidgets.QPushButton(self.RSSIValuesGroupBox)
        self.FindMeBtn.setGeometry(QtCore.QRect(10, 80, 301, 41))
        self.FindMeBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.FindMeBtn.setObjectName("FindMeBtn")

        # Group Box containing UI components related to the result of the User Position calculation
        self.YourPositionGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.YourPositionGroupBox.setGeometry(QtCore.QRect(350, 410, 241, 61))
        self.YourPositionGroupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.YourPositionGroupBox.setObjectName("YourPositionGroupBox")

        # Estimated Position label:
        self.EstimatedPosition_text = QtWidgets.QLabel(self.YourPositionGroupBox)
        self.EstimatedPosition_text.setGeometry(QtCore.QRect(12, 20, 161, 31))
        self.EstimatedPosition_text.setObjectName("EstimatedPosition_text")

        # Estimated Position value:
        self.EstimatedPosition_value = QtWidgets.QLabel(self.YourPositionGroupBox)
        self.EstimatedPosition_value.setGeometry(QtCore.QRect(170, 20, 61, 31))
        self.EstimatedPosition_value.setAlignment(QtCore.Qt.AlignCenter)
        self.EstimatedPosition_value.setObjectName("EstimatedPosition_value")

        # GroupBox containing UI elements related to the Margin of Error calculation
        self.MarginOfErrorGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.MarginOfErrorGroupBox.setGeometry(QtCore.QRect(350, 480, 241, 61))
        self.MarginOfErrorGroupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.MarginOfErrorGroupBox.setObjectName("MarginOfErrorGroupBox")

        # Margin of Error label:
        self.MarginOfError_Label = QtWidgets.QLabel(self.MarginOfErrorGroupBox)
        self.MarginOfError_Label.setGeometry(QtCore.QRect(140, 20, 61, 31))
        self.MarginOfError_Label.setAlignment(QtCore.Qt.AlignCenter)
        self.MarginOfError_Label.setObjectName("MarginOfError_Label")

        # Margin of Error value:
        self.MarginOfError_value = QtWidgets.QLabel(self.MarginOfErrorGroupBox)
        self.MarginOfError_value.setGeometry(QtCore.QRect(12, 20, 101, 31))
        self.MarginOfError_value.setObjectName("MarginOfError_value")

        # List Widget
        self.ListWidgetGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.ListWidgetGroupBox.setGeometry(QtCore.QRect(600, 10, 221, 531))
        self.ListWidgetGroupBox.setObjectName("ListWidgetGroupBox")
        self.ListWidgetGroupBox.setToolTip("Double click on an array to pick it!")

        # Container for values in X
        self.listWidget = QtWidgets.QListWidget(self.ListWidgetGroupBox)
        self.listWidget.setGeometry(QtCore.QRect(10, 20, 201, 501))
        self.listWidget.setObjectName("listWidget")
        for item in X_test.values:
            self.listWidget.addItem(str(item))
        self.listWidget.itemDoubleClicked.connect(self.getItem)

        # Menu Bar, top of the window:
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 601, 21))
        self.menubar.setObjectName("menubar")

        # Status Bar, sub-component to Menu Bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Item "Menu" in Menu bar
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)

        # Compoments within the "Menu bar" subcomponent
        # 1) Close component
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionClose.triggered.connect(lambda: self.close())

        # 2) About component
        self.actionAbout = QtWidgets.QAction(MainWindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout.triggered.connect(lambda: self.about())

        # Add the subcomponents to the menubar
        self.menuMenu.addAction(self.actionClose)
        self.menuMenu.addSeparator()
        self.menuMenu.addAction(self.actionAbout)
        self.menubar.addAction(self.menuMenu.menuAction())

        # Fill our UI components with text and format it.
        self.retranslateUi(MainWindow)

        # Is to catch any event-driven signals, our UI may generate.
        """
        Example:
        
        We have our button "Locate Me!". If we press it, we want an event to happen. This event should be the 
        function to calculate the X,Y position, based on the input in textfield "Beacon 1", "Beacon 2", "Beacon 3" and 
        "Beacon 4".
        
        What the below function does, is to listen for said events for all sub-components in our GUI.
        
        See https://doc.qt.io/qt-5/qmetaobject.html#connectSlotsByName for more information.
        """
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        # Window Title
        MainWindow.setWindowTitle(_translate("MainWindow", "Find My Location"))

        # Set text of our Instruction label
        self.Instructions.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; "
                                                           "font-weight:600;\">Instructions:</span></p><p>Either "
                                                           "enter 4 negative values, one for each beacon or choose a "
                                                           "set of RSSI values from the list to the right. "
                                                           "</p><p>These values have to be between -10 and -70. "
                                                           "Anything but negative values will not be tolerated. </p>"
                                                           "<p>Once filled, press the &quot;Locate Me!&quot; button, "
                                                           "and your approximate location will displayed on the "
                                                           "map.</p> </body></html>"))

        # Set title of our MapGroupBox object
        self.MapGroupBox.setTitle(_translate("MainWindow", "Map"))

        # RSSI Values GroupBox object and it's subcomponents
        self.RSSIValuesGroupBox.setTitle(_translate("MainWindow", "RSSI Values"))
        # Beacon Labels and input fields
        self.B2Label.setToolTip(_translate("MainWindow", "Type in an RSSI value between -10 and -70"))
        self.B2Label.setText(_translate("MainWindow", "<html><head/><body><p>Beacon 1:</p></body></html>"))
        self.B3Label.setToolTip(_translate("MainWindow", "Type in an RSSI value between -10 and -70"))
        self.B3Label.setText(_translate("MainWindow", "<html><head/><body><p>Beacon 2:</p></body></html>"))
        self.B4Label.setToolTip(_translate("MainWindow", "Type in an RSSI value between -10 and -70"))
        self.B4Label.setText(_translate("MainWindow", "<html><head/><body><p>Beacon 3:</p></body></html>"))
        self.B5Label.setToolTip(_translate("MainWindow", "Type in an RSSI value between -10 and -70"))
        self.B5Label.setText(_translate("MainWindow", "<html><head/><body><p>Beacon 4:</p></body></html>"))
        # Set tooltip and text value for our "Locate Me!" button
        self.FindMeBtn.setToolTip(_translate("MainWindow", "Press this button to locate yourself on the map"))
        self.FindMeBtn.setText(_translate("MainWindow", "Locate Me!"))

        # List Widget
        self.ListWidgetGroupBox.setTitle(_translate("MainWindow", "Predefined RSSI Values"))

        # Estimated User Position GroupBox object and it's subcomponents
        self.YourPositionGroupBox.setTitle(_translate("MainWindow", "Your Position"))
        self.EstimatedPosition_value.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt;\">X, Y</span></p></body></html>"))
        self.EstimatedPosition_text.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600;\">Your estimated position:</span></p></body></html>"))

        # Margin of Error GroupBox object and it's subcomponents
        self.MarginOfErrorGroupBox.setTitle(_translate("MainWindow", "Margin of Error"))
        self.MarginOfError_Label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt;\">0.0 "
                                                "meters</span></p></body></html>"))
        self.MarginOfError_value.setToolTip(_translate("MainWindow", "Shows how much your position is offset by"))
        self.MarginOfError_value.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt; "
                                               "font-weight:600;\">Margin of Error:</span></p></body></html>"))

        # Menu at the top and it's subcomponents
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))

        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionClose.setStatusTip(_translate("MainWindow", "Close the application"))
        self.actionClose.setShortcut(_translate("MainWindow", "Ctrl+W"))

        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionAbout.setStatusTip(_translate("MainWindow", "About this application"))

    def close(self):
        # close the app
        sys.exit(exit())

    def about(self):
        # create messageBox
        msgbox = QMessageBox()
        # set the Icon to the left of the text shown in the message box
        msgbox.setIcon(QMessageBox.Information)
        # set buttons to be shown in the messagebox window
        msgbox.setStandardButtons(QMessageBox.Open | QMessageBox.Close)
        # set default button to be highlighted
        msgbox.setDefaultButton(QMessageBox.Close)
        # set default button to be used, if Escape button is pressed
        msgbox.setEscapeButton(QMessageBox.Close)
        # set window title
        msgbox.setWindowTitle("About This Program")
        # set messagebox text
        msgbox.setText(
            "This program is developed for the Computer Science Semester Project at Roskilde University. It is purely "
            "developed in Python, with the following libraries:\n\n"
            
            "- sys (needed for PyQt5)\n"
            "- PyQt5 (for our GUI)\n"
            "- Numpy (for data manipulation)\n"
            "- Pandas (for reading our data from the excel sheet)\n"
            "- SciKit-learn (for datasplitting and KNN with libraries)\n"
            "- MatPlotLib (for plotting data onto a graph)\n\n"
            
            "It is developed by the following students:\n\n"
            "- Mathias Albøger Wiberg\n"
            "- Mikkel Helsted Madsen\n"
            "- William Meldorf Brøgger\n\n"

            "By pressing the button 'Open' your browser will open our GitHub project page.")

        msgbox.buttonClicked.connect(self.msgButtonPress)

        # execute the message box, showing it upon pressing "about" in the menu.
        msgbox.exec_()

    def msgButtonPress(self, i):
        # open the webbrowser at the given url, if the user presses the 'Open' button
        if i.text() == 'Open':
            webbrowser.open("https://github.com/mikkmad/Subject-Module-Project-5th-Semester")

    def getItem(self):
        # get the item
        item = self.listWidget.currentItem().text()
        # remove array brackets saved in string []
        item = item.replace('[', '')
        item = item.replace(']', '')
        # split the string on each space
        item = item.split(" ")
        # save the values
        value_1 = item[0]
        value_2 = item[1]
        value_3 = item[2]
        value_4 = item[3]
        # put the values into their corresponding Beacon input field
        self.B2Input.setText(value_1)
        self.B3Input.setText(value_2)
        self.B4Input.setText(value_3)
        self.B5Input.setText(value_4)

    def performKNN(self, k, test_point):
        # instantiate our KNN model and give it a k value
        KNN = kNN(k=k)

        # fit our KNN model
        KNN.fit(X_train.values, y_train.values)

        # predict values
        our_predictions = KNN.predict(test_point, y)

        # get our ground_truth variables
        ground_truth = KNN.ground_truth()

        # calculate the margin of error
        margin_of_error = KNN.margin_of_error(ground_truth, our_predictions)

        # plot our values into our GUI
        # self.EstimatedPosition_value.setText()


def window():
    default_font = QtGui.QFont('Arial', 12)
    default_font.setPixelSize(12)
    # QtWidgets.QApplication.setStyle("fusion")
    QtWidgets.QApplication.setFont(default_font)

    app = QtWidgets.QApplication(sys.argv)
    app.setFont(default_font)

    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setFont(default_font)

    ui = Ui_MainWindow()
    ui.setupUI(MainWindow)

    MainWindow.show()
    sys.exit(app.exec_())
