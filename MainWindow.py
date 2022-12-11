# import sys (used for PyQt5) and webbrowser (used to open our GitHub Project Repository when the button 'Open' is
# pressed in our "About" MessageBox).
import sys
import webbrowser

# PyQt5 modules
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QRegExp, QRect, QCoreApplication, Qt
from PyQt5.QtGui import QRegExpValidator, QCursor
from PyQt5.QtWidgets import QMessageBox, QMenuBar, QMenu, QStatusBar, QAction, QLabel, QLineEdit, QRadioButton, \
    QGroupBox, QListWidget, QWidget, qApp

# diagrams & needed modules
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# KNN models
from KNNFromScratch import KNNFromScratch as KNNFS
from KNNWithLibraries import KNNWithLibraries as KNNWL

# get data from X_test to show in List view
from excel_sheet_data import X_test, y_train, X_train


# Our GUI Class
class Ui_MainWindow(object):
    def __init__(self):
        """
        Initialize all self.objects needed in this class.
        """
        self.MapGroupBox_horizontal_layout = None
        self.Figure = None
        self.Canvas = None
        self.MapWidget = None
        self.PredefinedRSSIGroupBox = None
        self.KvalueLabel = None
        self.KvalueValue = None
        self.KNNWithLibraries = None
        self.KNNFromScratch = None
        self.PredefinedRSSIValues = None
        self.KValue_KNNMethod = None
        self.actionK_nearest_distances = None
        self.actionShow_help_window = None
        self.actionBest_K_value_1_15 = None
        self.menuDiagrams = None
        self.actionAbout = None
        self.actionClose = None
        self.menuMenu = None
        self.statusbar = None
        self.menubar = None
        self.listWidget = None
        self.ListWidgetGroupBox = None
        self.MarginOfError_value = None
        self.MarginOfError_Label = None
        self.MarginOfErrorGroupBox = None
        self.EstimatedPosition_value = None
        self.EstimatedPosition_text = None
        self.YourPositionGroupBox = None
        self.FindMeBtn = None
        self.B5Input = None
        self.B5Label = None
        self.B4Input = None
        self.B4Label = None
        self.B3Input = None
        self.B3Label = None
        self.centralwidget = None
        self.Instructions = None
        self.MapGroupBox = None
        self.B2Label = None
        self.B2Input = None
        self.RSSIValuesGroupBox = None

    def setupUI(self, MainWindow):
        """
        Set up barebone objects needed in our GUI.

        Objects used:
            * QWidget -- For the Widget window inside of our Application.
            * QGroupBox -- For grouping certain elements together in the UI, i.e. the K-value and KNN method.
            * QLabel -- Adding labels for our input fields, listwidget, buttons and the like.
            * QLineEdit -- Input fields, for taking user input.
            * QPushButton -- For the 'Locate Me!' button.
            * QListWidget -- For listing the x_test RSSI values.
            * QRadioButton -- For selecting which KNN method the user would like to use.
            * QStatusBar -- For adding the MenuBar.
            * QMenu -- For adding items to our Menubar.
            * QAction -- For adding sub-items to the Menubar.
            * ConnectSlotsByName -- To listen for events in the GUI, i.e. buttons being clicked.

        Also, the difference between:
            * ___.triggered.connect()
            * ___.clicked.connect()
        is rather simple: 'triggered.connect()' is used mostly for items that also has a shortcut attached to it.
        Such as a menu item, which can be opened by either pressing a shortcut or by clicking it with a mouse.
        Whereas, 'clicked.connect()' only works for items, that can't be 'triggered' by a shortcut, i.e. buttons.
        """
        # Main Window set up
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(841, 592)
        MainWindow.setFixedSize(841, 592)
        font = QtGui.QFont()  # create font object
        font.setFamily("Arial")  # set font family
        font.setWeight(50)  # set font weight
        font.setStyleStrategy(QtGui.QFont.PreferAntialias)
        MainWindow.setFont(font)  # set the font for our main window
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setWindowIcon(QtGui.QIcon("icon.svg"))  # add application icon

        # Menu Items
        self.actionClose = QAction("&Exit")
        self.actionClose.setObjectName(u"actionClose")
        self.actionClose.triggered.connect(qApp.quit)

        self.actionAbout = QAction(MainWindow)
        self.actionAbout.setObjectName(u"actionAbout")
        self.actionAbout.triggered.connect(lambda: self.about())

        # FIXME: Add a proper method to the triggered.connect() function for Best_K_value diagram.
        self.actionBest_K_value_1_15 = QAction(MainWindow)
        self.actionBest_K_value_1_15.setObjectName(u"actionBest_K_value_1_15")
        self.actionBest_K_value_1_15.triggered.connect(lambda: print("best K value (hardcoded): ", 6))

        # FIXME: Add a proper method to the triggered.connect() function for k_nearest_distances diagram.
        self.actionK_nearest_distances = QAction(MainWindow)
        self.actionK_nearest_distances.setObjectName(u"actionK_nearest_distances")
        self.actionK_nearest_distances.triggered.connect(lambda: print("K-nearest distances: UNABLE TO CALCULATE"))

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

        # Create layout for MapGroupBox
        self.MapGroupBox_horizontal_layout = QtWidgets.QHBoxLayout(self.MapGroupBox)
        self.MapGroupBox_horizontal_layout.setObjectName("MapGroupBox_layout")

        # Add the Map Widget
        self.MapWidget = QWidget(self.MapGroupBox)
        self.MapWidget.setObjectName(u"MapWidget")
        self.MapWidget.setGeometry(10, 20, 551, 251)

        # Create Canvas for Map Widget
        self.Figure = plt.figure()
        self.Canvas = FigureCanvas(self.Figure)

        # Add Canvas to Map Widget
        self.MapGroupBox_horizontal_layout.addWidget(self.Canvas)

        # Add Group Box for the RSSI related UI components
        self.RSSIValuesGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.RSSIValuesGroupBox.setGeometry(QRect(20, 410, 321, 131))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.RSSIValuesGroupBox.setFont(font)
        self.RSSIValuesGroupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.RSSIValuesGroupBox.setObjectName("RSSIValuesGroupBox")

        # Add Input Fields for all 4 beacons
        # accept only integers
        # FIXME: also makes it possible to enter positive values: we don't want this!
        only_integers = QRegExp("[1-9]\\d{1}")
        only_integers_negative = QRegExp("[-]?[1-9]\\d{1}")

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
        self.B2Input.setValidator(QRegExpValidator(only_integers_negative))

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
        self.B3Input.setValidator(QRegExpValidator(only_integers_negative))

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
        self.B4Input.setValidator(QRegExpValidator(only_integers_negative))

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
        self.B5Input.setValidator(QRegExpValidator(only_integers_negative))

        # Button to initiate calculation of the users position:
        self.FindMeBtn = QtWidgets.QPushButton(self.RSSIValuesGroupBox)
        self.FindMeBtn.setGeometry(QtCore.QRect(10, 80, 301, 41))
        self.FindMeBtn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.FindMeBtn.setObjectName("FindMeBtn")
        self.FindMeBtn.clicked.connect(lambda: self.checkKNNMethod())

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
        self.EstimatedPosition_value.setGeometry(QtCore.QRect(155, 28, 70, 30))
        self.EstimatedPosition_value.setAlignment(QtCore.Qt.AlignLeft)
        self.EstimatedPosition_value.setObjectName("EstimatedPosition_value")

        # GroupBox containing UI elements related to the Margin of Error calculation
        self.MarginOfErrorGroupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.MarginOfErrorGroupBox.setGeometry(QtCore.QRect(350, 480, 241, 61))
        self.MarginOfErrorGroupBox.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.MarginOfErrorGroupBox.setObjectName("MarginOfErrorGroupBox")

        # Margin of Error label:
        self.MarginOfError_Label = QtWidgets.QLabel(self.MarginOfErrorGroupBox)
        self.MarginOfError_Label.setGeometry(QtCore.QRect(12, 20, 101, 31))
        self.MarginOfError_Label.setObjectName("MarginOfError_Label")

        # Margin of Error value:
        self.MarginOfError_value = QtWidgets.QLabel(self.MarginOfErrorGroupBox)
        self.MarginOfError_value.setGeometry(QtCore.QRect(95, 28, 200, 31))
        self.MarginOfError_value.setAlignment(QtCore.Qt.AlignLeft)
        self.MarginOfError_value.setObjectName("MarginOfError_value")

        # Predefined RSSI values
        self.PredefinedRSSIGroupBox = QGroupBox(self.centralwidget)
        self.PredefinedRSSIGroupBox.setObjectName(u"PredefinedRSSIGroupBox")
        self.PredefinedRSSIGroupBox.setGeometry(QRect(600, 10, 221, 401))
        self.PredefinedRSSIGroupBox.setToolTip("Double click on an array to pick it!")

        self.PredefinedRSSIValues = QListWidget(self.PredefinedRSSIGroupBox)
        self.PredefinedRSSIValues.setGeometry(QRect(10, 20, 201, 371))
        self.PredefinedRSSIValues.setObjectName(u"PredefinedRSSIValues")
        for item in X_test.values:
            self.PredefinedRSSIValues.addItem(str(item))
        self.PredefinedRSSIValues.itemDoubleClicked.connect(self.getItem)

        self.KValue_KNNMethod = QGroupBox(self.centralwidget)
        self.KValue_KNNMethod.setObjectName(u"KValue_KNNMethod")
        self.KValue_KNNMethod.setGeometry(QRect(599, 420, 221, 121))

        self.KNNFromScratch = QRadioButton(self.KValue_KNNMethod)
        self.KNNFromScratch.setObjectName(u"KNNFromScratch")
        self.KNNFromScratch.setGeometry(QRect(10, 70, 200, 17))
        self.KNNFromScratch.setChecked(True)

        self.KNNWithLibraries = QRadioButton(self.KValue_KNNMethod)
        self.KNNWithLibraries.setObjectName(u"KNNWithLibraries")
        self.KNNWithLibraries.setGeometry(QRect(10, 90, 200, 17))

        self.KvalueValue = QLineEdit(self.KValue_KNNMethod)
        self.KvalueValue.setObjectName(u"KvalueValue")
        self.KvalueValue.setGeometry(QRect(10, 40, 201, 20))
        self.KvalueValue.setValidator(QRegExpValidator(only_integers))

        self.KvalueLabel = QLabel(self.KValue_KNNMethod)
        self.KvalueLabel.setObjectName(u"KvalueLabel")
        self.KvalueLabel.setGeometry(QRect(10, 20, 191, 16))
        self.KvalueLabel.setCursor(QCursor(QtCore.Qt.WhatsThisCursor))

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 841, 21))

        self.menuMenu = QMenu(self.menubar)
        self.menuMenu.setObjectName(u"menuMenu")

        self.menuDiagrams = QMenu(self.menubar)
        self.menuDiagrams.setObjectName(u"menuDiagrams")
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menuMenu.addAction(self.actionClose)
        self.menuMenu.addSeparator()
        self.menuMenu.addAction(self.actionAbout)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.menuDiagrams.addAction(self.actionBest_K_value_1_15)
        self.menuDiagrams.addSeparator()
        self.menuDiagrams.addAction(self.actionK_nearest_distances)
        self.menubar.addAction(self.menuDiagrams.menuAction())

        # Fill our UI components with text and format it.
        self.retranslateUi(MainWindow)

        # Is to catch any event-driven signals, our UI may generate.
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        """
        Example:
        
        We have our button "Locate Me!". If we press it, we want an event to happen. This event should be the 
        function to calculate the X,Y position, based on the input in textfield "Beacon 1", "Beacon 2", "Beacon 3" and 
        "Beacon 4".
        
        What the below function does, is to listen for said events for all sub-components in our GUI.
        
        See https://doc.qt.io/qt-5/qmetaobject.html#connectSlotsByName for more information.
        """

    def retranslateUi(self, MainWindow):
        """
        Add all necessary information to our GUI objects from MainWindow.setupUI(), like:\n
        - Setting Window Title.\n
        - Setting tooltips for different labels.\n
        - Giving the GroupBoxes titles.\n
        - Adding our Menubar and it's subcomponents, such as "Menu" and "Diagrams".
        """
        _translate = QtCore.QCoreApplication.translate
        # Window Title
        MainWindow.setWindowTitle(_translate("MainWindow", "Find My Location"))

        # Set text of our Instruction label
        self.Instructions.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; "
                                                           "font-weight:600;\">Instructions:</span></p><p>Enter "
                                                           "4 negative values or choose a set of RSSI values from the "
                                                           "list to the right.</p>"
                                                           "<p>These values have to be between -10 and -70. Anything "
                                                           "but negative values will not be tolerated.</p>"
                                                           "<p>Once filled, press the &quot;Locate Me!&quot; button "
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
        self.PredefinedRSSIGroupBox.setTitle(_translate("MainWindow", "Predefined RSSI Values"))

        # Estimated User Position GroupBox object and it's subcomponents
        self.YourPositionGroupBox.setTitle(_translate("MainWindow", "Your Position [X, Y]"))
        self.EstimatedPosition_value.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" "
                                                                      "font-size:9pt;\">[X, "
                                                                      "Y]</span></p></body></html>"))
        self.EstimatedPosition_text.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" "
                                                                     "font-size:9pt; font-weight:600;\">Your estimated "
                                                                     "position:</span></p></body></html>"))

        # Margin of Error GroupBox object and it's subcomponents
        self.MarginOfErrorGroupBox.setTitle(_translate("MainWindow", "Margin of Error"))
        self.MarginOfError_value.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" "
                                                              "font-size:9pt;\">0.0 "
                                                "meters</span></p></body></html>"))
        self.MarginOfError_Label.setToolTip(_translate("MainWindow", "Shows how much your position is offset by"))
        self.MarginOfError_Label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt; "
                                               "font-weight:600;\">Error margin:</span></p></body></html>"))

        # K-value and KNN method
        self.KValue_KNNMethod.setTitle(QCoreApplication.translate("MainWindow", u"K-value and KNN method"))
        self.KNNFromScratch.setText(QCoreApplication.translate("MainWindow", u"KNN from scratch"))
        self.KNNWithLibraries.setText(QCoreApplication.translate("MainWindow", u"KNN with sci-kit libraries"))
        self.KvalueValue.setPlaceholderText(QCoreApplication.translate("MainWindow", u"K-value (1-15)"))
        self.KvalueLabel.setToolTip(QCoreApplication.translate("MainWindow", u"Choose a K-value. For picking the "
                                                                             u"optimized one, run the Best K-value "
                                                                             u"diagram, in \"Diagrams -> Best K-value\""))
        self.KvalueLabel.setText(QCoreApplication.translate("MainWindow", u"<html><head/><body><p><span style=\""
                                                                          u"font-weight:600;\">K-value:</span>"))

        # Menu at the top and it's subcomponents
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.menuDiagrams.setTitle(_translate("MainWindow", "Diagrams"))

        self.actionClose.setText(_translate("MainWindow", "Quit"))
        self.actionClose.setStatusTip(_translate("MainWindow", "Quit the application"))
        self.actionClose.setShortcut(_translate("MainWindow", "Ctrl+W"))

        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionAbout.setStatusTip(_translate("MainWindow", "About this application and its authors"))

        self.actionK_nearest_distances.setText(_translate("MainWindow", "K-nearest distance"))
        self.actionK_nearest_distances.setStatusTip(_translate("MainWindow", "Show a diagram of k-nearest distance "
                                                                             "values"))

        self.actionBest_K_value_1_15.setText(_translate("MainWindow", "Best K-Value (1-15)"))
        self.actionBest_K_value_1_15.setStatusTip(_translate("MainWindow", "Shows a diagram of the best k-values "
                                                                           "from 1-15"))

    def about(self):
        """
        A function to show information about the application and it's developers.
        :return: A messsagebox containing information about libraries used and for what, the developers behind the code
         and a link to its GitHub page.
        """
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
        msgbox.exec()

    def msgButtonPress(self, i):
        """
        A method to check which button is pressed. "buttonClicked" from PyQt5 returns a string, with the name of the
        button clicked. If i = 'Open', then open the users browser and redirect them to the Code's GitHub page.
        :param i: The name of the button pressed.
        """
        # open the webbrowser at the given url, if the user presses the 'Open' button
        if i.text() == 'Open':
            webbrowser.open("https://github.com/mikkmad/Subject-Module-Project-5th-Semester")

    def getItem(self):
        """
        A method to fetch the items of the currently selected* array from the predefined RSSI values. When fetched,
        it updates the text for the Beacon 1-4 text fields.
        """
        # get the item
        item = self.PredefinedRSSIValues.currentItem().text()

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

    def performKNNFromScratch(self, test_point):
        """
        A method to run the KNNFromScratch algorithm - see KNNFromScratch.py. Shows a QMessageBox with an error,
        if the passed test_point variable is empty or partly empty.
        :param k: the amount of nearest neighbors, one wishes to look at.
        :param test_point: the four values of the Beacon 1-4 text fields.
        """
        if not test_point or not self.KvalueValue.text():
            self.error_message_box("MISSING VALUES", "You are missing either: "
                                                     "\n- one or more RSSI values"
                                                     "\n- the K-value")

        else:
            # instantiate our KNN model and give it a k value
            KNN = KNNFS(int(self.KvalueValue.text()))

            # fit our KNN model
            KNN.fit(X_train.values, y_train.values)

            # predict values
            prediction = KNN.predict(test_point, y_train)

            # calculate the margin of error
            # use KNN.evaluate_knn(y_test, prediction) function
            knn_eval = KNN.evaluate_knn(prediction)

            # set the value of below textfield to the result of above function
            self.MarginOfError_value.setText(str(knn_eval) + " meters")

            # plot our values into our GUI
            self.EstimatedPosition_value.setText(str(prediction))

            self.showUserPosition()

    # FIXME: How do we calculate the Margin of Error here?
    def performKNNWithLibraries(self, test_point):
        """
        A method to run the KNNWithLibraries algorithm - see KNNWithLibraries.py. Shows a QMessageBox with an error,
        if the passed test_point variable is empty or partly empty.
        :param k: the amount of nearest neighbors, one wishes to look at.
        :param test_point: the four values of the Beacon 1-4 text fields.
        """

        if not test_point or not self.KvalueValue.text():
            self.error_message_box("MISSING VALUES", "You are missing either: "
                                                     "\n- one or more RSSI values"
                                                     "\n- the K-value")

        else:
            # instantiate our KNN model and give it a k value
            KNN = KNNWL(int(self.KvalueValue.text()))

            # fit our KNN model
            KNN.fit()

            # predict values
            prediction = KNN.predict(test_point)

            # set the value of below textfield to the result of above function
            self.MarginOfError_value.setText("Calculation failed!")

            # plot our values into our GUI
            self.EstimatedPosition_value.setText(str(prediction))

    def getTestPoint(self):
        """
        A method to get the values from the Beacon 1-4 text fields.
        :return: If all text fields have values return an array containing these values. Otherwise, return None.
        """
        if self.B2Input.text() and self.B3Input.text() and self.B4Input.text() and self.B5Input.text():
            # save values in a list
            beacon_values = [int(self.B2Input.text()), int(self.B3Input.text()),
                             int(self.B4Input.text()), int(self.B5Input.text())]
            # return the list
            return beacon_values
        # otherwise, return None
        return None

    def checkKNNMethod(self):
        """
        A method to check which QRadioButton is checked in our GUI. Depending on which QRadioButton is checked,
        perform either "KNN From Scratch" method or the "KNN With Libraries" method.
        """
        # check if the KNNFromScratch radio button is checked...
        if self.KNNFromScratch.isChecked():
            self.performKNNFromScratch(self.getTestPoint())
        # else check if the KNNWithLibraries is checked...
        elif self.KNNWithLibraries.isChecked():
            self.performKNNWithLibraries(self.getTestPoint())
        # if neither is checked, show a QMessageBox containing an error
        else:
            self.error_message_box("Something went wrong...", "Something went terribly wrong and we do not know what!")

    def showUserPosition(self):
        """
        This method was supposed to show the users location on a map, in the "Map" section of our GUI. However,
        during development, we ran into multiple problems:

        * How do we make sure, that the scale of the graph matches the scale of the image? In other words, how are we sure that 1x is the same length as 1x in the graph?
        * How do we align the image of our service area with the graph? Also, how do we make sure that the predicted X,Y coordinate is shown in the correct spot on the map?

        This proved difficult to fix, and therefore we've abandoned it for now. We chose to spend our time and
        energy on more important features.

        Below is the code for the 'prototype'. It does plot the X,Y coordinates onto the Graph canvas,
        but it is always directly in the middle, and not placed on an image in the correct spot.
        """
        # save the position values
        position = self.EstimatedPosition_value.text()
        # remove unnecessary characters from the string
        position = position.replace('[', '')
        position = position.replace(']', '')
        # split the string on each comma (turns it into a list)
        position = position.split(",")
        # save the X and Y coordinates
        posX = [position[0]]
        posY = [position[1]]
        # clear the canvas
        self.Figure.clear()
        # set max size of the graph window
        #plt.xlim(0, 15)
        #plt.ylim(0, 15)
        # draw a grid in the graph window
        plt.grid()
        # plot the X,Y coordinate onto the canvas
        plt.plot(posX, posY, marker='o', markersize=15)
        # refresh the canvas
        self.Canvas.draw()

    def error_message_box(self, title, errormessage):
        """
        A function that creates a QMessageBox containing a custom title and message content.
        :param title: The title of the QMessageBox window.
        :param errormessage: The message content of the QMessageBox window.
        """
        # create messageBox
        msgbox = QMessageBox()
        # set the Icon to the left of the text shown in the message box
        msgbox.setIcon(QMessageBox.Critical)
        # set buttons to be shown in the messagebox window
        msgbox.setStandardButtons(QMessageBox.Close)
        # set default button to be used, if Escape button is pressed
        msgbox.setEscapeButton(QMessageBox.Close)
        # set window title
        msgbox.setWindowTitle(title)
        # set messagebox text
        msgbox.setText(errormessage)
        # execute the message box, showing it upon pressing "about" in the menu.
        msgbox.exec()

    # function to show best k-value
    def k_value_diagram(self):
        """
        A method to create a diagram, showing a graph containing the Margin of Error result from k-value 1-15,
        on a plot.
        """
        pass

    # function to show the k nearest distances
    def nearest_distances_diagram(self):
        """
        A method to create a diagram, showing a graph containing the k-nearest distances on a plot.
        """
        pass


class BestKValue(FigureCanvas):
    def __init__(self, parent):
        fig, self.ax = plt.subplots(figsize=(10, 10), dpi=200)
        super().__init__(fig)
        self.setParent(parent)

        # MatPlotLib
        margin_of_error = []
        for i in range(1, 15):
            KNN = KNNFS(k=i)
            KNN.fit(X_train.values, y_train.values)
            our_predictions = KNN.predict(X_test.values, y_train)
            margin_of_error.append(KNN.evaluate_knn(our_predictions))

        self.ax.plot(range(1, 15), margin_of_error)
        self.ax.set(xlabel="k-value", ylabel="margin of error (in meters)", title="Margin of Error by K-value")
        self.ax.grid()


class BestKValueWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(800, 640)
        chart = BestKValue(self)

    def showDiagram(self):
        self.diagram = BestKValueWindow()
        self.diagram.show()


def window():
    """
    The method to create the GUI object and add final touches to it, such as the font family, font size, GUI style (
    windows, windowsxp or fusion).
    :return: A GUI.
    """
    # set default font
    default_font = QtGui.QFont('Arial', 12)
    default_font.setPixelSize(12)
    QtWidgets.QApplication.setFont(default_font)    # set the font for our application
    # create our PyQt5 application
    app = QtWidgets.QApplication(sys.argv)
    app.setFont(default_font)
    # create the QtWidget MainWindow inside our PyQt5 application window
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.setFont(default_font)
    # create our UI Object from our Ui_MainWindow class
    ui = Ui_MainWindow()
    # run the set-up method, creating and placing all our GUI components
    ui.setupUI(MainWindow)
    # show our UI
    MainWindow.show()
    # closes the app properly, when the user exits it
    sys.exit(app.exec_())
