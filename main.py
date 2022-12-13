# GUI
import numpy as np

from MainWindow import window
from excel_sheet_data import X

X = np.array(X)
B1 = X[:, 0]
B2 = X[:, 1]

if __name__ == '__main__':
    # run our GUI
    window()
