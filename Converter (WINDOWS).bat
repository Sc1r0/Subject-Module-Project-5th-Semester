@echo OFF

TITLE ".UI to .PY converter"

pyuic5 -x MainWindow.ui -o test.py

echo The conversion is done. You may close this window...

pause
