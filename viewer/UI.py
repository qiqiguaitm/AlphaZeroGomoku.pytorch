import threading
from . import NativeUI
import sys
from PyQt5.QtWidgets import QApplication


class UI(threading.Thread):
    def __init__(self,pressaction,chessboardinfo,human_role=1,sizeunit=50):
        threading.Thread.__init__(self)
        self.ui = None
        self.app = None

        self.chessboardinfo = chessboardinfo
        self.sizeunit = sizeunit
        self.pressaction = pressaction
        self.human_role = human_role
    
    def run(self):
        print('Init UI...')
        self.app = QApplication(sys.argv)
        self.UI = NativeUI.NativeUI(pressaction=self.pressaction, chessboardinfo=self.chessboardinfo,role=self.human_role)
        print('app exec...')
        self.app.exec_()
        print('app exit')


    def setchessboard(self,chessboardinfo):
        return self.UI.setchessboard(chessboardinfo)
    
    def gameend(self,role):
        self.UI.gameend(winner=role)