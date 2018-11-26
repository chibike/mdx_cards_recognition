#!/usr/bin/env python


import sys
import thread
from PyQt5.QtWidgets import QApplication, QWidget

class Slider(object):
    def __init__(self, title="Sliders"):
        self.title = title
        self.app = None
    
    def start(self):
        self.app = QApplication(sys.argv)
        
        self.widget = QWidget()
        self.widget.setWindowTitle(self.title)

        # start app
        thread.start_new_thread(self.widget.show, ())
    
    def stop(self):
        if self.app is not None:
            sys.exit(self.app.exec_())

if __name__ == '__main__':
    
    import signal
    import time

    my_slider = Slider()
    my_slider.start()

    try:
        while True:
            time.sleep(0.5)
    except:
        pass
    finally:
        my_slider.stop()
        print ("\nterminating...")