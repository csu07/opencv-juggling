import sys
import time

import cv2.cv2 as cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from ui import OpenCamUi


class Camera(QtCore.QThread):
    raw_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cam = cv2.VideoCapture(0)
        # current_wb = self.cam.get(cv2.CAP_PROP_AUTO_WB)
        # print(current_wb)
        self.cam.set(cv2.CAP_PROP_AUTO_WB, cv2.CAP_PROP_AUTO_WB)
        if self.cam is None or not self.cam.isOpened():
            self.connect = False
            self.running = False
        else:
            self.connect = True
            self.running = False

    def run(self):
        while self.running and self.connect:
            ret, img = self.cam.read()
            if ret:

                # self.get_contours(img)
                self.get_face(img)
            else:
                print("Warning!!!")
                self.connect = False

    def get_contours(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        mid_gray = cv2.medianBlur(gray, 3)

        ret, thresh = cv2.threshold(mid_gray, 127, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # c = sorted(contours, key=cv2.contourArea, reverse=True)[-1]  # 按面积排序获取目标轮廓，按实际的图来
        # cir = cv2.arcLength(c, True)  # 获取闭合轮廓的周长
        # c = cv2.approxPolyDP(c, cir / 100, True)  # 获取近似轮廓
        img_ = cv2.drawContours(img, contours, -1, (0, 255, 255), 2)  # 画出轮廓
        self.raw_data.emit(img_)

    def get_face(self, img):
        # 载入检测模型
        fc = cv2.CascadeClassifier('./face.xml')
        #
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = fc.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for face in faces:
            (x, y, w, h) = face
            # 在原图上绘制矩形
            face = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
            self.raw_data.emit(face)

    def open(self):
        if self.connect:
            self.running = True

    def stop(self):
        if self.connect:
            self.running = False

    def close(self):
        if self.connect:
            self.running = False
            time.sleep(1)
            self.cam.release()


class MainWindow(QtWidgets.QMainWindow, OpenCamUi):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setup_ui(self)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.viewData.setScaledContents(True)

        self.view_x = self.view.horizontalScrollBar()
        self.view_y = self.view.verticalScrollBar()
        self.view.installEventFilter(self)
        self.last_move_x = 0
        self.last_move_y = 0

        self.frame_num = 0

        self.cam = Camera()
        if self.cam.connect:
            self.debug_bur('Connection!!!')
            self.cam.raw_data.connect(self.get_raw)
        else:
            self.debug_bur('Disconnection!!!')

        self.camBtn_open.clicked.connect(self.open_cam)
        self.camBtn_stop.clicked.connect(self.stop_cam)

    def get_raw(self, data):
        self.show_data(data)

    def open_cam(self):
        if self.cam.connect:
            self.cam.open()
            self.cam.start()
            self.camBtn_open.setEnabled(False)
            self.camBtn_stop.setEnabled(True)
            self.viewCbo_roi.setEnabled(True)

    def stop_cam(self):
        if self.cam.connect:
            self.cam.stop()
            self.camBtn_open.setEnabled(True)
            self.camBtn_stop.setEnabled(False)
            self.viewCbo_roi.setEnabled(False)

    def show_data(self, img):
        self.Ny, self.Nx, _ = img.shape

        q_img = QtGui.QImage(img.data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)
        self.viewData.setScaledContents(True)
        self.viewData.setPixmap(QtGui.QPixmap.fromImage(q_img))
        if self.viewCbo_roi.currentIndex() == 0:
            roi_rate = 0.5
        elif self.viewCbo_roi.currentIndex() == 1:
            roi_rate = 0.75
        elif self.viewCbo_roi.currentIndex() == 2:
            roi_rate = 1
        elif self.viewCbo_roi.currentIndex() == 3:
            roi_rate = 1.25
        elif self.viewCbo_roi.currentIndex() == 4:
            roi_rate = 1.5
        else:
            pass
        self.viewForm.setMinimumSize(self.Nx * roi_rate, self.Ny * roi_rate)
        self.viewForm.setMaximumSize(self.Nx * roi_rate, self.Ny * roi_rate)
        self.viewData.setMinimumSize(self.Nx * roi_rate, self.Ny * roi_rate)
        self.viewData.setMaximumSize(self.Nx * roi_rate, self.Ny * roi_rate)

        if self.frame_num == 0:
            self.time_start = time.time()
        if self.frame_num >= 0:
            self.frame_num += 1
            self.t_total = time.time() - self.time_start
            if self.frame_num % 100 == 0:
                self.frame_rate = float(self.frame_num) / self.t_total
                self.debug_bur('FPS: %0.3f frames/sec' % self.frame_rate)

    def event_filter(self, source, event):
        if source == self.view:
            if event.type() == QtCore.QEvent.MouseMove:
                if self.last_move_x == 0 or self.last_move_y == 0:
                    self.last_move_x = event.pos().x()
                    self.last_move_y = event.pos().y()
                distance_x = self.last_move_x - event.pos().x()
                distance_y = self.last_move_y - event.pos().y()
                self.view_x.setValue(self.view_x.value() + distance_x)
                self.view_y.setValue(self.view_y.value() + distance_y)
                self.last_move_x = event.pos().x()
                self.last_move_y = event.pos().y()
            elif event.type() == QtCore.QEvent.MouseButtonRelease:
                self.last_move_x = 0
                self.last_move_y = 0
            return QtWidgets.QWidget.event_filter(self, source, event)

    def close_event(self, event):
        if self.cam.running:
            self.cam.close()
            self.cam.terminate()
        QtWidgets.QApplication.closeAllWindows()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            if self.cam.running:
                self.cam.close()
                time.sleep(1)
                self.cam.terminate()
            QtWidgets.QApplication.closeAllWindows()

    def debug_bur(self, msg):
        self.statusBar.showMessage(str(msg), 5000)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
