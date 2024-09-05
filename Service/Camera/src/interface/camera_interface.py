# camera : <signal| image>						| >> filter

import cv2
from pypylon import pylon
from pypylon import genicam

from numpy import (
    c_, r_, array,
    zeros, float32, mgrid
)

from random import randint

from numpy.linalg import inv
from cv2 import Rodrigues
from glob import glob
import numpy as np
import pickle
from time import sleep
from threading import Thread, Event

class CameraProperties:
    def __init__(self) -> None:
        self.__intrinsic = None
        self.__extrinsic = None
        self.__matrix = None
        self.__rvec = None
        self.__tvec = None
        self.__rotation = None

    @property
    def size(self):
        raise NotImplementedError
    
    @property
    def intrinsic(self):
        """Getting wil return:
        Last stored camera intrinsic matrix.\n
        [fx, 0, 0x, 0\n
         0, fy, 0y, 0\n
         0,  0,  1, 0]
        """
        return self.__intrinsic
    
    @property
    def extrinsic(self):
        """Getting wil return:
        Last stored camera extrinsic matrix.\n
        [r11, r12, r13, tx\n
         r21, r22, r23, ty\n
         r31, r32, r33, tz\n
         0,   0,   0,   1]
        """
        return self.__extrinsic
    
    @property
    def matrix(self):
        """Getting wil return:
        Last stored camera matrix.\n
        [fx, 0, 0x\n
         0, fy, 0y\n
         0,  0,  1]\n
        """
        return self.__matrix
    

    @property
    def projection(self):
        """Getting wil return:
        Last stored projection matrix.\n
        [p12, p12, p13, p14\n
         p21, p22, p23, p24\n
         p31, p32, p33, p34]\n
        """
        try:
            return self.intrinsic @ self.extrinsic
        except ValueError:
            raise AssertionError("Projection matrix can't be calculated, verify rvecs and tvecs values")
    
    @property
    def rotation(self):
        """Getting wil return:
        Last stored rotation matrix.\n
        [r11, r12, r13,\n
         r21, r22, r23,\n
         r31, r32, r33]\n
        """
        return self.__rotation

    @property
    def rvec(self):
        """Getting wil return:
        Last stored rotation vector.\n 
        [r1, r2, r2]
        """
        return self.__rvec
    
    @property
    def tvec(self):
        """Getting wil return:
        Last stored transform vector.\n
        [t1, t2, t2]
        """
        return self.__tvec
   
    @property
    def vecs(self):
        """Getting wil return:
        Last stored Rotation and Transformation vectors
        """
        return self.__rvec, self.__tvec

    @matrix.setter
    def matrix(self, value):
        """Setting will:
        Store the camera matrix.\n
        Calculate and store the intrinsic camera matrix.\n\n

        ** Set the Rotation and Transformation vectors to calculate the Projection Matrix.
        """
        self.__matrix = value
        self.__intrinsic = c_[self.matrix, [0, 0, 0]]

    @rvec.setter
    def rvec(self, value):
        """Setting will:
        Store the rotation vector.\n
        Calculate and store the rotation matrix.\n
        """
        self.__rvec = value.reshape(3,1)
        self.__rotation = Rodrigues(self.rvec)[0]

    @tvec.setter
    def tvec(self, value):
        """Setting will:
        Store the transformation vector.\n
        Calculate and store the extrinsic camera matrix.\n
        ** Rotation matrix should be calculated before, setting the rotation vector.
        """
        self.__tvec = value.reshape(3,1)
        self.__extrinsic = r_[c_[self.rotation, self.tvec] , [[0,0,0,1]]]

    @vecs.setter
    def vecs(self, value):
        """Setting will:
        Store the rotation and transformation vectors to calculate the projection matrix.
        """
        r, t = value
        if r is not None and t is not None:
            self.rvec = r
            self.tvec = t

    def _2DP(self, xyz):
        """ 
        Convert an World Coordinate (wx,wy,wz) into an image point (px,py).\n
        (px,py) = Projection . (wx,wy,wz)"""
        uvw = self.projection @ r_[array(xyz).reshape(3,1), [[1]]]
        return array((uvw[0]/uvw[-1], uvw[1]/uvw[-1])).astype(int).reshape(1,2)[0]
    
    def _3DP(self, uv):
        """
        Convert an image point (px,py) into an World Coordinate (wx,wy,wz).\n
        XYZ = Rotation^-1 . ((Matrix^-1 . scale) - tvec)"""
        uv_homogeneous = r_[array(uv).reshape(2,1), [[1]]]  # Convert UV coordinates to homogeneous coordinates

        L = inv(self.rotation) @ inv(self.matrix) @ uv_homogeneous
        R = inv(self.rotation) @ self.tvec.reshape(3,1)

        s = 1 + R[2]/L[2]

        suv_1 = s * uv_homogeneous
        xyz_c = inv(self.matrix).dot(suv_1)
        xyz_c = xyz_c - self.tvec
        XYZ = inv(self.rotation).dot(xyz_c)
        return XYZ 

    def _2DPS(self, *n_xyz):
        """
        Convert a group of world coordinates (xw, xy, xz) into a group of image points (px,py).\n
        (px,py) = Projection . (wx,wy,wz) | for 'n' xyz """
        return np.array([ self._2DP(xyz) for xyz in n_xyz]).reshape(1,-1,2)
    
    def _3DPS(self, *n_uv):
        """
        Convert a group of image points (px,py) into a group of world coordinates (xw, xy, xz).\n
        XYZ = Rotation^-1 . ((Matrix^-1 . scale) - tvec) | for 'n' uv"""
        return np.array([ self._3DP(uv) for uv in n_uv]).reshape(1,-1,3)
    
class Camera(CameraProperties):
    def __init__(self) -> None:
        
        self.auto_update = True
        self.calibrated = False
        self.roi = None

        return super().__init__()

    def read(self):
        raise NotImplementedError
    
    def __enter__(self):
        raise NotImplementedError

    def __exit__(self):
        raise NotImplementedError

class EventPrinter(pylon.ConfigurationEventHandler):
    def OnAttach(self, camera):
        print(f'Before attaching the camera {camera}')

    def OnAttached(self, camera):
        print(f'Attached: {camera.GetDeviceInfo()}')

    def OnOpen(self, camera):
        print('Before opening')

    def OnOpened(self, camera):
        print('After Opening')

    def OnDestroy(self, camera):
        print('Before destroying')

    def OnDestroyed(self, camera):
        print('After destroying')

    def OnClosed(self, camera):
        print('Camera Closed')

    def OnDetach(self, camera):
        print('Detaching')

    def OnGrabStarted(self, camera):
        print('Grab started')
        # time.sleep(2)

class ImageEventPrinter(pylon.ImageEventHandler):
    def __init__(self):
        super().__init__()
        self.hasNewFrame = False
        self.frame = None
        self.__converter = pylon.ImageFormatConverter()
        self.__converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.__converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    
    def read(self):
        return self.hasNewFrame, self.frame

    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")
        print()

    def OnImageGrabbed(self, camera, grabResult):
        if grabResult and grabResult.GrabSucceeded():
            self.hasNewFrame = True
            self.frame = self.__converter.Convert(grabResult).GetArray()
            # cv2.imwrite('/app/a.jpg', self.frame)
        else:
            self.hasNewFrame = False
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())

class Basler(Camera):
    def __init__(self, configuration, stop_signal=None):
        super().__init__()
        self.mx, self.my = pickle.load(open('/app/config/mp_xy.pkl', 'rb'))
        self.img_event = ImageEventPrinter()
        # tl_factory = pylon.TlFactory.GetInstance()
        # self.__camera = pylon.InstantCamera()
        # self.__camera.Attach(tl_factory.CreateFirstDevice())
        self.__camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice(pylon.DeviceInfo()))
        self.__camera.RegisterConfiguration(EventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
        self.__camera.RegisterImageEventHandler(self.img_event, pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

        
        self.__camera.Open()
        # self.__camera.StreamGrabber.MaxTransferSize = 4 * 1024 * 1024
        self.__camera.EventNotification.Value = "On"
        pylon.FeaturePersistence.Load(f"/app/config/{configuration}.pfs", self.__camera.GetNodeMap(), True)
        
        self.hasNewFrame = False
        self.__frame = None

        self.stop= stop_signal

        self.__camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        # self.__trigger()
        self.thread = Thread(target=self.__trigger, daemon=True)
        self.thread.start()


    def read_raw(self):
        return self.img_event.read()
    
    def read(self):
        a, f = self.read_raw()
        if a:
            return a, cv2.remap(f, self.mx, self.my, cv2.INTER_LINEAR)
        return a, f

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.__camera.Close()
    
    def __trigger(self, qtd=0):
        print("start - capturing")
        try:
            while self.__camera.IsGrabbing() and not self.stop.is_set():
                grabResult = self.__camera.RetrieveResult(100, pylon.TimeoutHandling_ThrowException)
                # sleep(0.1)
        except genicam.GenericException as e:
            self.stop.set()
            print(e)
            print("This service will restart.")
            exit(1)
    
class OpenCV(Camera):
    CameraIndex = 0
    def __init__(self, configuration, stop_signal=None) -> None:
        super().__init__()
        self.stop = stop_signal
        self.index = configuration.pop('index', None) or OpenCV.CameraIndex
        self.__cap = cv2.VideoCapture(self.index)

        self.hasNewFrame, self.frame = False, None
        for k,v in configuration.items():
            self.__cap.set(k,v)
        if isinstance(self.index, int):
            OpenCV.CameraIndex = self.index + 1
        
        self.thread = Thread(target=self.__trigger, daemon=True)
        self.thread.start()
    
    @property
    def size(self):
        return (self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read_raw(self):
        # return self.__cap.read()
        return self.hasNewFrame, self.frame
    
    def read(self):
        # a, f = self.read_raw()
        return self.hasNewFrame, cv2.remap(self.frame, self.mx, self.my, cv2.INTER_LINEAR)
    
    def __trigger(self):
        while not self.stop.is_set():
            self.hasNewFrame, self.frame =  self.__cap.read()
            if not self.hasNewFrame and isinstance(self.index, str):
                self.__cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
    
    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.__camera.Close()
        OpenCV.CameraIndex-=1
    
    def __delattr__(self, name: str) -> None:
        self.__exit__()
        return super().__delattr__(name)
    

# class Calibration:
#     def __init__(self, device, chessboardSize = (24,17), size_of_chessboard_squares_mm = 9.5, frameSize = (1920,1080) ) -> None:
#         """Create an instance to represent the real world objects used to perform the camera calibration.

#         Args:
#             device (Camera): An capture device with CameraProperties capabilities.
#             chessboardSize (tuple, optional): Describe the rows and columns of an chessboard pattern. Defaults to (24,17).
#             size_of_chessboard_squares_mm (float, optional): Describes the size in millimeters to each square on the chessboard pattern. Defaults to 9.5.
#             frameSize (tuple, optional): Describe the size of the frames used on the calibration process. Defaults to (1920,1080).
#         """

#         self.chessboardSize = chessboardSize
#         self.frameSize = frameSize
#         self.size_of_chessboard_squares_mm = size_of_chessboard_squares_mm
        
#         objp = zeros((chessboardSize[1]*chessboardSize[0],3), float32)
#         objp[:,:2] = mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

#         self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#         self.object_points = objp * size_of_chessboard_squares_mm
#         self.camera = device
 


#     def calibrate_from_dir(self, path="./pictures/", extension="jpg"):
#         """Read all imagens from an path with specific extension (jpg, png, etc)
#         and perform the calibration with `calibrate_from_images()` method.
#         """
#         return self.calibrate_from_images([ cv2.imread(image) for image in glob(f'{path}*.{extension}')])
    
#     def calibrate_from_images(self, images):
#         """Use and dataset of images with chessboard pattern to perform an camera calibration.

#         Args:
#             images (list(MatLike, ...)): Lista de imagens.

#         Returns:
#             MatLike: (CameraMatrix)
#             MatLike: (UndistortedCameraMatrix)
#             MatLike: (DistortionCoefficient)
#             Rect:    ROI to apply `after` remove the distortion with values above to remove the visual artifacts.
#         """
        
#         object_points = []
#         image_points = []
#         for img in images:
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             ret, corners = cv2.findChessboardCorners(gray, self.chessboardSize, None)

#             if ret:

#                 object_points.append(self.object_points)
#                 corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
#                 image_points.append(corners2)
        
#         h,  w = self.frameSize[1], self.frameSize[0]
#         ret, RawCameraMatrix, Distortion, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (w,h), None, None)
#         UndistortedCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(RawCameraMatrix, Distortion, (w,h), 1, (w,h))
#         if self.camera.auto_update:
#             self.camera.matrix = UndistortedCameraMatrix
#             self.camera.distortion = Distortion
#             self.camera.roi = roi
#             self.camera.raw_matrix = RawCameraMatrix
#             self.camera.calibrated = True
#         return RawCameraMatrix, UndistortedCameraMatrix, Distortion, roi


class WorkingArea:
    def __init__(self, device,  dict_type = cv2.aruco.DICT_6X6_250, board_shape= (7, 5), square_size=25, marker_size=20, ids_offset=100):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_type)
        self.board = cv2.aruco.CharucoBoard(board_shape, square_size, marker_size, self.dictionary, np.arange(ids_offset, ids_offset+int((board_shape[0]*board_shape[1])/2),1))
        self.board.setLegacyPattern(True)
        self.camera = device

        objp = zeros((board_shape[1]*board_shape[0],3), float32)
        objp[:,:2] = mgrid[0:board_shape[0],0:board_shape[1]].T.reshape(-1,2)
        self.object_points = objp * square_size

        size_x = marker_size*board_shape[0]
        size_y = marker_size*board_shape[1]

        self.working_plane = [
            (0,0,0),
            (0, size_y,0),
            (size_x,size_y,0),
            (size_x, 0 ,0),
        ]

    @property
    def virtual_plane(self): 
        return self.camera._2DPS(*self.working_plane)
    
    @property
    def mask(self):
        return AR(self.virtual_plane, self.blank_canvas)

    @property
    def virtual_world(self):
        return AR(self.virtual_plane, self.board_image)
    
    @property
    def board_image(self):
        return self.self.board.generateImage(self.camera.size)
    
    @property
    def blank_canvas(self):
        _blank_canvas = np.zeros(self.camera.size, dtype="uint8")
        _blank_canvas.fill(255)
        return _blank_canvas
    
    def get_info_from_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, self.dictionary)
        # cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        try:
            ret, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
            # cv2.aruco.drawDetectedCornersCharuco(frame, c_corners, c_ids, (255, 0, 255))
        except cv2.error:
            ret =  c_corners = c_ids = False
        return ret, c_corners, c_ids

    def read(self):
        frame_status, frame = self.camera.read_raw()
        if frame_status:
            ret, c_corners, c_ids = self.get_info_from_frame(frame)
            if ret:
                if len(c_corners) >=6:                                                               #! RI                   #! DI
                    _, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(c_corners, c_ids, self.board, self.camera.raw_matrix, self.camera.distortion, np.empty(1), np.empty(1))
                    if len(rvec) >1 and len(tvec)>1:
                        if self.camera.auto_update:
                            self.camera.vecs = (rvec, tvec)
                        return frame_status, frame, (rvec, tvec)
        return False, frame, (None, None)
    
    def calibrate_from_dir(self, path="./pictures/", extension="jpg"):
        """Read all imagens from an path with specific extension (jpg, png, etc)
        and perform the calibration with `calibrate_from_images()` method.
        """
        return self.calibrate_from_images([ cv2.imread(image) for image in glob(f'{path}*.{extension}')])
    
    def calibrate_from_images(self, images):
        """Use and dataset of images with chessboard pattern to perform an camera calibration.

        Args:
            images (list(MatLike, ...)): Lista de imagens.

        Returns:
            MatLike: (CameraMatrix)
            MatLike: (UndistortedCameraMatrix)
            MatLike: (DistortionCoefficient)
            Rect:    ROI to apply `after` remove the distortion with values above to remove the visual artifacts.
        """
        
        object_points = []
        image_points = []
        for img in images:
            ret, c_corners, c_ids = self.get_info_from_frame(img)
            if ret:
                object_points.append(c_corners)
                image_points.append(c_ids)
        return self.calib(object_points, image_points)
    
    def calib(self, object_points, image_points, w=1920,h=1080):
        ret, RawCameraMatrix, Distortion, rvecs, tvecs =  cv2.aruco.calibrateCameraCharuco(object_points, image_points, self.board, (1920,1080), None, None)
        UndistortedCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(RawCameraMatrix, Distortion, (w,h), 1, (w,h))
        if self.camera.auto_update:
            self.camera.matrix = UndistortedCameraMatrix
            self.camera.distortion = Distortion
            self.camera.roi = roi
            self.camera.raw_matrix = RawCameraMatrix
            self.camera.calibrated = True
        return RawCameraMatrix, UndistortedCameraMatrix, Distortion, roi
    
    def read_raw(self):
        return self.read()
    
    @property
    def vecs(self):
        """Getting wil return:
        Last stored Rotation and Transformation vectors
        """
        return self.camera.vecs
    
    @property
    def matrix(self):
        """Getting wil return:
        Last stored Rotation and Transformation vectors
        """
        return self.camera.matrix
    
    @property
    def thread(self):
        return self.camera.thread


def AR(refPts, source):
    (refPtTL, refPtBL, refPtBR, refPtTR) = refPts[0]

    dstMat = [refPtTL, refPtTR, refPtBR, refPtBL] #! Maybe detect de charuco?
    dstMat = np.array(dstMat)
    
    (srcH, srcW) = source.shape[:2]
    srcMat = np.array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
    
    (H, _) = cv2.findHomography(srcMat, dstMat)
    return cv2.warpPerspective(source, H, (1920, 1080))


if __name__ == "__main__":

    cam = OpenCV(configuration={cv2.CAP_PROP_FRAME_HEIGHT: 1080, cv2.CAP_PROP_FRAME_WIDTH:1920})
    Calib = Calibration(cam)
    Plane = WorkingArea(cam, square_size=23.7, marker_size=19)

    RI, URI, DI, ROI = Calib.calibrate_from_dir()
    x,y,w,h = ROI

    size_x = 200
    size_y = 125
    alpha = 0.4
    working_plane = [
        (0,0,0),
        (0, size_y,0),
        (size_x,size_y,0),
        (size_x, 0 ,0),
    ]
    while cv2.waitKey(1) != 27:
        ret, _, frame,  = Plane.read()
        if ret:
            frame = cv2.undistort(frame, RI, DI, None, URI)

            for PR in Calib.object_points:
                try:
                    PV = cam._2DP(PR)
                    cv2.circle(frame, PV, 2, (0, 255, 0), -1)


                except AssertionError:
                    print("Erro")
                    pass

            virtual_plane = cam._2DPS(*working_plane)
            
            b = cv2.fillPoly(frame.copy(), virtual_plane, (0,0,0))
            frame = cv2.addWeighted(b, alpha, frame, 1-alpha, 0)

            cv2.imshow("frame", frame[y:y+h, x:x+w])
    cv2.imwrite("test.jpg", frame)
