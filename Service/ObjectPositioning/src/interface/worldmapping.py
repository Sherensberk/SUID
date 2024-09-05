from typing import Any
from numpy import (
    c_, r_, array,
    zeros, float32, mgrid, arange,
    empty,
)
from glob import glob

from numpy.linalg import inv

from cv2.aruco import (
    DICT_6X6_250,
    getPredefinedDictionary,
    CharucoBoard,
    detectMarkers,
    interpolateCornersCharuco,
    estimatePoseCharucoBoard,
    calibrateCameraCharuco
)

from cv2 import (
    cvtColor, COLOR_BGR2GRAY, error, Rodrigues, getOptimalNewCameraMatrix, imread,
    findHomography, warpPerspective
)

class PointCorrelation:
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
        return array([ self._2DP(xyz) for xyz in n_xyz]).reshape(1,-1,2)
    
    def _3DPS(self, *n_uv):
        """
        Convert a group of image points (px,py) into a group of world coordinates (xw, xy, xz).\n
        XYZ = Rotation^-1 . ((Matrix^-1 . scale) - tvec) | for 'n' uv"""
        return array([ self._3DP(uv) for uv in n_uv]).reshape(1,-1,3)

class Plane:
    def __init__(self, dict_type = DICT_6X6_250, board_shape= (7, 5), square_size=25, marker_size=20, ids_offset=100):
        self.dictionary = getPredefinedDictionary(dict_type)
        self.board = CharucoBoard(board_shape, square_size, marker_size, self.dictionary, arange(ids_offset, ids_offset+int((board_shape[0]*board_shape[1])/2),1))
        self.board.setLegacyPattern(True)
        self.correlation = PointCorrelation()

        objp = zeros((board_shape[1]*board_shape[0],3), float32)
        objp[:,:2] = mgrid[0:board_shape[0],0:board_shape[1]].T.reshape(-1,2)
        self.object_points = objp * square_size

        size_x = square_size*board_shape[0]
        size_y = square_size*board_shape[1]

        self.working_plane = [
            (0,0,0),
            (0, size_y,0),
            (size_x,size_y,0),
            (size_x, 0 ,0),
        ]

        self.size = (1920, 1080)
    
    @property
    def virtual_plane(self): 
        return self.correlation._2DPS(*self.working_plane)
    
    @property
    def mask(self):
        return AR(self.virtual_plane, self.blank_canvas, *self.size)

    # @property
    # def virtual_world(self):
    #     return AR(self.virtual_plane, self.board_image)
    
    # @property
    # def board_image(self):
    #     return self.board.generateImage(
    #         (
    #             self.working_plane[2][0],
    #             self.working_plane[2][1]
    #         )
    #     )
    
    @property
    def virtual_plane_size(self):
        return (
            self.virtual_plane[0][0][0] - self.virtual_plane[0][2][0],
            self.virtual_plane[0][0][1] - self.virtual_plane[0][2][1]
        )
    @property
    def blank_canvas(self):
        _blank_canvas = zeros(self.virtual_plane_size, dtype="uint8")
        _blank_canvas.fill(255)
        return _blank_canvas
    
    def update_real_image(self, frame):
        r, (rv,cv) = spacial_info(self, frame)
        if r :
            self.correlation.vecs = (rv,cv)
        self.size = frame.shape[:2][::-1]

    def calibrate_from_dir(self, path="./pictures/", extension="jpg"):
        """Read all imagens from an path with specific extension (jpg, png, etc)
        and perform the calibration with `calibrate_from_images()` method.
        """
        return self.calibrate_from_images([ imread(image) for image in glob(f'{path}*.{extension}')])
    
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
            ret, c_corners, c_ids = get_info_from_frame(img, self.dictionary, self.board)
            if ret:
                object_points.append(c_corners)
                image_points.append(c_ids)
        return self.calib(object_points, image_points, *images[-1].shape[:2][::-1])
    
    def calib(self, object_points, image_points, w=1920,h=1080):
        ret, RawCameraMatrix, Distortion, rvecs, tvecs =  calibrateCameraCharuco(object_points, image_points, self.board, (w,h), None, None)
        UndistortedCameraMatrix, roi = getOptimalNewCameraMatrix(RawCameraMatrix, Distortion, (w,h), 1, (w,h))
        self.correlation.matrix = UndistortedCameraMatrix
        self.correlation.distortion = Distortion
        self.correlation.roi = roi
        self.correlation.raw_matrix = RawCameraMatrix
        self.correlation.calibrated = True
        return RawCameraMatrix, UndistortedCameraMatrix, Distortion, roi

def AR(refPts, source, w,h):
    (refPtTL, refPtBL, refPtBR, refPtTR) = refPts[0]

    dstMat = [refPtTL, refPtTR, refPtBR, refPtBL] #! Maybe detect de charuco?
    dstMat = array(dstMat)
    
    (srcH, srcW) = source.shape[:2]
    srcMat = array([[0, 0], [srcW, 0], [srcW, srcH], [0, srcH]])
    
    (H, _) = findHomography(srcMat, dstMat)
    return warpPerspective(source, H, (w, h))


#! Class?
def get_info_from_frame(frame, dictionary, board):
    gray = cvtColor(frame, COLOR_BGR2GRAY)
    corners, ids, rejected_img_points = detectMarkers(gray, dictionary)
    try:
        ret, c_corners, c_ids = interpolateCornersCharuco(corners, ids, gray, board)
    except error:
        ret =  c_corners = c_ids = False
    return ret, c_corners, c_ids

def get_vecs_from_info(c_corners, c_ids, board, raw_matrix, distortion):
    if len(c_corners) >=6:                                                #! RI       #! DI
        _, rvec, tvec = estimatePoseCharucoBoard(c_corners, c_ids, board, raw_matrix, distortion, empty(1), empty(1))
        if len(rvec) >1 and len(tvec)>1:
            return True, (rvec, tvec)
    return False, (False, False)

def spacial_info(plane, frame):
    r, cc, ci = get_info_from_frame(frame, plane.dictionary, plane.board)
    if r:
        return get_vecs_from_info(cc, ci, plane.board, plane.correlation.raw_matrix, plane.correlation.distortion)
    return False, (False, False)

# #! Refatorar
# def draw_dark_info(i,result):
#     centers = []
#     for _idx, res in enumerate(result):
#         if res['best_class'] == 0:
#             c = round(res['best_probability'], 2)
#             if c > 0.1:
#                 x,y,w,h = res['rect']['x'], res['rect']['y'], res['rect']['width'], res['rect']['height']
#                 color = (0,int(255*c),int(255*(1-c)))
#                 cv2.rectangle(i, (x,y), (x+w, y+h), color, 2)

#                 if c > 0.9:
#                     cv2.drawMarker(i, (int(x+(w/2)), int(y+(h/2))), (255,0,0), 2, 100, 1)
#                     centers.append(tuple((int(x+(w/2)), int(y+(h/2)))))
#                 for idx2, k in enumerate(zip(['pred', 'confidence', 'class'],
#                     ['name', 'best_probability', 'best_class'])):
#                     cv2.putText(i, f"{k[0] or k[1]}: {res[k[1]]}", (x+w, y+(idx2*20)+15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, color, 1)
#                     # cv2.putText(i, f"{res[k]}", (x+w, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,255,255), 2)
#     return centers


class coordinate():
    def __init__(self, x=0, y=0, z=0) -> None:
        self.x = x
        self.y = y
        self.z = z

    def values(self):
        return self.x,self.y,self.z

class referencial:
    def __init__(self, real={'x':0, 'y':0, 'z':0}, virtual={'x':0, 'y':0, 'z':0}, plane:Plane = None):
        self._real = coordinate(**real)
        self._virtual = coordinate(**virtual)
        self.plane = plane
    
    @property
    def real(self):
        if self.plane is not None:
           return self.plane.correlation._3DP(array(self._virtual.values())[:2])     
        return self._real
   
    @property
    def virtual(self):
        if self.plane is not None:
           return self.plane.correlation._2DP(array(self._real.values()))     
        return self._virtual

class Item:
    def __init__(self, name, uv={'x':0, 'y':0, 'z':0}, xyz={'x':0, 'y':0, 'z':0}, plane:Plane=None):
        self.name = name
        self.coordinate = referencial(real=xyz, virtual=uv, plane=plane)

if __name__ == '__main__':
    import cv2
    plane = Plane(cv2.aruco.DICT_4X4_1000, (14,10),20,15,100 )

    RI, URI, DI, ROI = plane.calibrate_from_dir('/home/delta/protonet/Refactor/Camera/calibration/charuco/')

    frame = cv2.imread('/home/delta/protonet/Refactor/Interpreter/src/interface/img.jpg')

    new_frame = cv2.undistort(frame, RI, DI, None, URI)
    i = Item('a', uv={'x':478, 'y':387}, xyz={'x':0, 'y':0 }, plane=plane)

    plane.update_real_image(frame)

    for PR in plane.object_points:
        try:
            PV = plane.correlation._2DP(PR)
            PT = plane.correlation._3DP(PV)
            cv2.circle(frame, PV, 2, (0, 255, 0), -1)
            cv2.circle(new_frame, PV, 2, (0, 255, 0), -1)
        except AssertionError:
            print("Erro")
            pass

    cv2.imwrite('./frame.jpg', frame)
    cv2.imwrite('./nframe.jpg', new_frame)
    # cv2.waitKey(0)
    # print(coordinate(**{'x':1, 'y':2, 'z':3}))

    print(i.coordinate._real, i.coordinate._virtual)
    print(i.coordinate.real, i.coordinate.virtual)
