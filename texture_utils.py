import numpy as np
import scipy
import scipy.optimize
import cv2
import torch


def draw_p2d(img, p2d, index=None, draw_id=False, p_size=1):
    ''' img: RGB image
        p2d: [[x,y]...] or [[x,y,z]...]
        index: only draw p2d[index]
        draw_id: todo
        return: new img
    '''
    if type(p2d) == list:
        p2d = np.array(p2d)
    p2d = p2d.squeeze()
    if len(p2d.shape) == 1:
        p2d = p2d.reshape(1, -1)
    assert(len(p2d.shape)==2)
    assert(p2d.shape[1] in (2,3))
    if p2d.shape[1] == 3:
        z = p2d[:,2]
        p2d = p2d[:,:2]
    else:
        z = np.ones(p2d.shape[0])
    colors = np.arange(0,25).astype(np.uint8)*10
    colors = cv2.applyColorMap(colors, cv2.COLORMAP_HSV)
    colors = colors.reshape(-1, 3).tolist()
    ncolors = len(colors)
    if index is None:
        index = range(len(p2d))
    img = img.copy()
    h,w = img.shape[:2]
    for no in index:
        x,y = p2d[no]
        x,y = int(x), int(y)
        if z[no]>0 and 0<=x<w and 0<=y<h:
            cv2.circle(img, (x,y), p_size, colors[no%ncolors], -1)
            if draw_id:
                img = cv2.putText(img, '%d'%no, (x+5,y+5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, 
                      (0,0,255), 1)
    return img

def solve_Ax_B(A, B):
    return np.linalg.lstsq(A, B, rcond=-1)[0]
    
    
def obj_vv(fname): # read vertices: (x,y,z)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('v '):
                tmp = line.split(' ')
                v = [float(i) for i in tmp[1:4]]
                res.append(v)
                
    return np.array(res, dtype=np.float)
    
def obj_vt(fname): # read texture coordinates: (u,v)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('vt '):
                tmp = line.split(' ')
                v = [float(i) for i in tmp[1:3]]
                res.append(v)
    return np.array(res, dtype=np.float)

def obj_fv(fname): # read vertices id in faces: (vv1,vv2,vv3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[0]) for i in tmp[1:4]]
                else:
                    v = [int(i) for i in tmp[1:4]]
                res.append(v)
    return np.array(res, dtype=np.int) - 1 # obj index from 1

def obj_ft(fname): # read texture id in faces: (vt1,vt2,vt3)
    res = []
    with open(fname) as f:
        for line in f:
            if line.startswith('f '):
                tmp = line.split(' ')
                if '/' in tmp[1]:
                    v = [int(i.split('/')[1]) for i in tmp[1:4]]
                else:
                    raise(Exception("not a textured obj file"))
                res.append(v)
    return np.array(res, dtype=np.int) - 1 # obj index from 1


def fv2norm(fv, vv):
    ''' calculate face norm
    # similar to the following method using trimesh 
    # mesh = trimesh.Trimesh(vv, fv, process=False)
    # return mesh.face_normals
    '''    
    p1 = vv[fv[:,0]]
    p2 = vv[fv[:,1]]
    p3 = vv[fv[:,2]]
    p12 = p2 - p1
    p13 = p3 - p1
    n = np.cross(p12, p13)
    n = n / (np.sqrt((n*n).sum(1))).reshape(-1,1)
    fnorm = n
    return fnorm

def p3d_p2d(p3d, rmat, tvec, cameraMatrix, distCoeffs=None, img_shape=None):
    ''' similar to cv2.projectPoints, there are 3 difference:
        1. use rmat as input rather than rvec
        2. the output is [[u,v,z]...], when z>0, the point is visible
        3. by providing img_shape, you can get rid of invisible points due to high distortion
    '''
    p_w = p3d.copy().T # world coordinate
    p_c_0 = rmat.dot(p_w) + tvec.reshape(3, 1) # camera coordinate
    z = p_c_0[2, :].copy().reshape(1, -1) # z (>0 are visible points)
    p_c_1 = p_c_0[:2] / z # camera coordinate (normalized)
    
    if img_shape:
        p_c_dis = np.ones_like(p_c_0) # undistort coordinates
        p_c_dis[:2,:] = p_c_1       
        p_uv = cameraMatrix.dot(p_c_dis)
        h,w = img_shape
        z[0,p_uv[0,:]<0]=-1
        z[0,p_uv[0,:]>w]=-1
        z[0,p_uv[1,:]<0]=-1
        z[0,p_uv[1,:]>h]=-1
    
    if distCoeffs is not None:
        k1,k2,p1,p2,k3 = distCoeffs
        r2 = (p_c_1 * p_c_1).sum(0).copy()
        kv = (1+k1*r2+k2*r2*r2+k3*r2*r2*r2)
        xy = (p_c_1[0,:] * p_c_1[1,:]).copy()
        x2 = p_c_1[0,:] * p_c_1[0,:]
        y2 = p_c_1[1,:] * p_c_1[1,:]
        p_c_1[0,:] = p_c_1[0,:]*kv + 2*p1*xy + p2*(r2+2*x2)
        p_c_1[1,:] = p_c_1[1,:]*kv + p1*(r2+2*y2) + 2*p2*xy
    p_c_dis = np.ones_like(p_c_0) # undistort coordinates
    p_c_dis[:2,:] = p_c_1       
    p_uv = cameraMatrix.dot(p_c_dis)
    p_uv[2,:] = z
    return p_uv.T

def cv2_triangle(img, p123):
    ''' draw triangles using OpenCV '''
    p1, p2, p3 = (tuple(i) for i in p123)
    cv2.line(img, p1, p2, (255, 0, 0), 1) 
    cv2.line(img, p2, p3, (255, 0, 0), 1) 
    cv2.line(img, p1, p3, (255, 0, 0), 1)
    return img



INVALID_TRANS=np.ones(3)*-1


def estimate_translation_cv2(joints_3d, joints_2d, focal_length=600, img_size=np.array([512.,512.]), proj_mat=None, cam_dist=None):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0,0], camK[1,1] = focal_length, focal_length
        camK[:2,2] = img_size//2
    else:
        camK = proj_mat
    ret, rvec, tvec,inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                              flags=cv2.SOLVEPNP_EPNP,reprojectionError=20,iterationsCount=100)

    if inliers is None:
        return INVALID_TRANS
    else:
        tra_pred = tvec[:,0]
        return tra_pred


def estimate_translation_np(joints_3d, joints_2d, joints_conf, focal_length=600, img_size=np.array([512.,512.]), proj_mat=None):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = joints_3d.shape[0]
    if proj_mat is None:
        # focal length
        f = np.array([focal_length,focal_length])
        # optical center
        center = img_size/2.
    else:
        f = np.array([proj_mat[0,0],proj_mat[1,1]])
        center = proj_mat[:2,2]

    # transformations
    Z = np.reshape(np.tile(joints_3d[:,2],(2,1)).T,-1)
    XY = np.reshape(joints_3d[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(joints_3d, joints_2d, pts_mnum=4, focal_length=600, proj_mats=None, cam_dists=None,
                         img_size=np.array([512., 512.])):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """
    if torch.is_tensor(joints_3d):
        joints_3d = joints_3d.detach().cpu().numpy()
    if torch.is_tensor(joints_2d):
        joints_2d = joints_2d.detach().cpu().numpy()

    if joints_2d.shape[-1] == 2:
        joints_conf = joints_2d[:, :, -1] > -2.
    elif joints_2d.shape[-1] == 3:
        joints_conf = joints_2d[:, :, -1] > 0
    joints3d_conf = joints_3d[:, :, -1] != -2.

    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float)
    if proj_mats is None:
        proj_mats = [None for _ in range(len(joints_2d))]
    if cam_dists is None:
        cam_dists = [None for _ in range(len(joints_2d))]
    # Find the translation for each example in the batch
    for i in range(joints_3d.shape[0]):
        S_i = joints_3d[i]
        joints_i = joints_2d[i, :, :2]
        valid_mask = joints_conf[i] * joints3d_conf[i]
        if valid_mask.sum() < pts_mnum:
            trans[i] = INVALID_TRANS
            continue
        if len(img_size.shape) == 1:
            imgsize = img_size
        elif len(img_size.shape) == 2:
            imgsize = img_size[i]
        else:
            raise NotImplementedError
        try:
            trans[i] = estimate_translation_cv2(S_i[valid_mask], joints_i[valid_mask],
                                                focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i],
                                                cam_dist=cam_dists[i])
        except:
            trans[i] = estimate_translation_np(S_i[valid_mask], joints_i[valid_mask],
                                               valid_mask[valid_mask].astype(np.float32),
                                               focal_length=focal_length, img_size=imgsize, proj_mat=proj_mats[i])

    return torch.from_numpy(trans).float()



def padding_image(image):
    h, w = image.shape[:2]
    side_length = max(h, w)
    pad_image = np.zeros((side_length, side_length, 3), dtype=np.uint8)
    top, left = int((side_length - h) // 2), int((side_length - w) // 2)
    bottom, right = int(top+h), int(left+w)
    pad_image[top:bottom, left:right] = image
    image_pad_info = torch.Tensor([top, bottom, left, right, h, w])
    return pad_image, image_pad_info


def img_preprocess(image, input_size=512):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pad_image, image_pad_info = padding_image(image)
    input_image = torch.from_numpy(cv2.resize(pad_image, (input_size,input_size), interpolation=cv2.INTER_CUBIC))[None].float()
    return input_image, image_pad_info

