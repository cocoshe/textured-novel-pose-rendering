import math
import os

import cv2
from tqdm import tqdm
import numpy as np
from smpl.smpl_numpy import SMPL
import trimesh
import pyrender
import matplotlib.pyplot as plt
import torch
from texture_utils import estimate_translation
from texture_utils import img_preprocess

def batch_orth_proj(X, camera, mode='2d',keep_dim=False):
    X = torch.tensor(X)
    camera = torch.tensor(camera)
    camera = camera.view(-1, 1, 3)
    X_camed = X[:,:,:2] * camera[:, :, 0].unsqueeze(-1)
    X_camed += camera[:, :, 1:]
    if keep_dim:
        X_camed = torch.cat([X_camed, X[:,:,2].unsqueeze(-1)],-1)
    return X_camed


def convert_cam_to_3d_trans2(j3ds, pj3d):
    j3ds = torch.tensor(j3ds)
    predicts_j3ds = j3ds[:,:24].contiguous().detach().cpu().numpy()
    predicts_pj2ds = (pj3d[:,:,:2][:,:24].detach().cpu().numpy()+1)*256
    cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, \
                                focal_length=443.4, img_size=np.array([512,512])).to(j3ds.device)
    return cam_trans

def convert_proejection_from_input_to_orgimg(kps, offsets):
    top, bottom, left, right, h, w = offsets
    img_pad_size = max(h,w)
    kps[:, :, 0] = (kps[:,:,0] + 1) * img_pad_size / 2 - left
    kps[:, :, 1] = (kps[:,:,1] + 1) * img_pad_size / 2 - top
    if kps.shape[-1] == 3:
        kps[:, :, 2] = (kps[:,:,2] + 1) * img_pad_size / 2
    return kps


def body_mesh_projection2image(j3d_preds, cam_preds, vertices=None, input2org_offsets=None):
    pj3d = batch_orth_proj(j3d_preds, cam_preds, mode='2d')
    pred_cam_t = convert_cam_to_3d_trans2(j3d_preds, pj3d)
    projected_outputs = {'pj2d': pj3d[:,:,:2], 'cam_trans':pred_cam_t}
    if vertices is not None:
        projected_outputs['verts_camed'] = batch_orth_proj(vertices, cam_preds, mode='3d',keep_dim=True)

    if input2org_offsets is not None:
        projected_outputs['pj2d_org'] = convert_proejection_from_input_to_orgimg(projected_outputs['pj2d'], input2org_offsets)
        projected_outputs['verts_camed_org'] = convert_proejection_from_input_to_orgimg(projected_outputs['verts_camed'], input2org_offsets)
    return projected_outputs


def novel_pose_rendering(sex='neutral', model='smpl', front_img='P01125-150055.jpg', image_pad_info=None):
    MODEL_DIR = '{}/models'.format(model)
    # load SMPL weight
    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)
    # load template SMPL mesh
    template_mesh = trimesh.load('models/smpl_uv.obj', process=False, maintain_order=True)

    # define body_beta (human shape relative estimated by smplify-x)
    image_name = front_img.split('.')[0]
    betas = np.load('data/obj1/smpl/results/{}/000.pkl'.format(image_name), allow_pickle=True)['betas'][0]
    # print('betas.shape: {}'.format(betas.shape))

    # load motion sequences
    romp_results = np.load('./motion_results/video_results.npz', allow_pickle=True)['results'][()]
    # romp_results = {k: v for k, v in romp_results.items() if k.startswith('0')}
    keys = list(romp_results.keys())
    keys.sort()
    # print(keys)

    if not os.path.exists('motion_results'):
        os.mkdir('motion_results')

    r = pyrender.OffscreenRenderer(512, 512)
    for k in tqdm(keys):
        # romp_results[k]['global_orient'][0][0] += np.pi
        poses = np.concatenate([romp_results[k]['global_orient'][0], romp_results[k]['body_pose'][0]]).reshape([72, 1])
        v, _ = smpl_model(poses, betas)
        romp_results[k]['verts'] = v[None]  # fix dimension
        romp_results[k]['smpl_betas'] = betas
        output = romp_results[k]
        output.update(body_mesh_projection2image(output['joints'], output['cam'], vertices=output['verts'], input2org_offsets=image_pad_info))

        tmp = template_mesh.copy()
        tmp.vertices = output['verts_camed'][0].cpu().numpy()
        # print(tmp.vertices.shape)
        # render
        mesh = pyrender.Mesh.from_trimesh(tmp)
        scene = pyrender.Scene()
        scene.add(mesh)

        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.5],
            [0.0, 0.0, 1.0, -1.5],
            [0.0, 0.0, 0.0, 1.0],
        ])
        cam_rot_mat, _ = cv2.Rodrigues(np.array([np.pi - 0.08, 0., 0.]).reshape(3, 1))
        camera_pose[:3, :3] = cam_rot_mat
        # print(camera_pose)

        # render front view
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        scene.add(camera, pose=camera_pose)

        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        scene.add(dl, pose=camera_pose)


        # render 1: 3D Viewer
        # pyrender.Viewer(scene, use_raymond_lighting=True,)

        # # render 2: Offscreen Rendering
        # r = pyrender.OffscreenRenderer(512, 512)
        color, depth = r.render(scene)
        plt.axis('off')
        # plt.imshow(color)
        # plt.show()
        plt.imsave('motion_snapshots/{}.png'.format(k.split('.')[0]), color)
    r.delete()


if __name__ == '__main__':
    novel_pose_rendering()
