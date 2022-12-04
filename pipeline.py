import os
import sys
import shutil
import cv2
import textured_smplx
import argparse

# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'third_party'))
# sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'third_party/human_body_prior/src'
#                                                                          '/human_body_prior/tools/'))

# print(sys.path)
from prepare_smpl_sequences import novel_pose_rendering
from texture_utils import img_preprocess


# print(sys.path)
# sys.path.append(os.path.join(os.path.dirname(__file__), 'smplify'))
# sys.path.append(os.path.join(os.path.dirname(__file__), 'smplify/smplifyx'))


USAGE = """python %s data_path, front_img, back_img, [model]
    data_path: the path to the data, should be like:
                data_path/images/XXX.jpg # image path
                data_path/smpl # output from smplify-x for SMPL model
                data_path/smplx # output from smplify-x for SMPLX model            
                data_path/images/XXX_PGN.jpg # PGN segmentation result (optional)
    front_img: name for the front image
    back_img: name for the back image
    model: can be 'smpl' or 'smplx', default
"""

# os.environ['PYOPENGL_PLATFORM'] = 'egl'

def texture_extraction(data_path, front_img, back_img, model='smpl'):
    ''' data_path: the path to the data, should be like:
            data_path/images/XXX.jpg # image path
            data_path/smpl # output from smplify-x for SMPL model
            data_path/smplx # output from smplify-x for SMPLX model
            data_path/images/XXX_PGN.jpg # PGN segmentation result (optional)

        front_img: name for the front image
        back_img: name for the back image
        model: can be 'smpl' or 'smplx'

        return: texture will be data_path/texture_smpl.png or data_path/texture_smplx.png
    '''

    # step.0: check all the input data
    front_img = os.path.split(front_img)[-1]  # remove the path
    back_img = os.path.split(back_img)[-1]  # remove the path

    tmp = front_img.rfind('.')
    front_id = front_img[:tmp]
    tmp = back_img.rfind('.')
    back_id = back_img[:tmp]

    if model == 'smpl':
        template_obj = 'models/smpl_uv.obj'
        template_mask = 'models/smpl_mask_1000.png'
    elif model == 'smplx':
        template_obj = 'models/smplx_uv.obj'
        template_mask = 'models/smplx_mask_1000.png'
    else:
        raise (Exception("model type not found"))

    if not os.path.isfile(template_obj) or not os.path.isfile(template_mask):
        raise (Exception("model not found"))

    f_img = os.path.join(data_path, 'images', front_img)
    f_obj = os.path.join(data_path, model, 'meshes', front_id, '000.obj')
    f_pkl = os.path.join(data_path, model, 'results', front_id, '000.pkl')
    print(f_img, f_obj, f_pkl)
    # f_pgn = os.path.join(data_path, 'PGN', '%s_PGN.png' % front_id)
    # if not os.path.isfile(f_pgn):
    #     f_pgn = None
    for fname, ftype in zip([f_img, f_obj, f_pkl], ['image', 'obj', 'pkl']):
        if not os.path.isfile(fname):
            raise (Exception("%s file for the front is not found" % ftype))

    b_img = os.path.join(data_path, 'images', back_img)
    b_obj = os.path.join(data_path, model, 'meshes', back_id, '000.obj')
    b_pkl = os.path.join(data_path, model, 'results', back_id, '000.pkl')
    # b_pgn = os.path.join(data_path, 'PGN', '%s_PGN.png' % back_id)
    # if not os.path.isfile(b_pgn):
    #     b_pgn = None
    for fname, ftype in zip([b_img, b_obj, b_pkl], ['image', 'obj', 'pkl']):
        if not os.path.isfile(fname):
            raise (Exception("%s file for the back is not found" % ftype))

    npath = os.path.join(data_path, 'texture_%s' % model)
    # if f_pgn and b_pgn:
    #     pgn_path = os.path.join(data_path, 'PGN_%s' % model)
    # else:
    #     pgn_path = None

    # step.1: produce single frame texture
    textured_smplx.get_texture_SMPL(f_img, f_obj, f_pkl, npath, 'front', template_obj)
    textured_smplx.get_texture_SMPL(b_img, b_obj, b_pkl, npath, 'back', template_obj)

    # step.2: produce PGN texture (optional)

    # textured_smplx.get_texture_SMPL(f_pgn, f_obj, f_pkl, npath, 'front_PGN', template_obj)
    # textured_smplx.get_texture_SMPL(b_pgn, b_obj, b_pkl, npath, 'back_PGN', template_obj)

    # step3: combine all the textures

    textured_smplx.combine_texture_SMPL(npath)

    # step4: complete all the textures

    f_acc_texture = os.path.join(npath, 'back_texture_acc.png')
    f_acc_vis = os.path.join(npath, 'back_texture_vis_acc.png')
    f_mask = template_mask

    textured_smplx.complete_texture(f_acc_texture, f_acc_vis, f_mask)

    # finish: copy the result

    # shutil.copyfile(f_acc_texture[:-4] + 'complete.png', os.path.join(data_path, 'texture_%s.png' % model))
    shutil.copyfile(f_acc_texture[:-4] + 'complete.png', os.path.join(data_path, 'texture_%s.png' % model))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/obj1', help='path to the data')
    parser.add_argument('--front_img', type=str, default='data/obj1/images/P01125-150055.jpg', help='name for the front image')
    parser.add_argument('--back_img', type=str, default='data/obj1/images/P01125-150146.jpg', help='name for the back image')
    parser.add_argument('--model', type=str, default='smpl', help='can be smpl or smplx')  # todo: add smplx support
    parser.add_argument('--romp_imgs', type=str, default='images/', help='path to the romp images')
    parser.add_argument('--romp_output', type=str, default='romp_output/', help='path to save romp output(include img and pkl)')
    parser.add_argument('--sex', type=str, default='neutral')
    parser.add_argument('--output_video', type=str, default='motion_render/motion_video.mp4')
    args = parser.parse_args()


    data_path = args.data_path
    front_img = args.front_img
    back_img = args.back_img
    model = args.model
    romp_imgs = args.romp_imgs
    romp_output = args.romp_output
    sex = args.sex
    output_video = args.output_video
    front_img_name = front_img.split('/')[-1]
    if not os.path.exists(output_video):
        os.makedirs(output_video.split('/')[0])


    img_prepare_folder = os.path.join(data_path, 'images')
    keypoints_output_folder = os.path.join(data_path, 'keypoints')
    pose_imgs_output_folder = os.path.join(data_path, 'pose_images')
    _, image_pad_info = img_preprocess(cv2.imread(front_img))
    os.system('openpose.bin --display 0 --render_pose 1 --image_dir {} --write_json {} '
              '--write_images {} --hand --face'
              .format(img_prepare_folder, keypoints_output_folder, pose_imgs_output_folder))

    cmd = './smplify-x/smplifyx/main.py --config ./smplify-x/cfg_files/fit_{}.yaml --data_folder ./{} ' \
          '--output_folder ./{}/{}  --model_folder ./smplify-x/models --vposer_ckpt ./smplify-x/V02_05'.format(model, data_path, data_path, model)

    print(cmd)
    os.system('{} {}'.format('python3', cmd))
    # os.system('romp --mode=video --calc_smpl --render_mesh -i={} -o={} -t -sc=1.'.format(romp_imgs, romp_output))

    os.system('cp {} {}/'.format(os.path.join(romp_output, 'video_results.npz'), 'motion_results'))
    texture_extraction(args.data_path, args.front_img, args.back_img, args.model)
    os.system('cp {} {}/'.format(os.path.join(data_path, 'texture_{}.png'.format(model)), 'models'))
    # novel_pose_rendering(sex=sex, model=model, front_img=front_img_name, image_pad_info=image_pad_info)
    novel_pose_rendering(sex=sex, model=model, front_img=front_img_name, image_pad_info=None)

    if os.path.exists(output_video):
        os.remove(output_video)
    os.system('ffmpeg -f image2 -i motion_snapshots/%06d.png {}'.format(output_video))

    print('Done!')
