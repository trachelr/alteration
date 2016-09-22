# Authors: Romain Trachel <trachelr@gmail.com>
#
# License: Apache License, Version 2.0

import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import sys, os, cv2
sys.path.insert(1, '/home/romain/demo_video/code/')
from tools import deepdream_jitter_ctrl, optflow_deepdream, preprocess
sys.path.insert(1, '/home/romain/packages/caffe/python/')
import caffe

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
caffe.set_mode_gpu()
# select GPU device
caffe.set_device(1)

# set random seed to be able to reproduce the dream
np.random.seed(42)

data_path = '/home/romain/demo_video/'
model_path = '/home/romain/packages/caffe/models/vgg_face/'
net_fn = model_path + 'VGG_FACE_deploy.prototxt'
param_fn = model_path + 'VGG_FACE.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" 
# line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       # ImageNet mean, training set dependent
                       mean=np.float32([104.0, 116.0, 122.0]),
                       # the reference model has channels in BGR order instead of RGB
                       channel_swap=(2, 1, 0))
iter_n = 10
octave_n = 4
octave_scale = 1.8
clip = True

layer = 'conv5_2'

# set frame start and stop to apply the dream
n_start = 670
n_stop = 1380
n_len = n_stop - n_start

# set automation for deep dream iterations
iter_up = np.linspace(1, 10, 10)
iter_down = np.ones([n_stop - n_start])
iters = np.linspace(1, 5, n_len).astype(int)
iter_n = iters[0]

#guide_list = ['CHAT/brian-gordon-green-gros-plan-d-un-chat-noir-et-blanc-a-poil-court.jpg',
#              'CHAT/Valerie-Albertosi-blog-chat-noir-et-blanc.jpg',
guide_list = ['VISAGES/CAST_ALTERATION_Jer02_Page_3_Image_0001.jpg',
              'VISAGES/CAST2_ALTERATION_Jer02_Page_5_Image_0001.jpg',
              'VISAGES/CAST_ALTERATION_Jer02_Page_5_Image_0004.jpg',
              'VISAGES/CAST_ALTERATION_Jer02_Page_2_Image_0001.jpg']

for iguide in range(1):
    guide_img = PIL.Image.open(data_path + 'BqueIMAGE/' + guide_list[iguide])
    guide_img = guide_img.resize((224, 224), PIL.Image.ANTIALIAS)
    guide = np.float32(guide_img)

    end = layer
    h, w = guide.shape[:2]
    src, dst = net.blobs['data'], net.blobs[end]
    src.reshape(1, 3, h, w)
    src.data[0] = preprocess(net, guide)
    net.forward(end=end)
    guide_features = dst.data[0].copy()

    # define the objective function
    def objective_guide(dst):
        x = dst.data[0].copy()
        y = guide_features
        ch = x.shape[0]
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)
        # compute the matrix of dot-products with guide features
        A = x.T.dot(y)
        # select ones that match best
        dst.diff[0].reshape(ch, -1)[:] = y[:, A.argmax(1)]

    dream_path = data_path + 'output/' + layer
    dream_path = dream_path + '_zoomend_slow_final_optiflow_octave_%i_iter_prog' % (octave_n)
    dream_path += '_guide%i' % iguide

    #dream_path = data_path + 'parti_communiste_1/' + layer
    #dream_path = dream_path + '_zoomend_slow_final_optiflow_octave_%i_iter_prog' % (octave_n)
    #dream_path += '_guide%i' % iguide

    if not os.path.exists(dream_path):
        os.mkdir(dream_path)

    image_name = 'IPHILIP_2880X1440_extrait_plage1' + ('%4i' % n_start).replace(' ', '0')
    img = PIL.Image.open(data_path + 'extrait/' + image_name + '.jpg')

    #image_name = 'IPHILIP_parti_communiste__00' + ('%3i' % n_start).replace(' ', '0')
    #img = PIL.Image.open(data_path + 'parti_communiste_1/IPHILIP_2880X1440_extrait_plage1/' + image_name + '.jpg')

    img = np.float32(img.resize((2880 / 2, 1440 / 2), PIL.Image.BILINEAR))
    h, w, c = img.shape
    hallu = optflow_deepdream(net, img, iter_n=iter_n, objective=objective_guide,
                              octave_n=octave_n, octave_scale=octave_scale,
                              end=layer, clip=True)

    res = PIL.Image.fromarray(hallu.astype(np.uint8))
    # res.resize((2880, 1440), PIL.Image.BILINEAR)
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for n in range(n_start, n_stop):
        iter_n = iters[n - n_start]
        previousImg = img.copy()
        previousGrayImg = grayImg.copy()

        image_name = 'IPHILIP_2880X1440_extrait_plage1' + ('%4i' % n).replace(' ', '0')
        img = PIL.Image.open(data_path + 'extrait/' + image_name + '.jpg')

        #image_name = 'IPHILIP_parti_communiste__00' + ('%3i' % n).replace(' ', '0')
        #img = PIL.Image.open(data_path + 'parti_communiste_1/IPHILIP_2880X1440_extrait_plage1/' + image_name + '.jpg')
        img = np.float32(img.resize((2880 / 2, 1440 / 2), PIL.Image.BILINEAR))

        grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # compute optical flow
        flow = cv2.calcOpticalFlowFarneback(previousGrayImg, grayImg, pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        flow = -flow
        flow[:, :, 0] += np.arange(w)
        flow[:, :, 1] += np.arange(h)[:, np.newaxis]
        halludiff = hallu - previousImg
        halludiff = cv2.remap(halludiff, flow, None, cv2.INTER_LINEAR)
        hallu = (img + halludiff * .7).copy()

        hallu = optflow_deepdream(net, hallu, iter_n=iter_n, objective=objective_guide,
                                  octave_n=octave_n, octave_scale=octave_scale,
                                  end=layer, clip=True)

        np.clip(hallu, 0, 255, out=hallu)
        res = PIL.Image.fromarray(hallu.astype(np.uint8))
        # res.resize((2880, 1440), PIL.Image.BILINEAR)
        print 'saving ' + image_name
        res.save(dream_path + '/' + image_name + '.jpg')

################################
#       Zooming end
################################
h, w = hallu.shape[:2]
slen = 600
# scale coefficient
scoef = 0.05
scales = np.linspace(0, 1, slen) * scoef
for i in xrange(slen):
    hallu = optflow_deepdream(net, hallu, iter_n=iter_n, objective=objective_guide,
                              octave_n=octave_n, octave_scale=octave_scale,
                              end=layer, clip=True)
    hallu = hallu.astype(np.uint8)
    res = PIL.Image.fromarray(hallu)
    # res.resize((2880, 1440), PIL.Image.BILINEAR)
    image_name = 'IPHILIP_2880X1440_extrait_plage1' + ('%4i' % (n + 1)).replace(' ', '0')
    print 'saving ' + image_name
    res.save(dream_path + '/' + image_name + '.jpg')
    print 'zooming %i / %i' % (i, slen)
    s = scales[i]
    hallu = nd.affine_transform(hallu,
                                [1 - s, 1 - s, 1],
                                [h * s / 2, w * s / 2, 0], order=1)
    n += 1

