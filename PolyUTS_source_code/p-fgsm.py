"""
Scene privacy protection
C.Y. Li, A.S. Shamsabadi, R. Sanchez-Matilla, R. Mazzon, A. Cavallaro
Proc. of IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), Brighton, UK, May 12-17, 2019
If you find this code useful in your research, please consider citing
Please check the license of this code in License.txt
"""

import os
import glob
import argparse
from tqdm import tqdm

import numpy as np
import cv2

import torch
from torch import nn
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F

from functions import *
from PIL import Image
cuda = torch.cuda.is_available()


class attack():
    """
        Private Fast Gradient Sign Method (P-FGSM)
    """
    def __init__(self, model, epsilon, lb, up, args):

        self.model = model

        # Movement multiplier per iteration
        self.epsilon = epsilon

        # Perturbation step
        self.delta = 1/255.

        # Image boundaries for clipping
        self.lb = lb
        self.up = up

        # Create the folder to export images if not exists
        self.adv_path = createDirectories(args)

        # Maximum number of iterations
        self.maxIters = 100

    def generateAdversarial(self, original_image, img_name, org_class, set_target_classes, f_name, args):

        # Pick randomly target_label from set_target_classes
        # rand = np.random.randint(0, high=len(set_target_classes))
        target_class = set_target_classes[0]
        # Log initialisation
        text = img_name + '\n'

        # Target class
        im_label_as_var = Variable(torch.from_numpy(np.asarray([target_class])).long())
        if cuda:
            im_label_as_var = im_label_as_var.cuda()

        # Define loss function
        ce_loss = nn.CrossEntropyLoss()

        # Process image
        adv_image = preprocess_image(original_image, resize_im=True)

        # Iterative adversarial generation
        for itr in range(self.maxIters):

            # Compute loss
            adv_image.grad = None
            out = model(adv_image)
            loss = ce_loss(out, im_label_as_var)
            loss.backward()

            # Compute adversarial noise for current iteration
            adv_noise = self.delta * torch.sign(adv_image.grad.data)

            # Add noise and clip
            adv_image.data = addNoise(adv_image.data, adv_noise, self.lb, self.up)

            # Compute current class and probability for stopping condition
            logit = model(adv_image)
            h_x = F.softmax(logit, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            current_class = idx[0]
            current_class_prob = probs[0]
            target_class_prob = h_x[target_class]
            org_class_prob = h_x[org_class]

            # Check stopping condition
            if target_class_prob > 0.99:
                break
        cv2.imwrite('{}/{}'.format(self.adv_path, img_name), recreate_image(adv_image))
        path = './images/' + img_name
        path2 = './images/adversarials/'+img_name
        org_img = Image.open(path)
        b, g, r = org_img.split()
        org_img = Image.merge("RGB", (r, g, b))
        adv_image = Image.open(path2)
        w = org_img.size[0]
        h = org_img.size[1]
        org_img = org_img.resize((456, 256), Image.ANTIALIAS)
        org_img.paste(adv_image, (116, 16, 340, 240))
        org_img = org_img.resize((w, h),Image.ANTIALIAS)
        adv_image = org_img
        # Save final adversarial image

        adv_image.save('{}/{}'.format(self.adv_path, img_name))

        # Update log
        f = open(f_name, 'a+')
        f.write('{}\t{}\t{}\t{:.5f}\t{}\t{:.5f}\t{}\t{:.5f}\n'.format(
            img_name, itr+1, org_class, org_class_prob, current_class,
            current_class_prob, target_class, target_class_prob))
        f.close()
        return


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./Images')
    parser.add_argument('--eps', type=str, required=False, default='8/255')
    parser.add_argument('--sigma', type=float, required=False, default=0.99)
    args = parser.parse_args()
    try:
        args.eps = float(args.eps.split('/')[0])/float(args.eps.split('/')[1])
    except:
        print('Epsilon should be a fraction written as --eps=x/255 (in [0,1] domain)')

    # Initialisation

    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load('models/model.pth'))
    model.cuda()
    model.eval()
    imagePaths = loadImagePaths(args)
    f, f_name = createLogFiles(args)

    for index in tqdm(range(len(imagePaths))):

        # Read image and compute lower and uper bound
        img_name, org_image, lb, ub = readImage(args, imagePaths, index)

        # Computer set of target classes
        org_class, org_class_prob, set_target_classes = calculateSetTargetClasses(org_image, model, args)

        # Generate adversarial image
        if org_class==2:
            path='./images/'+img_name
            org = Image.open(path)
            b, g, r = org.split()
            org = Image.merge("RGB", (r, g, b))
            org.save('./images/adversarials/{}'.format(img_name))

        else:
            PFGSM = attack(model, args.eps, lb, ub, args)
            PFGSM.generateAdversarial(org_image, img_name, org_class, set_target_classes, f_name, args)
