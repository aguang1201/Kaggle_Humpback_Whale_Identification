# Read the dataset description
import gzip
# Read or generate p2h, a dictionary of image name to image id (picture to hash)
import pickle
import platform
import random
# Suppress annoying stderr output when importing keras.
import sys
from lap import lapjv
from math import sqrt
# Determine the size of each image
from os.path import isfile
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
# from keras.utils import multi_gpu_model
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as pil_image
from imagehash import phash
from keras import backend as K
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, MaxPooling2D, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from keras.utils import Sequence
from pandas import read_csv
from scipy.ndimage import affine_transform
from tqdm import tqdm_notebook as tqdm
import time
import os
import tensorflow as tf
from datetime import datetime
# from callback import MultiGPUModelCheckpoint
from losses import focal_loss
from build_model import build_model

data_dir = '/home/ys1/dataset/Humpback_Whale/'
TRAIN_DF = os.path.join(data_dir, 'train.csv')
SUB_Df = os.path.join(data_dir, 'sample_submission.csv')
TRAIN = os.path.join(data_dir, 'train/')
TEST = os.path.join(data_dir, 'test/')
P2H = os.path.join(data_dir, 'p2h.pickle')
P2SIZE = os.path.join(data_dir, 'p2size.pickle')
BB_DF = os.path.join(data_dir, 'bounding_boxes_concat.csv')
# MPIOTTE_STANDARD_MODEL = os.path.join(data_dir, 'mpiotte-standard.model')
output_dir = 'experiments/binary_crossentropy_no_anisotropy_imgsize512_shenxing_margin0'
MPIOTTE_STANDARD_MODEL = os.path.join('experiments/binary_crossentropy_no_anisotropy_imgsize512/models/weights_finetuning_epoch250.h5')
tagged = dict([(p, w) for _, p, w in read_csv(TRAIN_DF).to_records()])
submit = [p for _, p, _ in read_csv(SUB_Df).to_records()]
join = list(tagged.keys()) + submit
batch_size = 32             #image_size=512
workers = 12
max_queue_size = 10
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
img_shape = (512, 512, 1)
# anisotropy = 2.15  # The horizontal compression ratio
crop_margin = 0.05  # The margin added around the bounding box to compensate for bounding box inaccuracy

class TrainingData(Sequence):
    def __init__(self, score, steps=1000, batch_size=32):
        """
        @param score the cost matrix for the picture matching
        @param steps the number of epoch we are planning with this score matrix
        """
        super(TrainingData, self).__init__()
        self.score = -score  # Maximizing the score is the same as minimuzing -score.
        self.steps = steps
        self.batch_size = batch_size
        for ts in w2ts.values():
            idxs = [t2i[t] for t in ts]
            for i in idxs:
                for j in idxs:
                    self.score[
                        i, j] = 10000.0  # Set a large value for matching whales -- eliminates this potential pairing
        self.on_epoch_end()

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, len(self.match) + len(self.unmatch))
        size = end - start
        assert size > 0
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        b = np.zeros((size,) + img_shape, dtype=K.floatx())
        c = np.zeros((size, 1), dtype=K.floatx())
        j = start // 2
        for i in range(0, size, 2):
            a[i, :, :, :] = read_for_training(self.match[j][0])
            b[i, :, :, :] = read_for_training(self.match[j][1])
            c[i, 0] = 1  # This is a match
            a[i + 1, :, :, :] = read_for_training(self.unmatch[j][0])
            b[i + 1, :, :, :] = read_for_training(self.unmatch[j][1])
            c[i + 1, 0] = 0  # Different whales
            j += 1
        return [a, b], c

    def on_epoch_end(self):
        if self.steps <= 0: return  # Skip this on the last epoch.
        self.steps -= 1
        self.match = []
        self.unmatch = []
        _, _, x = lapjv(self.score)  # Solve the linear assignment problem
        y = np.arange(len(x), dtype=np.int32)

        # Compute a derangement for matching whales
        for ts in w2ts.values():
            d = ts.copy()
            while True:
                random.shuffle(d)
                if not np.any(ts == d): break
            for ab in zip(ts, d): self.match.append(ab)

        # Construct unmatched whale pairs from the LAP solution.
        for i, j in zip(x, y):
            if i == j:
                print(self.score)
                print(x)
                print(y)
                print(i, j)
            assert i != j
            self.unmatch.append((train[i], train[j]))

        # Force a different choice for an eventual next epoch.
        self.score[x, y] = 10000.0
        self.score[y, x] = 10000.0
        random.shuffle(self.match)
        random.shuffle(self.unmatch)
        # print(len(self.match), len(train), len(self.unmatch), len(train))
        assert len(self.match) == len(train) and len(self.unmatch) == len(train)

    def __len__(self):
        return (len(self.match) + len(self.unmatch) + self.batch_size - 1) // self.batch_size

# A Keras generator to evaluate only the BRANCH MODEL
class FeatureGen(Sequence):
    def __init__(self, data, batch_size=64, verbose=1):
        super(FeatureGen, self).__init__()
        self.data = data
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose > 0: self.progress = tqdm(total=len(self), desc='Features')

    def __getitem__(self, index):
        start = self.batch_size * index
        size = min(len(self.data) - start, self.batch_size)
        a = np.zeros((size,) + img_shape, dtype=K.floatx())
        for i in range(size): a[i, :, :, :] = read_for_validation(self.data[start + i])
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return a

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size


class ScoreGen(Sequence):
    def __init__(self, x, y=None, batch_size=2048, verbose=1):
        super(ScoreGen, self).__init__()
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.verbose = verbose
        if y is None:
            self.y = self.x
            self.ix, self.iy = np.triu_indices(x.shape[0], 1)
        else:
            self.iy, self.ix = np.indices((y.shape[0], x.shape[0]))
            self.ix = self.ix.reshape((self.ix.size,))
            self.iy = self.iy.reshape((self.iy.size,))
        self.subbatch = (len(self.x) + self.batch_size - 1) // self.batch_size
        if self.verbose > 0:
            self.progress = tqdm(total=len(self), desc='Scores')

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.ix))
        a = self.y[self.iy[start:end], :]
        b = self.x[self.ix[start:end], :]
        if self.verbose > 0:
            self.progress.update()
            if self.progress.n >= len(self): self.progress.close()
        return [a, b]

    def __len__(self):
        return (len(self.ix) + self.batch_size - 1) // self.batch_size

def set_sess_cfg():
    config_sess = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config_sess.gpu_options.allow_growth = True
    sess = tf.Session(config=config_sess)
    K.set_session(sess)

def expand_path(p):
    if isfile(TRAIN + p):
        return TRAIN + p
    if isfile(TEST + p):
        return TEST + p
    return p

def match(h1, h2):
    for p1 in h2ps[h1]:
        for p2 in h2ps[h2]:
            i1 = pil_image.open(expand_path(p1))
            i2 = pil_image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size: return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            a1 = a1 / sqrt((a1 ** 2).mean())
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            a2 = a2 / sqrt((a2 ** 2).mean())
            a = ((a1 - a2) ** 2).mean()
            if a > 0.1: return False
    return True


def show_whale(imgs, per_row=2):
    n = len(imgs)
    rows = (n + per_row - 1) // per_row
    cols = min(per_row, n)
    fig, axes = plt.subplots(rows, cols, figsize=(24 // per_row * cols, 24 // per_row * rows))
    for ax in axes.flatten(): ax.axis('off')
    for i, (img, ax) in enumerate(zip(imgs, axes.flatten())): ax.imshow(img.convert('RGB'))


def read_raw_image(p):
    img = pil_image.open(expand_path(p))
    return img

def prefer(ps):
    if len(ps) == 1: return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0] * s[1] > best_s[0] * best_s[1]:  # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p

def build_transform(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    """
    Build a transformation matrix with the specified characteristics.
    """
    rotation = np.deg2rad(rotation)
    shear = np.deg2rad(shear)
    rotation_matrix = np.array(
        [[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
    # shift_matrix = np.array([[1, 0, height_shift], [0, 1, width_shift], [0, 0, 1]])
    shear_matrix = np.array([[1, np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])
    zoom_matrix = np.array([[1.0 / height_zoom, 0, 0], [0, 1.0 / width_zoom, 0], [0, 0, 1]])
    shift_matrix = np.array([[1, 0, -height_shift], [0, 1, -width_shift], [0, 0, 1]])
    return np.dot(np.dot(rotation_matrix, shear_matrix), np.dot(zoom_matrix, shift_matrix))

def read_cropped_image(p, augment):
    """
    @param p : the name of the picture to read
    @param augment: True/False if data augmentation should be performed
    @return a numpy array with the transformed image
    """
    # If an image id was given, convert to filename
    if p in h2p:
        p = h2p[p]
    size_x, size_y = p2size[p]

    # Determine the region of the original image we want to capture based on the bounding box.
    row = p2bb.loc[p]
    x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
    dx = x1 - x0
    dy = y1 - y0
    x0 = max(0, x0 - dx * crop_margin)
    x1 = min(size_x, x1 + dx * crop_margin + 1)
    y0 = max(0, y0 - dy * crop_margin)
    y1 = min(size_y, y1 + dy * crop_margin + 1)

    # Generate the transformation matrix
    trans = np.array([[1, 0, -0.5 * img_shape[0]], [0, 1, -0.5 * img_shape[1]], [0, 0, 1]])
    trans = np.dot(np.array([[(y1 - y0) / img_shape[0], 0, 0], [0, (x1 - x0) / img_shape[1], 0], [0, 0, 1]]), trans)
    if augment:
        trans = np.dot(build_transform(
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            random.uniform(0.8, 1.0),
            random.uniform(0.8, 1.0),
            random.uniform(-0.05 * (y1 - y0), 0.05 * (y1 - y0)),
            random.uniform(-0.05 * (x1 - x0), 0.05 * (x1 - x0))
        ), trans)
    trans = np.dot(np.array([[1, 0, 0.5 * (y1 + y0)], [0, 1, 0.5 * (x1 + x0)], [0, 0, 1]]), trans)

    # Read the image, transform to black and white and comvert to numpy array
    img = read_raw_image(p).convert('L')
    img = img_to_array(img)

    # Apply affine transformation
    matrix = trans[:2, :2]
    offset = trans[:2, 2]
    img = img.reshape(img.shape[:-1])
    img = affine_transform(img, matrix, offset, output_shape=img_shape[:-1], order=1, mode='constant',
                           cval=np.average(img))
    img = img.reshape(img_shape)

    # Normalize to zero mean and unit variance
    img -= np.mean(img, keepdims=True)
    img /= np.std(img, keepdims=True) + K.epsilon()
    return img

def read_for_training(p):
    """
    Read and preprocess an image with data augmentation (random transform).
    """
    return read_cropped_image(p, True)


def read_for_validation(p):
    """
    Read and preprocess an image without data augmentation (use for testing).
    """
    return read_cropped_image(p, False)

def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return K.get_value(model.optimizer.lr)


def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m


def compute_score(verbose=1):
    """
    Compute the score matrix by scoring every pictures from the training set against every other picture O(n^2).
    """
    features = branch_model.predict_generator(FeatureGen(train, verbose=verbose), max_queue_size=max_queue_size, workers=workers,
                                              verbose=0)
    score = head_model.predict_generator(ScoreGen(features, verbose=verbose), max_queue_size=max_queue_size, workers=workers, verbose=0)
    score = score_reshape(score, features)
    return features, score


def make_steps(step, ampl):
    """
    Perform training epochs
    @param step Number of epochs to perform
    @param ampl the K, the randomized component of the score matrix.
    """
    global w2ts, t2i, steps, features, score, histories

    # shuffle the training pictures
    random.shuffle(train)

    # Map whale id to the list of associated training picture hash value
    w2ts = {}
    for w, hs in w2hs.items():
        for h in hs:
            if h in train_set:
                if w not in w2ts: w2ts[w] = []
                if h not in w2ts[w]: w2ts[w].append(h)
    for w, ts in w2ts.items(): w2ts[w] = np.array(ts)

    # Map training picture hash value to index in 'train' array
    t2i = {}
    for i, t in enumerate(train): t2i[t] = i

    # Compute the match score for each picture pair
    features, score = compute_score()

    csv_logger = CSVLogger(os.path.join(history_dir, f'trained_{steps + step}.csv'))

    output_weights_path = os.path.join(models_dir, 'model_finetuning.h5')
    checkpoint = ModelCheckpoint(
        output_weights_path,
        save_weights_only=True,
        save_best_only=False,
        verbose=1,
    )
    # gpus = len(os.getenv("CUDA_VISIBLE_DEVICES", "0,1").split(","))
    # if gpus > 1:
    #     print(f"** multi_gpu_model is used! gpus={gpus} **")
    #     model_multi_GPU = multi_gpu_model(model, gpus)
    #     # FIXME: currently (Keras 2.1.2) checkpoint doesn't work with multi_gpu_model
    #     checkpoint = MultiGPUModelCheckpoint(
    #         filepath=output_weights_path,
    #         base_model=model,
    #         save_best_only=False,
    #         save_weights_only=True,
    #     )
    # else:
    #     model_multi_GPU = model
    #     checkpoint = ModelCheckpoint(
    #         output_weights_path,
    #         # save_weights_only=True,
    #         save_best_only=False,
    #         verbose=1,
    #     )
    # model_multi_GPU.compile(Adam(lr=64e-5), loss=focal_loss(gamma=2., alpha=.5), metrics=['binary_crossentropy', 'acc'])
    # model_multi_GPU.compile(Adam(lr=64e-5), loss='binary_crossentropy', metrics=['binary_crossentropy', 'acc'])

    callbacks = [
        csv_logger,
        checkpoint,
        TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
    ]

    # Train the model for 'step' epochs
    history = model.fit_generator(
        TrainingData(score + ampl * np.random.random_sample(size=score.shape), steps=step, batch_size=batch_size),
        initial_epoch=steps,
        epochs=steps + step,
        max_queue_size=max_queue_size,
        workers=workers,
        verbose=1,
        callbacks=callbacks,).history
    steps += step

    # Collect history data
    history['epochs'] = steps
    history['ms'] = np.mean(score)
    history['lr'] = get_lr(model)
    print(history['epochs'], history['lr'], history['ms'])
    histories.append(history)

def prepare_submission(threshold, filename):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to 'new_whale'
    @param filename the submission file name
    """
    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    with open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i, p in enumerate(tqdm(submit)):
            t = []
            s = set()
            a = score[i, :]
            for j in list(reversed(np.argsort(a))):
                h = known[j]
                if a[j] < threshold and new_whale not in s:
                    pos[len(t)] += 1
                    s.add(new_whale)
                    t.append(new_whale)
                    if len(t) == 5: break;
                for w in h2ws[h]:
                    assert w != new_whale
                    if w not in s:
                        if a[j] > 1.0:
                            vtop += 1
                        elif a[j] >= threshold:
                            vhigh += 1
                        s.add(w)
                        t.append(w)
                        if len(t) == 5: break;
                if len(t) == 5: break;
            if new_whale not in s: pos[5] += 1
            assert len(t) == 5 and len(s) == 5
            f.write(p + ',' + ' '.join(t[:5]) + '\n')
    return vtop, vhigh, pos

set_sess_cfg()
models_dir = os.path.join(output_dir, 'models')
if not os.path.isdir(models_dir):
    os.makedirs(models_dir)

history_dir = os.path.join(output_dir, 'history')
if not os.path.exists(history_dir):
    os.mkdir(history_dir)

if isfile(P2SIZE):
    print("P2SIZE exists.")
    with open(P2SIZE, 'rb') as f:
        p2size = pickle.load(f)
else:
    p2size = {}
    for p in tqdm(join):
        size = pil_image.open(expand_path(p)).size
        p2size[p] = size

if isfile(P2H):
    print("P2H exists.")
    with open(P2H, 'rb') as f:
        p2h = pickle.load(f)
else:
    # Compute phash for each image in the training and test set.
    p2h = {}
    for p in tqdm(join):
        img = pil_image.open(expand_path(p))
        h = phash(img)
        p2h[p] = h

    # Find all images associated with a given phash value.
    h2ps = {}
    for p, h in p2h.items():
        if h not in h2ps: h2ps[h] = []
        if p not in h2ps[h]: h2ps[h].append(p)

    # Find all distinct phash values
    hs = list(h2ps.keys())

    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    h2h = {}
    for i, h1 in enumerate(tqdm(hs)):
        for h2 in hs[:i]:
            if h1 - h2 <= 6 and match(h1, h2):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2: s1, s2 = s2, s1
                h2h[s1] = s2

    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for p, h in p2h.items():
        h = str(h)
        if h in h2h: h = h2h[h]
        p2h[p] = h
#     with open(P2H, 'wb') as f:
#         pickle.dump(p2h, f)
# For each image id, determine the list of pictures
h2ps = {}
for p, h in p2h.items():
    if h not in h2ps: h2ps[h] = []
    if p not in h2ps[h]: h2ps[h].append(p)

h2p = {}
for h, ps in h2ps.items():
    h2p[h] = prefer(ps)
# len(h2p), list(h2p.items())[:5]

p2bb = pd.read_csv(BB_DF).set_index("Image")
old_stderr = sys.stderr
sys.stderr = open('/dev/null' if platform.system() != 'Windows' else 'nul', 'w')
sys.stderr = old_stderr

# p = list(tagged.keys())[312]

model, branch_model, head_model = build_model(lr=64e-5, l2=0, img_shape=img_shape)
model.summary()
h2ws = {}
new_whale = 'new_whale'
for p, w in tagged.items():
    if w != new_whale:  # Use only identified whales
        h = p2h[p]
        if h not in h2ws: h2ws[h] = []
        if w not in h2ws[h]: h2ws[h].append(w)
for h, ws in h2ws.items():
    if len(ws) > 1:
        h2ws[h] = sorted(ws)

# For each whale, find the unambiguous images ids.
w2hs = {}
for h, ws in h2ws.items():
    if len(ws) == 1:  # Use only unambiguous pictures
        w = ws[0]
        if w not in w2hs: w2hs[w] = []
        if h not in w2hs[w]: w2hs[w].append(h)
for w, hs in w2hs.items():
    if len(hs) > 1:
        w2hs[w] = sorted(hs)

train = []  # A list of training image ids
for hs in w2hs.values():
    if len(hs) > 1:
        train += hs
random.shuffle(train)
train_set = set(train)

w2ts = {}  # Associate the image ids from train to each whale id.
for w, hs in w2hs.items():
    for h in hs:
        if h in train_set:
            if w not in w2ts:
                w2ts[w] = []
            if h not in w2ts[w]:
                w2ts[w].append(h)
for w, ts in w2ts.items():
    w2ts[w] = np.array(ts)

t2i = {}  # The position in train of each training image id
for i, t in enumerate(train):
    t2i[t] = i

# Test on a batch of 32 with random costs.
# score = np.random.random_sample(size=(len(train), len(train)))
# data = TrainingData(score, batch_size=128)
# (a, b), c = data[0]

histories = []
steps = 0

if isfile(MPIOTTE_STANDARD_MODEL):
    model.load_weights(MPIOTTE_STANDARD_MODEL)

# epoch -> 10
set_lr(model, 16e-5)
for _ in range(2): make_steps(5, 0.25)
model.save(os.path.join(models_dir, 'model_finetuning_epoch10.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch10.h5'))

# epoch -> 20
set_lr(model, 4e-5)
for _ in range(2): make_steps(5, 0.2)
model.save(os.path.join(models_dir, 'model_finetuning_epoch20.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch20.h5'))

# epoch -> 30
set_lr(model, 1e-5)
for _ in range(2): make_steps(5, 0.15)
model.save(os.path.join(models_dir, 'model_finetuning_epoch30.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch30.h5'))

# epoch -> 40
set_lr(model, 4e-6)
for _ in range(2): make_steps(5, 0.1)
model.save(os.path.join(models_dir, 'model_finetuning_epoch40.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch40.h5'))

# epoch -> 50
set_lr(model, 1e-6)
for _ in range(2): make_steps(5, 0.05)
model.save(os.path.join(models_dir, 'model_finetuning_epoch50.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch50.h5'))

# epoch -> 60
set_lr(model, 8e-7)
for _ in range(2): make_steps(5, 0.02)
model.save(os.path.join(models_dir, 'model_finetuning_epoch60.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch60.h5'))

# epoch -> 70
set_lr(model, 4e-7)
for _ in range(2): make_steps(5, 0)
model.save(os.path.join(models_dir, 'model_finetuning_epoch70.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch70.h5'))

weights = model.get_weights()
model, branch_model, head_model = build_model(lr=1e-5, l2=0.0002, img_shape=img_shape)
model.set_weights(weights)
# epoch -> 80
for _ in range(2): make_steps(5, 1.0)
model.save(os.path.join(models_dir, 'model_finetuning_epoch80.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch80.h5'))

# epoch -> 90
set_lr(model, 8e-6)
for _ in range(2): make_steps(5, 0.5)
model.save(os.path.join(models_dir, 'model_finetuning_epoch90.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch90.h5'))

# epoch -> 100
set_lr(model, 4e-6)
for _ in range(2): make_steps(5, 0.25)
model.save(os.path.join(models_dir, 'model_finetuning_epoch100.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch100.h5'))

# epoch -> 110
set_lr(model, 1e-6)
for _ in range(2): make_steps(5, 0.2)
model.save(os.path.join(models_dir, 'model_finetuning_epoch110.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch110.h5'))

# epoch -> 120
set_lr(model, 8e-7)
for _ in range(2): make_steps(5, 0.15)
model.save(os.path.join(models_dir, 'model_finetuning_epoch120.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch120.h5'))

# epoch -> 130
set_lr(model, 4e-7)
for _ in range(2): make_steps(5, 0.1)
model.save(os.path.join(models_dir, 'model_finetuning_epoch130.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch130.h5'))

# epoch -> 140
set_lr(model, 2e-7)
for _ in range(2): make_steps(5, 0.05)
model.save(os.path.join(models_dir, 'model_finetuning_epoch140.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch140.h5'))

# epoch -> 150
set_lr(model, 1e-7)
for _ in range(2): make_steps(5, 0)
model.save(os.path.join(models_dir, 'model_finetuning_epoch150.h5'))
model.save_weights(os.path.join(models_dir, 'weights_finetuning_epoch150.h5'))
# Find elements from training sets not 'new_whale'
tic = time.time()
h2ws = {}
for p, w in tagged.items():
    if w != new_whale:  # Use only identified whales
        h = p2h[p]
        if h not in h2ws: h2ws[h] = []
        if w not in h2ws[h]: h2ws[h].append(w)
known = sorted(list(h2ws.keys()))

# Dictionary of picture indices
h2i = {}
for i, h in enumerate(known): h2i[h] = i

# Evaluate the model.
fknown = branch_model.predict_generator(FeatureGen(known), max_queue_size=max_queue_size, workers=workers, verbose=0)
fsubmit = branch_model.predict_generator(FeatureGen(submit), max_queue_size=max_queue_size, workers=workers, verbose=0)
score = head_model.predict_generator(ScoreGen(fknown, fsubmit), max_queue_size=max_queue_size, workers=workers, verbose=0)
score = score_reshape(score, fknown, fsubmit)

# Generate the subsmission file.
time_now = datetime.now()
submission_file = f"submit/submission_{time_now}.csv"
prepare_submission(0.99, submission_file)
toc = time.time()
print("Submission time: ", (toc - tic) / 60.)