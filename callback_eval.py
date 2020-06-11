import numpy as np
import keras
import keras.backend as K
from timeit import default_timer as timer

import skimage.io as io
import skimage.transform as transform
import os, glob, time

import cv2

from decode_np import Decode

def search_all_files_return_by_time_reversed(path, reverse=True):
    return sorted(glob.glob(os.path.join(path, '*.h5')), key=lambda x: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getctime(x))), reverse=reverse)

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        model_body=None,
        anchors=None,
        class_names=None,
        iou_threshold=0.45,
        score_threshold=0.5,
        max_boxes=450,
        tensorboard=None,
        weighted_average=False,
        eval_file='2007_val.txt',
        log_dir='logs/000/',
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.model_body      = model_body
        self.anchors         = anchors
        self.class_names     = class_names
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_boxes       = max_boxes
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.eval_file       = eval_file
        self.log_dir         = log_dir
        self.verbose         = verbose

        self.sess = K.get_session()

        # 验证时的分数阈值和nms_iou阈值
        conf_thresh = score_threshold
        nms_thresh = 0.45

        self._decode = Decode(conf_thresh, nms_thresh, (608,608), self.model_body, self.class_names)

        super(Evaluate, self).__init__()

    def calc_image(self, image, model_image_size=(608, 608)):
        start = timer()

        image, boxes, scores, classes = self._decode.detect_image(image, False)

        end = timer()
        #print(end - start)

        return boxes, scores, classes

    def calc_result(self, epoch):
        with open(self.eval_file) as f:
            lines = f.readlines()

        #np.random.shuffle(lines)

        result_file = open('eval_result_{}.txt'.format(epoch+1), 'w')
        count = 0
        for annotation_line in lines[:500]:
            #print(count)
            annotation = annotation_line.split()
            image = cv2.imread(annotation[0])
            out_boxes, out_scores, out_classes = self.calc_image(image)
            result_file.write(annotation[0] + ' ')
            if out_boxes is None:
                result_file.write('\n')
                count = count+1
                continue
            for i in range(len(out_boxes)):
                top, left, bottom, right = out_boxes[i]
                result_file.write(' ' + ','.join([str(left), str(top), str(right), str(bottom)]) + ',' + str(out_scores[i]) + ',' + str(out_classes[i]))
            result_file.write('\n')
            count = count+1

    def parse_rec(self, annotations):
        objects = []
        for obj in annotations:
            values = obj.split(',')
            obj_struct = {}
            obj_struct['name'] = values[4]
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [int(values[0]),
                                  int(values[1]),
                                  int(values[2]),
                                  int(values[3])]
            objects.append(obj_struct)

        return objects

    def voc_ap(self, rec, prec, use_07_metric=False):
        """Compute VOC AP given precision and recall. If use_07_metric is true, uses
        the VOC 07 11-point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


    def map_eval(self,
                 result_path,
                 anno_path,
                 classname,
                 ovthresh=0.5,
                 use_07_metric=False):
        """rec, prec, ap = voc_eval(detpath,
                                    annopath,
                                    imagesetfile,
                                    classname,
                                    [ovthresh],
                                    [use_07_metric])
        Top level function that does the PASCAL VOC evaluation.
        result_path: Path to detections
            detpath.format(classname) should produce the detection results file.
        anno_path: Path to annotations
            annopath.format(imagename) should be the xml annotations file.
        classname: Category name (duh)
        [ovthresh]: Overlap threshold (default = 0.5)
        [use_07_metric]: Whether to use VOC07's 11 point AP computation
            (default False)
        """
        # assumes detections are in detpath.format(classname)
        # assumes annotations are in annopath.format(imagename)

        # first load gt
        recs = {}
        imagenames = []
        with open(anno_path, 'r') as f:
            lines = f.readlines()
        for annotation_line in lines:
            annotation = annotation_line.split()
            imagename = annotation[0]
            imagenames.append(imagename)
            recs[imagename] = self.parse_rec(annotation[1:])

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}

        # read dets
        with open(result_path, 'r') as f:
            lines = f.readlines()

        image_ids = []
        confidence = []
        BB = []
        for result_line in lines:
            result = result_line.split()
            for obj in result[1:]:
                values = obj.split(',')
                if values[5] == classname:
                    image_ids.append(result[0])
                    confidence.append(float(values[4]))
                    BB.append([float(values[1]), float(values[0]), float(values[3]), float(values[2])])

        confidence = np.reshape(confidence, (len(image_ids)))
        BB = np.reshape(BB, (len(image_ids), 4))

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = self.voc_ap(rec, prec, use_07_metric)

        return rec, prec, ap, nd

    def on_epoch_end(self, epoch, logs=None):
        weight_latest = search_all_files_return_by_time_reversed(self.log_dir)[0]
        print("Epoch end eval mAP on weight {}".format(weight_latest))
        self.model_body.load_weights(weight_latest)
        self.calc_result(epoch)

        #计算mAP
        aps = []
        counts = []
        for classname in self.class_names:
            rec, prec, ap, count = self.map_eval('eval_result_{}.txt'.format(epoch+1), self.eval_file, classname)
            aps.append(ap)
            counts.append(count)

        aps = np.array(aps)
        counts = np.array(counts)

        mAP = np.sum(aps * counts) / np.sum(counts)

        print('Epoch {} mAP {}'.format(epoch+1, mAP))
