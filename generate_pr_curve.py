from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import tensorflow as tf
import os
import shutil
import matplotlib.image as mpimg
import numpy as np
from object_detection.utils import visualization_utils as vis_util

import io

from object_detection.utils import dataset_util
import glob

import time

##### Define variables #####

flags = tf.app.flags
flags.DEFINE_string('record_dir', '', 'Path to record directory')
flags.DEFINE_string('res_dir', '', 'Path to output folder where results will be saved')
flags.DEFINE_string('graphfile','','Path to frozen graph inference')
flags.DEFINE_string('gt_dir','','File directory where Groundtruth textfiles will be saved to')
flags.DEFINE_string('det_dir','','File directory where Detection textfiles will be saved to')

FLAGS = flags.FLAGS

graphfile = FLAGS.graphfile
det_dir = FLAGS.det_dir
gt_dir = FLAGS.gt_dir
res_dir = FLAGS.res_dir
record_dir = FLAGS.record_dir

record_list = glob.glob(record_dir+'/*.record')

#%matplotlib inline
class TFRecordExtractor:
    def __init__(self, tfrecord_file, graphfile):
        
        self.graphfile = graphfile
        
        # self.tfrecord_file = os.path.abspath(tfrecord_file)
        self.images = []
#         self.images_ = 
        self.records = []
        self.labels = []
        self.predictions = []
        self.scores=[]
        self.bboxes = []
        self.results = []
        self.filenames = []
        self.annotations = []
        self.groundtruths = {}
        self.time = 0
        self.num_time_rev = 0

    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation

        features={
            
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label':tf.io.VarLenFeature(tf.int64),
        }
        
        # Extract the data record
        sample = tf.parse_single_example(tfrecord, features)
#         print(sample['image/encoded'])
        image = tf.image.decode_image(sample['image/encoded'])
#         print(image)
        # image is Tensor with type dtype and shape [height, width, num_channels] for BMP, JPEG, and PNG images
        img_shape = tf.stack([sample['image/height'], sample['image/width'], 3]) # 3 channels assumed
        
        objects = {
                   'filename':sample['image/filename'],
                   'width':sample['image/width'],
                   'height':sample['image/height'],
                   'format':sample['image/format'],
                   'source_id': sample['image/source_id'],
                   'encoded': sample['image/encoded'],
                   'labels':sample['image/object/class/label'],
                   'xmins':sample['image/object/bbox/xmin'],
                   'ymins':sample['image/object/bbox/ymin'],
                   'xmaxs':sample['image/object/bbox/xmax'],
                   'ymaxs':sample['image/object/bbox/ymax'],
                   'classes':sample['image/object/class/text'],
        }
                           
        filename = sample['image/filename']
        return [image, objects, filename, img_shape]

    def get_data(self, objects):
        #boxes: a numpy array of shape [N, 4]
        #classes: a numpy array of shape [N]. class indices are 1-based, match the keys in the label map.
        boxes, classes = [],[]
        for i in range(len(objects['labels'].indices)):
            classes.append(objects['labels'].values[i])
            ymin = objects['ymins'].values[i]
            xmin = objects['xmins'].values[i]
            ymax = objects['ymaxs'].values[i]
            xmax = objects['xmaxs'].values[i]
            coord = [ymin,xmin,ymax,xmax]
            boxes.append(coord)

            
        return np.asarray(boxes), np.asarray(classes)
    
    def decode_bytes(self, x):
            return x.decode("utf-8")
    
    def get_annotations(self,objects,filename): ## for each example
        lst = []
        groundtruths = []
        for i in range(len(objects['xmins'].values)):
            xmin, ymin, xmax, ymax = objects['xmins'].values[i], objects['ymins'].values[i], objects['xmaxs'].values[i], objects['ymaxs'].values[i]
            width = xmax - xmin
            height = ymax - ymin 
            annotation = {
                "id" : i, 
                "image_id" : filename, 
                "category_id" : objects['labels'].values[i], 
                "segmentation" : [], 
                "area" : height*width, 
                "bbox" : [xmin,ymin,width,height], 
                "iscrowd" : 0,
            }
#             print(ymin, ymax)
#             print(objects['classes'])
            groundtruths.append([objects['classes'].values[i].decode(),xmin, ymin, width, height])
            lst.append(annotation)
        self.groundtruths[filename.decode()]=groundtruths
        return lst
    
#     def get_groundtruths(self,objects,filename):
        
        
    def extract_data(self):
        # Create folder to store extracted images
    #     folder_path = FLAGS.output_dir
    # #         folder_path = './ExtractedImages'
    #     shutil.rmtree(folder_path, ignore_errors = True)
    #     os.mkdir(folder_path)

        # Pipeline of dataset and iterator 
#         print('hi')
        while len(record_list)>0:
            print("enter record list loop")
            tf_rec = record_list.pop()
            print(tf_rec)
            tf_rec = os.path.abspath(tf_rec)
#             dataset = tf.data.TFRecordDataset([self.tfrecord_file])
            dataset = tf.data.TFRecordDataset([tf_rec])
            dataset = dataset.map(self._extract_fn)
            iterator = dataset.make_one_shot_iterator()
            next_record_data = iterator.get_next()

            def decode_bytes(x):
                return x.decode("utf-8")
            size = 0
            batch = []
            batch_label = []
            files = []
            with tf.Session() as sess:

                try:
    #                 Keep extracting data till TFRecord is exhausted
                    print("begin extraction")
                    while True:
        #                 while len(self.images[0])<3:
        #                     if size>=32:
#                         print('append image')

                        self.labels.append(batch_label)
                        size+=1
        #                 print(size)

                        record_data = sess.run(next_record_data)
                        self.images.append(record_data[0])
                        batch.append(record_data[0])
    #                         print(record_data[2],  record_data[3])
                        batch_label.append(record_data[1]['labels'])
                        self.filenames.append(record_data[2])
                        self.annotations.append(self.get_annotations(record_data[1], record_data[2]))
                #                 self.groundtruth.append([record_data[1]['xmins'],record_data[1]['ymaxs'], record_data[1]['xmaxs'], record_data[1]['ymins']])

                #                     print(len(self.images))
                #                     print(self.get_data(record_data[1]))

                except:
                        pass

        try:

            graph_file = self.graphfile
            print("Running graph_file from "+ graph_file)
            with tf.gfile.GFile(graph_file, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name='')

            with tf.Session(graph=graph) as sess:
                print('run graph sess')
                sess.run(tf.global_variables_initializer())
                input_node = 'image_tensor:0'
                output_nodes = ['detection_boxes:0', 'detection_scores:0', 'detection_classes:0']
    #                     self.records = []
    #                 for batch in self.images:
                print(len(self.images))
                count = 0
                time_100 = 0
                for batch in self.images:
                    # try:
    #                print(batch.shape)
    #                     print(batch)
                    if(batch.shape[-1]==1):
                        batch = cv2.cvtColor(batch,cv2.COLOR_GRAY2RGB)
                    batch_exp = np.expand_dims(batch, axis = 0)
#                         print(batch_exp.shape)
    #                 print('abou to run sess')
                    start = time.time()
                    output = sess.run(output_nodes, feed_dict={input_node: batch_exp})
                    end = time.time()
                    count += 1
                    time_100+=(end - start)
                    if(count == 100):
                        self.time+=time_100/100
                        self.num_time_rev += 1
                        count = 0
                        print(time_100/100)
                        time_100 = 0
    #                 output = sess.run(output_nodes, feed_dict={input_node: batch_exp})
    #                 print('here')
                    self.records.append(output)
    #                     print(output)
                    self.predictions.append(output[2])
                    self.scores.append(output[1])
                    
                    # except:
                    #     pass


        except:
            print("Error occurred during inference")
            pass



if __name__ == '__main__':
    t = TFRecordExtractor(record_list,os.path.join(os.getcwd(),graphfile))
    t.extract_data()


print("Number of filenames ", len(t.filenames))
print("Number of records ", len(t.records))
print("Number of groundtruth files ", len(t.groundtruths))

print('Average runtime per 100 frames: ', t.time/t.num_time_rev)
with open(res_dir,'w') as f:
    f.write('Average of average runtimes per frame: ' + str(t.time/t.num_time_rev))
######## Create Groundtruth (GT) text files ##########

for img in t.groundtruths:
    val = t.groundtruths[img]
#     img = str(img).strip('/')
#     print(img[-12:-4])
    # print(os.getcwd()+'/'+gt_dir+'/'+img[-12:-4]+'.txt')
    file = open(os.getcwd()+'/'+gt_dir+'/'+img[-12:-4]+'.txt','w') 
    for pred in val:
#         print(pred)
        pred = [str(x) for x in pred]
        file.write(" ".join(pred)+'\n')
    file.close()


def create_gt(boxes, scores, preds, filename): #create dic for 1 image
    #will have 100 predictions
    def map_pred(x):
#         if x==3.0 or x==6.0 or x==8.0:
#         print(x, int(x) == 1)
        if int(x) == 1:
            return 'plate'
        # elif int(x) == 2:
        #     return 'rear_vehicle'
        # elif int(x) == 3:
        #     return 'side_vehicle'
        else:
            # print(x)
            return 'others'
    
    lst = []
    if len(boxes)!=len(scores) or len(boxes)!=len(preds):
        print("Length of inputs do no match")
    
#     print(len(boxes))
    for i in range(len(boxes)):
        coords = boxes[i]
        ymin, xmin, ymax, xmax = coords[0], coords[1], coords[2], coords[3]
        width, height = xmax - xmin, ymax - ymin
#         print(width)
        if width == 0 or height == 0:
#             np.delete(scores,i)
            continue
        lst.append([map_pred(preds[i]), scores[i], xmin, ymin, width, height])
#     print(filename.decode())
    dic = {filename.decode(): lst}
    return dic


####### Create textfiles for predictions ######

print("Number of detection files ", len(t.records))
pred_list = []
count = 0
# print(t.filenames)
# for i in range(len(t.records)): #batch
#     for j in range(len(t.records[i][0])): #image
for i in range(len(t.records)):
#         boxes_, scores_, pred_, filenames_ = t.records[i][0][j], t.scores[i][j], t.predictions[i][j], t.filenames[i][j]
        boxes_, scores_, pred_, filenames_ = t.records[i][0], t.scores[i], t.predictions[i], t.filenames[i]

#         print(boxes_, scores_, pred_, filenames_)
        pred_list.append(create_gt(boxes_[0], scores_[0], pred_[0], filenames_))
        count += 1

print("Sample output of each detection file ", pred_list[2])

for img in pred_list:
#     print(img)
    for key in img:
        val = img[key]
#         print(key)
        file = open(os.getcwd()+'/'+det_dir+'/'+key[-12:-4]+'.txt','w')
        for pred in val:
#             print(pred)
            pred = [str(x) for x in pred]
            file.write(" ".join(pred)+'\n')
        file.close()


#### Run pascalvoc_v1.py script ####

import subprocess
## Assuming that pas
print("Attempting to run script to obtain PR Curve in ", FLAGS.res_dir)
# get_pr_curve =  subprocess.Popen('python Object-Detection-Metrics-master/pascalvoc_v1.py -gt ../'+gt_dir+ ' -det ../'+det_dir+' -sp ../'+ res_dir, shell=True, stdout=subprocess.PIPE).stdout
get_pr_curve =  subprocess.Popen('python Object-Detection-Metrics-master/pascalvoc_v1.py -gt '+gt_dir+ ' -det '+det_dir+' -sp '+ res_dir, shell=True, stdout=subprocess.PIPE).stdout

pr_curve =  get_pr_curve.read()

print(pr_curve.decode())
if pr_curve:
	print('Results can be found at '+res_dir)

print('Average runtime per 100 frames: ', t.time/t.num_time_rev)
