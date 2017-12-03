#### Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/determine_scales.png
[image4]: ./output_images/sliding window search 01.png
[image5]: ./output_images/sliding window search 02.png
[image6]: ./output_images/heatmap_frame1.png
[image7]: ./output_images/heatmap_frame2.png
[image8]: ./output_images/heatmap_frame3.png
[image9]: ./output_images/heatmap_frame4.png
[image10]: ./output_images/heatmap_frame5.png
[image11]: ./output_images/heatmap_frame6.png
[image12]: ./output_images/integrated_heatmap.png
[image13]: ./output_images/output_bboxes.png
[video1]: ./project_video_output.mp4

---
##### Histogram of Oriented Gradients (HOG)

###### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.
The code for this step is contained in the first code cell of the IPython notebook 
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I did some experiments in selecting HOG parameters and color spaces. The configurations are listed in the following tables. Conf 1 is selected by considering test accuracy and perdiction speed. 

|Config| Colorspace|Channel|Orientation|pix_per_cell|cell_per_block|spatial|histbin
|-|--------------|:-------------:|:------:|:----:|:--------:|:--------:|:--------:|
|1| YCrCb  | ALL | 9 | 8 | 2 | 32 | 32|
|2| YCrCb  | ALL | 9 | 12 | 2 | 32 | 32|
|3| YCrCb  | 0 | 9 | 8 | 2 | 32 | 32|
|4| YCrCb  | ALL | 9 | 8 | 2 | N/A| N/A|
|5| YCrCb  | ALL | 9 | 8 | 2 | 32 | 64|
|6| YCrCb  | ALL | 9 | 8 | 2 | 16 | 32|
|7| RGB |    ALL | 9 | 8 | 2 | 32 | 32|
|8| HLS |    ALL | 9 | 8 | 2 | 32 | 32|

|Config| Classifier|Test Accuracy|Prediction Speed (20 labels)|
|-|--------------|:-------------:|:------:|
|1| Linear SVM  | 0.9916 | 0.01489s |
|2| Linear SVM  | 0.9842 | 0.03788  |
|2| Linear SVM  | 0.9738 | 0.00305s |
|3| Linear SVM  | 0.9778 | 0.01745s |
|4| Linear SVM  | 0.9865 | 0.01979s | 
|5| Linear SVM  | 0.9862 | 0.00446s | 
|6| Linear SVM  | 0.9792 | 0.00621s |
|6| Linear SVM  | 0.9907 | 0.00868s |
|7| Linear SVM  | 0.9907 | 0.00427s |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM with the default classifier parameters and using HOG features togehter with spatial intensity and channel intensity histogram features and was able to achieve a test accuracy > 99%.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I adapted the method find_cars from the lesson materials. The method combines multiple features (spatial, color histgram and HOG) extraction with a sliding window search, but rather than perform feature extraction on each window individually which can be time consuming, the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window (scaling based on original size) and then fed to the classifier. The method performs the classifier prediction on the features for each window region and returns a list of rectangle objects corresponding to the windows that generated a positive ("car") prediction.
To determine what scale and overlap widnows to use, example images are used to test out what parameters will give good vehicle detections, the following image showed identified matching windows by swiping scaling factors from 0.5, up to 2.5 of the original window size.

![alt text][image3]

As illustated in the image, 0.5 detected the most window regions but many false detections among them to make filtering much harder. 2.5, on the other side, generated no detections at all neither far nor near cars. 
1.5 and 2 can both detect far and near vehicles so they are choosen to be the search scales. Considering pipeline execution speed, scale 1 was excluded because it also tended to generate considerable false positives. 
Overlap window size is set to be 75% (2 cell_per_step) to blance between speed and accuracy.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image5]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap to identify vehicle positions. The hotarea in the map normally identified where the vehicles are. However there are also some sparoidcally distributed false detections. In order to effectively filter them out, rather than thresholding individual frame's heatmap, an integrated heatmaps were generated by adding up 6 consecutive frames' heatmap. Assuming false detections are not showing up consistently as the true detections, this made the true detections more distinguishable from the false ones so that an apporiate threshold can be applied to remove the false positives. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. Assuming each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image12]

### Here are the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image13]

---
### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main challenge I am facing here was how to effectively filter out false positive detections while not sacrafying true positive detections. Cars appeared in distance or in complex lighting conditions might be difficult for the classifier to sepearte them from some not-car images. For example, my pipe line had trouble making robust detection during project video 41-43 seconds. The interleaved shades and lights on the left road side confused the classifier if I apply a low threshold on the detection heatmaps. However, if the threshold was set too high, the ture detection for the car running on upper right hand will be filtered out briefly which caused wobbly bounding box tracking on that car. The submitted video showed the reasonable trade-offs made on the threshold value. The false positives were limited on the far left (outside the left yellow line), I could potentially apply the techniques developped for the last project -- the advanced lane line finding project to further filter those false positives out or marked them as harmless objects. For the wobbly true car bounding boxes, I could develop more sophisticated car position estimation algorithm to recover the lost true detections because the car's velocity can be derived from availiable detections and its acceleartion should be within a reasonable range.

