
import face_recognition
import pickle
import cv2
import os
from Face_Crop import Face_crop
from Data_Augmentation import augment_data

"""
# Cropping images to get only face part
Face_crop('/home/admin/PycharmProjects/Envision/FaceRecognition/Images/', '/home/admin/PycharmProjects/Envision/FaceRecognition/Cropped/')
"""
# Creating new images from cropped image
augment_data('/home/admin/PycharmProjects/Envision/FaceRecognition/Cropped/', '/home/admin/PycharmProjects/Envision/FaceRecognition/Dataset')

dir = '/home/admin/PycharmProjects/Envision/FaceRecognition/Dataset'
data = dict()
subdirs = [x[0] for x in os.walk(dir)]
print(subdirs)
for subdir in subdirs[1:]:
	knownEncodings = []
	print(subdir)
	name = subdir.split(os.path.sep)[-1]
	print(name)
	files = os.walk(subdir).__next__()[2]
	if (len(files) > 0):
		print(len(files))
		for file in files:
			# r.append(os.path.join(subdir, file))
			imagePath = os.path.join(subdir, file)
			# load the input image and convert it from BGR (OpenCV ordering)
			# to dlib ordering (RGB)
			image = cv2.imread(imagePath)
			rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			# Use Face_recognition to locate faces
			boxes = face_recognition.face_locations(rgb, model='hog')
			# compute the facial embedding for the face
			encodings = face_recognition.face_encodings(rgb, boxes)
			# loop over the encodings
			for encoding in encodings:
				knownEncodings.append(encoding)
		data[name] = knownEncodings

print(list(data.keys()))

# use pickle to save data into a file for later use
f = open("/home/admin/PycharmProjects/Envision/FaceRecognition/face_model_encode", "wb")
f.write(pickle.dumps(data))
f.close()

"""# Cropping images to get only face part
Face_crop('Images/', 'Cropped/')

# Creating new images from cropped image
augment_data('Cropped/', 'Dataset')


# get paths of each file in folder named Images
# Images here contains my data(folders of various persons)
imagePaths = list(paths.list_images('Dataset'))
knownEncodings = []
knownNames = []
print(len(imagePaths))
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    name = imagePath.split(os.path.sep)[-2]
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Use Face_recognition to locate faces
    boxes = face_recognition.face_locations(rgb, model='hog')
    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)
    # loop over the encodings
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

# save emcodings along with their names in dictionary data
data = {"encodings": knownEncodings, "names": knownNames}
print(len(data['encodings']))
print(len(data['names']))
# use pickle to save data into a file for later use
f = open("face_model_enc", "wb")
f.write(pickle.dumps(data))
f.close()"""