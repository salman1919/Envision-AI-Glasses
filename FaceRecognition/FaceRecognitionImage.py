import pickle
import cv2
import face_recognition


class Face:
	
	def __init__(self):
		self.data = pickle.loads(open('/home/admin/PycharmProjects/Envision/FaceRecognition/face_model_encode', "rb").read())
		self.face_cascade = cv2.CascadeClassifier("/home/admin/PycharmProjects/Envision/FaceRecognition/cascade/haarcascade_frontalface_default.xml")
		print("OK")
	
	def recognize_face(self):
		image = cv2.imread('/home/admin/PycharmProjects/Envision/Input/input.jpeg')
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60),
		                                   flags=cv2.CASCADE_SCALE_IMAGE)
		
		# the facial embeddings for face in input
		encodings = face_recognition.face_encodings(rgb)
		names = []
		name = "Unknown"
		# loop over the facial embeddings incase
		# we have multiple embeddings for multiple faces
		for encoding in encodings:
			# Compare encodings with encodings in data["encodings"]
			# Matches contain array with boolean values and True for the embeddings it matches closely
			# and False for rest
			for k, v in self.data.items():
				matches = face_recognition.compare_faces(v, encoding)
				print(len(matches))
				# set name =unknown if no encoding matches
				false_count = 0
				for match in matches:
					if not match:
						false_count += 1
				print("Unmatched Count", false_count)
				# check to see if we have found a match
				if false_count < 5:
					name = k
					break
		
		names.append(name)
		
		""" loop over the recognized faces
		for ((x, y, w, h), name) in zip(faces, names):
			# rescale the face coordinates
			# draw the predicted face name on the image
			cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.line(image, (x + w // 2, y), (x + w // 2, y + h), (0, 0, 255), 2)
			center = (x + w // 2, y + h // 2)
			radius = 2
			cv2.circle(image, center, radius, (255, 255, 0), 2)
			cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
						0.75, (0, 255, 0), 2)
		cv2.imshow("Frame", image)
		cv2.waitKey(0)
		"""
		return name
