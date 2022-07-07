import os
import time
from picamera import PiCamera
from FaceRecognition.FaceRecognitionImage import Face
from gtts import gTTS


"""
speech_write('Welcome to envision', 'welcome')
speech_write('Face Recognition Mode', 'face_mode')
speech_write('Object Detection Mode', 'object_mode')
speech_write('Text Reading Mode', 'text_mode')
"""


def speech_out(file):
	os.system('mpg321 /home/admin/PycharmProjects/Envision/Speech/' + file + '.mp3')


def text_to_speech(text,file):
	speech = gTTS(text=text, lang='en', slow=False)
	speech.save('/home/admin/PycharmProjects/Envision/Speech/' + file + '.mp3')


speech_out('welcome')
time.sleep(1)

camera = PiCamera()

output = 'Output'
face = Face()



while True:
	trigger = input("Enter Number: ")
	if trigger == '1':
		speech_out('face_mode')
		camera.capture("/home/admin/PycharmProjects/Envision/Input/input.jpeg")
		print("Image Captured")
		
		output = face.recognize_face()
		if output == 'Unknown':
			output = 'Unable to recognize the face'
		else:
			output = 'Human recognized as ' + output
	elif trigger == '2':
		speech_out('object_mode')
	
	elif trigger == '3':
		speech_out('text_mode')
	
	print(output)
	text_to_speech(output, 'output')
	speech_out('output')
