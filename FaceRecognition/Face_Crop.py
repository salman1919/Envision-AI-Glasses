from matplotlib import pyplot
from mtcnn.mtcnn import MTCNN
from PIL import Image
import os


def Face_crop(read_path, save_path):
    ## change the folder name accordingly for training and testing
    detector = MTCNN()
    folders = os.listdir(read_path)
    print(folders)
    try:
        folders.remove('.DS_Store')
    except:
        print("No DS_Store created")

    ## Iterate over the folder and detect and crop faces and save them in respective folder
    for files in folders:
        try:
            pixels = pyplot.imread(read_path + files)
            faces = detector.detect_faces(pixels)
            x, y, width, height = faces[0]['box']
            # width = width + width/5
            # height = height + height/5
            Image.fromarray(pixels).crop((x, y, x + width, y + height)).save(save_path + files)
        except (IndexError or SystemError):
            print('Face Not Found')
