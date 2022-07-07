from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os


def augment_data(read_path, save_path):
    datagen = ImageDataGenerator(
            rotation_range = 40,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True,
            brightness_range = (0.5, 1.5))

    #SIZE = 224
    dataset = []
    my_images = os.listdir(read_path)
    try:
        my_images.remove('.DS_Store')
    except:
        print("No DS_Store created")
    for i, image_name in enumerate(my_images):
        image = cv2.imread(read_path + image_name)
        name = image_name.split('.')[0]
        image = image.reshape((1,) + image.shape)
        folder = os.path.join(save_path, name)
        try:
            os.makedirs(folder)
        except OSError as error:
            print(error)
        #x = np.array(dataset)
        j = 0
        for batch in datagen.flow(x=image, batch_size=16,
                                  save_to_dir= folder,
                                  save_prefix= name + '#',
                                  save_format='jpeg'):
            j += 1
            if j > 50:
                break

