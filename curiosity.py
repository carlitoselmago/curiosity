import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
#import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from threading import Thread
from time import sleep

class curiosity():

    difference_margin=0.8 #difference btween the sum of the two sides, if the value is greather than this, it would start moving to sides
    shock_margin=25 # if greater than this value, the algo will stop and watch
    procesimgsize=64#28#42
    saved_model_uri="tmp/saved_model"
    frame=None

    def __init__(self,savemodel=False):
        self.savemodel=savemodel
       
        self.ready=False
        print("curiosity NN init")

        #self.cap = cv2.VideoCapture(0, cv2.CAP_V4L)
        #self.cap = cv2.VideoCapture(0)

        self.c=0
        self.last_state="STOP"
        self.state="STOP"
        self.state_vals=[0,0]
        self.autoencoder=self.model()
        self.ready=True

        self.new_image=False

    def get_frames(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30) # we read the stream at 30 fps
     
        while cap.isOpened():
            ret, self.frame = cap.read()
            if not ret: break
            if self.preview:
                self.frame = self.paint_frame(self.frame)
                cv2.imshow('frame', self.frame)
                cv2.waitKey(1)
            
            #self.frame_queue.put(self.frame)

        cap.release()

    def preprocess_image(self,image, size=(96, 96)):
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        preprocessed_image = preprocess_input(resized_image)
        return preprocessed_image


    def predict_and_calculate_mse(self,image):
        try:
            decoded_image = self.autoencoder.predict(image,verbose=0)
            mse = np.mean((image - decoded_image) ** 2)
        except Exception as e:
            print("predict_and_calculate_mse EXCEPTION!!!:",e)
            print("")
            self.end()

        return mse

    def update_model_with_new_image(self,image, epochs=5):
        try:
            self.autoencoder.fit(image, image, epochs=epochs, verbose=0)
        except Exception as e:
            print("update_model_with_new_image EXCEPTION!!!:",e)
            print("")
            self.end()

    def model(self):
        if os.path.isfile(self.saved_model_uri) and self.savemodel:
            model = keras.models.load_model(self.saved_model_uri)
            return model
        else:
            input_img = Input(shape=(self.procesimgsize, self.procesimgsize, 1))

            params=32#16#32#16#32 # has to be bigger than 3
            size=6#3
            size2=4#2
            # Encoder
            x = Conv2D(params, (size, size), activation='relu', padding='same')(input_img)
            x = MaxPooling2D((size2, size2), padding='same')(x)
            x = Conv2D(params, (size, size), activation='relu', padding='same')(x)
            encoded = MaxPooling2D((size2, size2), padding='same')(x)

            # Decoder
            x = Conv2D(params, (size, size), activation='relu', padding='same')(encoded)
            x = UpSampling2D((size2, size2))(x)
            x = Conv2D(params, (size, size), activation='relu', padding='same')(x)
            x = UpSampling2D((size2, size2))(x)
            decoded = Conv2D(1, (size, size), activation='sigmoid', padding='same')(x)

            autoencoder = Model(input_img, decoded)

            autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

            return autoencoder




    def run_curisity(self):
        if self.frame:
            self.new_image=self.frame
            #ret,self.new_image = self.cap.read()
            #showimage=new_image.copy()
            #cv2.imshow('Webcam', showimage)

            # Preprocess the new image
            #img = self.preprocess_image(self.new_image)
            img=self.new_image
            # Predict and calculate the MSE
            mse = self.predict_and_calculate_mse(img)
        
            # Check for anomaly
            #print("ANOMALLY",mse_new_image)

            mse_converted=mse*1000

            numbers=("MSE:"+"{:.2f}".format(mse_converted))

            self.state_vals=[mse_converted] #left and right

            self.ST.curiosity_data=self.state_vals

            if (mse_converted)>self.shock_margin:
                #very important change
                self.state="STOP"

                print("STOP and watch::::: ",numbers)

                ##if (((mse_left_converted+mse_right_converted)>10) and (self.last_state!="STOP")):
                #if (self.last_state!="STOP" and self.state=="STOP") or ((mse_left_converted+mse_right_converted)>40):

                #    print("TOOK A PHOTO!!!! #"+str(self.c))
                #    cv2.imwrite("tmp/"+str(self.c)+".jpg", self.new_image)
                #    self.c+=1
                self.last_state="STOP"
           

            #boringness
        

            # Update the model with the new image
            self.update_model_with_new_image(img)
            self.c+=1
        else:
            sleep(0.1)
            print("no frame")

    def curiosity_process(self):
        while True:
            if self.ready:
                self.run_curisity()
            else:
                sleep(1)

    def start(self):
        tf = Thread(target=self.get_frames)
        tf.start()
        sleep(4)
        tc = Thread(target=self.curiosity_process)
        tc.start()

    def take_photo(self,filename):
        #cv2.imwrite("tmp/"+str(filename)+".jpg", self.new_image)
        cv2.imwrite("tmp/"+str(filename)+".jpg", self.new_image)


   
    def end(self):
        if self.savemodel:
            self.autoencoder.save(self.saved_model_uri)
            #self.cap.release()


if __name__ == "__main__":
    import cv2
    import numpy as np
    import sys

    #start cam thread
    
    curiosity=curiosity()
    #sleep(5)
    curiosity.start()
    #while True:
    """
    frame=ST.frame
    screen.fill([0,0,0])
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0,0))
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == KEYDOWN:
            sys.exit(0)

    """
    #cv2.imshow("video", ST.frame)
    #cv2.waitKey(1)# == ord('q'):
    #    break
    #self.preview_capture(self.frame.copy())
    #self.preview_capture(self.frame.copy())
    #curiosity.run_curisity()
    