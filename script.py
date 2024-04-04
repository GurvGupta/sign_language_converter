# Streamlit library for making User Friendly interface.
import streamlit as st
# tkinter library for making interface of selecting video by browsing in local computer
from tkinter import *
import tkinter as tk
from tkinter import filedialog
# pytorch for loading the weights of model and loading the architecture of model and running the model
import torch
# numpy library for applying some processing on the arrays
import numpy as np
import torch.nn.functional as F
from pytorch_i3d import InceptionI3d
# opencv library for reading the video, frame by frame
import cv2
# kettotext library for making nlp pipeline
from keytotext import pipeline
# language.py file contains some functions for predicting some similar words
import language
# itertools for iterating over the function
from itertools import chain
# picklt library for loading the model file used for predicting similar words
import pickle
# translate library for conversion of English sentences to Arabic
from translate import Translator

# Setting the title of UI
st.title('Sign Language Recognition')

# Function for loading the video and reading it frame by frame, converting it to pytorch tensor and applying the run on tensor function for predicting output
def load_rgb_frames_from_video(video_path, confidence_threshold):
    # reading the video using the path 
    vidcap = cv2.VideoCapture(video_path)
    # Defining some local variables to be used further in the function
    frames = []
    offset = 0
    text = " "
    batch = 40
    text_list = []
    word_list = []
    sentence = ""
    text_count = 0
    
    # Now we are running a while loop until each frame of the video is read.
    while True:
        ret, frame1 = vidcap.read()
        offset = offset + 1
        font = cv2.FONT_HERSHEY_TRIPLEX
        
        if ret == True:
            w, h, c = frame1.shape
            sc = 224 / w
            sx = 224 / h
            frame = cv2.resize(frame1, dsize=(0, 0), fx=sx, fy=sc)
            frame1 = cv2.resize(frame1, dsize = (1280,720))    
            frame = (frame / 255.) * 2 - 1
            
            if offset > batch:
                frames.pop(0)
                frames.append(frame)
                
                if offset % 20 == 0:
                    # calling the run_on_tensor function
                    text = run_on_tensor(torch.from_numpy((np.asarray(frames, dtype=np.float32)).transpose([3, 0, 1, 2])), confidence_threshold)
                    if text != " ":
                        text_count = text_count + 1
                        
                        if bool(text_list) != False and bool(word_list) != False and text_list[-1] != text and word_list[-1] != text or bool(text_list) == False:
                            text_list.append(text)
                            word_list.append(text)
                            sentence = sentence + " " + text
                        
                        # getting similar word prediction by get_suggestions function of language.py file
                        word = language.get_suggestions(text_list, n_gram_counts_list, vocabulary, k = 1.0)
                        if(word != " ."):
                            sentence += word
                            text_list.append(word)
                        
                        if(text_count > 2):
                            sentence = nlp(text_list,**params)
                        
            else:
                frames.append(frame)
                if offset == batch:
                    text = run_on_tensor(torch.from_numpy((np.asarray(frames, dtype=np.float32)).transpose([3, 0, 1, 2])), confidence_threshold)
                    if text != " ":
                        text_count = text_count + 1
                        if bool(text_list) != False and bool(word_list) != False and text_list[-1] != text and word_list[-1] != text or bool(text_list) == False:
                            text_list.append(text)
                            word_list.append(text)
                            sentence = sentence + " " + text

                        word = language.get_suggestions(text_list, n_gram_counts_list, vocabulary, k = 1.0)
                        if(word != " ." ):
                            sentence += word
                            text_list.append(word)
                            
                        if(text_count > 2):
                            sentence = nlp(text_list,**params)
                        # cv2.putText(frame1, sentence, (120, 520), font, 0.9, (0, 255, 255), 2, cv2.LINE_4)
                            
            if len(text_list) > 10:
                text_list.pop()
                text_list.pop()
                text_list.pop()
            
        else:
            break
           
    vidcap.release()

    # In the last, we have appended all the predicted words into a sentence and returning it.
    return sentence
    

# a function to load the all the 3 models
def load_model(weights, num_classes, video_path, confidence_threshold):
    # defining a global variable i3d for creating object of InceptionI3d class for the model architecture
    global i3d 
    i3d = InceptionI3d(400, in_channels=3)
    i3d.replace_logits(num_classes)
    # loading the weight of the model here
    i3d.load_state_dict(torch.load(weights, map_location=torch.device('cpu'))) 
    # Setting the model to evaluation state
    i3d.eval()

    # defining a global variable nlp for creating keytotext pipeline on the cpu
    global nlp
    nlp = pipeline("k2t-new", use_cuda=False)
    global params
    params = {"do_sample":True, "num_beams": 5, "no_repeat_ngram_size":2, "early_stopping":True}
    
    # loading the nlp_data_processed pre-trained model here which is present in binary format in NLP folder
    with open("NLP/nlp_data_processed", "rb") as fp:  
        train_data_processed = pickle.load(fp)
    
    # loading the nlp_gram_counts pre-trained model here which is present in binary format in NLP folder
    global n_gram_counts_list
    with open("NLP/nlp_gram_counts", "rb") as fp:   
        n_gram_counts_list = pickle.load(fp)

    # creating a list, vocabulary from the train_data_processed
    global vocabulary
    vocabulary = list(set(chain.from_iterable(train_data_processed)))
    # calling the load_rgb_frames_from_video function with video_path as the parameter to get the predicted sentence
    sentence = load_rgb_frames_from_video(video_path, confidence_threshold)

    # here we are returning the sentence
    return sentence
    
# here is the function where model is predicting the output word on a frame of video
def run_on_tensor(ip_tensor, confidence_threshold):
    ip_tensor = ip_tensor[None, :]
    
    t = ip_tensor.shape[2] 
    per_frame_logits = i3d(ip_tensor)

    # calling the interpolate function of pytorch by giving the logits of frame
    predictions = F.interpolate(per_frame_logits, t, mode='linear')
    predictions = predictions.transpose(2, 1)
    # getting the output labels by applying the argsort function of numpy
    out_labels = np.argsort(predictions.detach().numpy()[0])
    # creating a output copy of predictions to be store in arr
    arr = predictions.detach().numpy()[0] 

    # print(float(max(F.softmax(torch.from_numpy(arr[0]), dim=0))))
    # print(wlasl_dict[out_labels[0][-1]])

    # applying a softmax function for getting the word which has output probability of more than 20%
    if max(F.softmax(torch.from_numpy(arr[0]), dim=0)) >= confidence_threshold:
        return wlasl_dict[out_labels[0][-1]]
    else:
        return " " 
        

# function for creating a dictionary of WLASAL dataset
def create_WLASL_dictionary():
    global wlasl_dict 
    wlasl_dict = {}
    
    with open('wlasl_class_list.txt') as file:
        for line in file:
            split_list = line.split()
            if len(split_list) != 2:
                key = int(split_list[0])
                value = split_list[1] + " " + split_list[2]
            else:
                key = int(split_list[0])
                value = split_list[1]
            wlasl_dict[key] = value
                

# function for selecting the video on User interface
def select_video():
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfiles(
        master=root,
        title="Select Video",
    )
    root.destroy()
    return video_path


# main function of script file
if __name__=="__main__":

    # getting the path of the video
    selected_video_path = st.session_state.get("video_path", None)
    # Creating a sidebar for options like model selection, conf and translation language
    st.sidebar.image("sign.jpg", width=250)
    with st.sidebar:
        
        select_model = st.selectbox("Model", ["Model1", "Model2"])
        confidence_threshold = st.slider(
            "Confidence Threshold", min_value=0.1,max_value=0.9,step=0.05
        )
        translate_to = st.selectbox("Translate To", ["English", "Arabic", "Spanish"])
        video_select_button = st.button("Select Video")

    # if path is Not None
    if video_select_button:
        selected_video_path = select_video()
        video_path = selected_video_path[0].name
        st.session_state.video_path = video_path
        # defining the video path in User Interface
        st.write(f'You selected {video_path}')
        # showing the video in User Interface
        st.video(video_path)

    answer = ""
    try:
        if video_path is not None:
            with st.spinner("Recognizing and Translating..."):

                mode = 'rgb'
                num_classes = 300
                if select_model == "Model1":
                    weights = 'model/sign_lan.pt'
                else:
                    num_classes = 1000
                    weights = "model/sign_lan_1000.pt"
                create_WLASL_dictionary()
                # calling the function to load the model
                answer = load_model(weights, num_classes, video_path, confidence_threshold)
                # Using Translator class of translate library for conversion to arabic language
                if len(answer) > 0:
                    translator = Translator(to_lang=translate_to)
                    translation = translator.translate(answer)
                # print(translation)
                # Writing the arabic output on the User Interface
            if len(translation) > 0:
                st.success("Translation")
                st.write(f"The video says(In {translate_to}): {translation}")
            else:
                
                st.write(f"Not able to Recognize, Try other model")
    except:
        pass

    
    
    
