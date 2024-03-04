import io
import json
import requests
import streamlit as st
import torch
import PIL.Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from spire.pdf.common import *
from spire.pdf import *
import numpy as np
import re
import pytesseract
from PIL import Image
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import nltk
import aspose.slides as slides
import aspose.pydrawing as drawing

nltk.download('punkt')
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def extract_images_from_pdf(pdf_path):
    try:
        doc = PdfDocument()
        doc.LoadFromFile(pdf_path)
        images = []
        index=0
        for i in range(doc.Pages.Count):
            print(i)
            page = doc.Pages.get_Item(i)
            for image in page.ExtractImages():
                images.append(image)
                imageFileName = 'Image-{0:d}.png'.format(index)
                print(index)
                index += 1
                image.Save(imageFileName, ImageFormat.get_Png())
        print(images)
        st.write("Total Extractd Imagies: ",len(images))
        doc.Close()
        return len(images)
    except Exception as e:
        st.error(f"Error occurred while extracting images from PDF: {e}")
        return []

def get_image_format(image_type):
    return {
        "jpeg": drawing.imaging.ImageFormat.jpeg,
        "png": drawing.imaging.ImageFormat.png,
    }.get(image_type, None)

def extract_images_from_ppt(pdf_path):
    imageIndex = 1

    with slides.Presentation(pdf_path) as pres:
        for image in pres.images:
            file_name = "Image_{0}.{1}"
            image_type = image.content_type.split("/")[1].lower()
            image_format = get_image_format(image_type)
            if image_format:
                image.system_image.save(file_name.format(imageIndex, image_type), image_format)
                imageIndex += 1
        st.write("Total Extractd Imagies: ",imageIndex)
    return imageIndex-1

def predict_step(images):
    processed_images = []
    for image in images:
        resized_image = image.resize((224, 224))  
        np_image = np.array(resized_image)
        if np_image.shape[-1] != 3:
            np_image = np_image[..., :3]  
        processed_images.append(np_image)

    processed_images = [PIL.Image.fromarray(img) for img in processed_images]

    pixel_values = feature_extractor(images=processed_images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


st.title("Image Captioning")

option = st.radio("Choose an option:", ("Extract images from PDF","Extract images from PPT"))

if option == "Upload an image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        result = predict_step([image])
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("### Predicted Caption:")
        st.write(result[0])

elif option == "Extract images from PDF":
    pdf_file = st.file_uploader("Upload a PDF file...", type=["pdf"])
    
    if pdf_file is not None:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getvalue())

        pdf_images = extract_images_from_pdf("temp.pdf")

        os.remove("temp.pdf")

        for i in range(pdf_images):
            image = PIL.Image.open(f'Image-{i}.png')
            cap=predict_step([image])
            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
            extracted_text = pytesseract.image_to_string(image)

            clean_text = re.sub(r'[^\w\s]', '', extracted_text)
            clean_text = re.sub(r'\d+', '', clean_text)

            sentences = sent_tokenize(clean_text)
            
            nouns = []
            verbs = []
            adjectives = []
            print(sentences)
            if len(sentences)!=0:
                for sentence in sentences:
                    words = word_tokenize(sentence)
                    try:
                        tagged_words = pos_tag(words)
                        for word, pos_tag in tagged_words:
                            if pos_tag.startswith('NN'):
                                nouns.append(word)
                            elif pos_tag.startswith('VB'):  
                                verbs.append(word)
                            elif pos_tag.startswith('JJ'):  
                                adjectives.append(word)

                        paragraph = f"The extracted text from the image appears to contain {len(sentences)} sentences. " \
                                    f"It discusses various topics, including {', '.join(set(nouns))} and actions such as " \
                                    f"{', '.join(set(verbs))}. The text also provides descriptive details using " \
                                    f"{', '.join(set(adjectives))} adjectives."

                        print(paragraph)
                    except Exception as e:
                        paragraph=""
            else:
                paragraph=""
            st.image(image, caption=f'{cap},"\n",{paragraph}')
elif option == "Extract images from PPT":
    pdf_file = st.file_uploader("Upload a PPT file...", type=["pptx"])
    
    if pdf_file is not None:
        with open("temp.pptx", "wb") as f:
            f.write(pdf_file.getvalue())

        pdf_images = extract_images_from_ppt("temp.pptx")

        os.remove("temp.pptx")

        for i in range(pdf_images):
            try:
                image = PIL.Image.open(f'Image_{i+1}.jpeg')
            except Exception as e:
                image = PIL.Image.open(f'Image_{i+1}.png')
            cap=predict_step([image])
            pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
            extracted_text = pytesseract.image_to_string(image)

            clean_text = re.sub(r'[^\w\s]', '', extracted_text)
            clean_text = re.sub(r'\d+', '', clean_text)

            sentences = sent_tokenize(clean_text)
            
            nouns = []
            verbs = []
            adjectives = []
            print(sentences)
            if len(sentences)!=0:
                for sentence in sentences:
                    words = word_tokenize(sentence)
                    try:
                        tagged_words = pos_tag(words)
                        for word, pos_tag in tagged_words:
                            if pos_tag.startswith('NN'):
                                nouns.append(word)
                            elif pos_tag.startswith('VB'):  
                                verbs.append(word)
                            elif pos_tag.startswith('JJ'):  
                                adjectives.append(word)

                        paragraph = f"The extracted text from the image appears to contain {len(sentences)} sentences. " \
                                    f"It discusses various topics, including {', '.join(set(nouns))} and actions such as " \
                                    f"{', '.join(set(verbs))}. The text also provides descriptive details using " \
                                    f"{', '.join(set(adjectives))} adjectives."

                        print(paragraph)
                    except Exception as e:
                        paragraph=""
            else:
                paragraph=""
            st.image(image, caption=f'{cap},"\n",{paragraph}')