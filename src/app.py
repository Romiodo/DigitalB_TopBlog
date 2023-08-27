import streamlit as st
import nest_asyncio
import torch
from paddleocr import PaddleOCR
import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
from PIL import Image
import numpy as np
import zipfile
import os
import pandas as pd
import shutil

# Compile regex patterns once
pattern_1 = re.compile(r'^(?!\+)(\d+(\.\d+)?\s?%?)$')
pattern_2 = re.compile(r'(?<![+])\b(\d+)(?:\.(\d+))?(K)?\b')
pattern_3 = re.compile(r'(\d+,\d+)')

@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def extract_valid_entries_corrected(data, platform):
    """
    Extract entries from the given data based on platform and patterns.

    Parameters:
    - data (list): The list containing data entries.
    - platform (str): Platform type

    Returns:
    - list: A list of extracted entries.
    """
    
    extracted_entries = []
    
    if platform in ('dz', 'tg'):
        for entry in data:
            text = entry[1][0].replace(" ", "")
            if pattern_1.match(text):
                extracted_entries.append(entry[-1][0])
            else:
                extracted_entries.append(-1)

    elif platform == 'vk':
        sum_vk = 0
        for entry in data:
            match = pattern_2.search(entry[-1][0])
            if match:
                # Get the whole part, decimal part, and the 'K' multiplier if present
                whole, decimal, multiplier = match.groups()
                
                # If there's a decimal part, combine it with the whole part
                if decimal:
                    value = int(whole) * 1000 + int(decimal) * (1000 // (10 ** len(decimal)))
                else:
                    value = int(whole) * 1000 if multiplier and multiplier == 'K' else int(whole)
                    
                # Add the calculated value to the sum
                sum_vk += value
                
            else:
                sum_vk += 0  # Add 0 if no match
        extracted_entries.append(sum_vk)

    elif platform == 'yt_s':
        for entry in data:
            text = entry[-1][0].replace(' ', '')
            if pattern_1.match(text):
                extracted_entries.append(text)
            else:
                leading_digits_match = re.match(r'^\d+', text)
                if leading_digits_match:
                    extracted_entries.append(leading_digits_match.group())
                else:
                    extracted_entries.append(-1)

    elif platform == 'yt_v':
        for entry in data:
            text = entry[-1][0].replace(' ', '')
            if ('TbIC'.lower() in text.lower() or 'TblC'.lower() in text.lower()):
                match = pattern_3.search(text)
                value = float(match.group().replace(',', '.')) * 1000 if match else -1
                extracted_entries.append(str(int(value)) if match else -1)
            elif pattern_1.match(text):
                extracted_entries.append(text)
            else:
                leading_digits_match = re.match(r'^\d+', text)
                if leading_digits_match:
                    extracted_entries.append(leading_digits_match.group())
                else:
                    extracted_entries.append(-1)
        
    return [str(item).replace('%', '').replace(',', '.').replace(' ','') for item in extracted_entries if item != -1]

nest_asyncio.apply()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load("ultralytics/yolov5", 'custom', 'model/best.pt').to(DEVICE)
model.max_det = 2
ocr = PaddleOCR(lang='en')


executor = ThreadPoolExecutor(max_workers=4)

async def process_image_async(image_path, platform):
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(executor, model, image_path, 640)
    crops = results.crop(save=False)
    _dicts = [d for d in crops if d['label'].startswith(platform)]
    
    output = []
    for data in _dicts:
        result = await loop.run_in_executor(executor, ocr.ocr, data['im'])
        k = extract_valid_entries_corrected(result[0], platform=platform)
        if len(k) > 0:
            ans = float(k[0].replace(' ',''))
            output.append(ans)
    
    if len(output) == 0:
        return 'Метрика не была распознана'
    elif platform == 'vk':
        return sum(output)
    elif platform == 'tg':
        return max(output)
    elif platform in ['yt_s', 'yt_v', 'dz']:
        return output[0]
    else:
        return output


st.title("Async OCR App")

# Step 1: Add a radio button to choose the mode
mode = st.radio("Выберите режим:", ["Зип-архив", "Одно изображение"])

# Step 2: Update the file uploader based on the mode
if mode == "Зип-архив":
    uploaded_file = st.file_uploader("Выбран зип-архив...", type=['zip'])
elif mode == "Одно изображение":
    uploaded_file = st.file_uploader("Выбрана картинка..", type=['jpg', 'png', 'jpeg'])


platform_mapping = {
    "dz": 'Дзен',
    "tg": 'Телеграм',
    "vk": 'ВКонтакте',
    "yt_v": 'Ютуб-просмотры',
    'yt_s': 'Ютуб-подписчики'
}
platform_metric = {
    'tg': 'VR',
    'dz': 'Количество дочитываний',
    'vk': 'Количество подписчиков',
    'yt_v': 'Количество просмотров',
    'yt_s': 'Количество подписчиков'
}

selected_platform_fullname = st.selectbox("Выберите платформу:", list(platform_mapping.values()))

selected_platform = [key for key, value in platform_mapping.items() if value == selected_platform_fullname][0]


if uploaded_file:
    if mode == "Зип-архив":
        with open('tmp.zip',"wb") as f:
            f.write((uploaded_file).getbuffer())
        
        shutil.rmtree('tmp', ignore_errors=True)
        os.makedirs('tmp', exist_ok=True)
        with zipfile.ZipFile('tmp.zip', 'r') as zf:
            zf.extractall('tmp')
        
        names = []
        preds = []
        for path1 in os.listdir('tmp'):
            for path2 in os.listdir(f'tmp/{path1}'):
                    if path2[-4:] not in ['.jpg', '.png', '.jpeg']:
                        continue

                    image = Image.open(f'tmp/{path1}/{path2}')
                    image_array = np.array(image)
                    results = asyncio.run(process_image_async(image_array, selected_platform))

                    names.append(path2)
                    preds.append(results)
        
            
        df = pd.DataFrame({
            'image': names,
            platform_metric[selected_platform]: preds
        })
        csv = convert_to_csv(df)
        st.success("Файл успешно обработан, можете скачать архив")
        st.dataframe(df)

        st.download_button('Скачать csv отчет', csv, file_name='данные_метрик.csv')

    elif mode == "Одно изображение":
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        st.image(image_array, caption="Загруженное изображение", use_column_width=True)
        results = asyncio.run(process_image_async(image_array, selected_platform))
        st.write("Результат:", results)