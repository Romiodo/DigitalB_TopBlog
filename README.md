# DigitalB_TopBlog
Solution for Digital Breakthrough

## Решение от команды UdiTeam

### Перед запуском решения, необходимо установить библиотеки
````
pip install -r requirements.txt
````


### Запуск решения в streamlit на локальной машине
````
streamlit run src/app.py
````

### Запуск решения в streamlit на google colab
````
npm install localtunnel
streamlit run src/app.py &>logs.txt & npx localtunnel \
--port 8501 & curl ipv4.icanhazip.com
````
