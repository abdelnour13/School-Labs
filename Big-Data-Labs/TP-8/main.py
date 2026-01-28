import json
import os
import requests
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

ROOT_DIR = "/home/abdelnour/Documents/4eme_anne/S2/BDT/TP-2024/TP-8"
LAUNCES_FOLDER = os.path.join(ROOT_DIR,'tmp','launches.json')
IMAGES_FOLDER = os.path.join(ROOT_DIR,'tmp','images')

### DAG Creation

default_args = {
    'owner': 'tp8',
    'start_date': datetime(2024, 5, 10),
    'retries': 1
}

dag = DAG(
    dag_id='tp8',
    default_args=default_args,
    schedule_interval='@daily'
)

### Tasks

# 1- Download launches

download_launches = BashOperator(
    task_id="download_launches",
    bash_command=f'curl -Lk "https://ll.thespacedevs.com/2.0.0/launch/upcoming" > {LAUNCES_FOLDER}',
    dag=dag
)

# 2- get pictures

def download_image(url, dst):

    img_data = requests.get(url).content

    with open(dst, 'wb') as handler:
        handler.write(img_data)

def get_pictures():

    content = None
    
    with open(LAUNCES_FOLDER, "r") as f:
        content = json.load(f)

    results = content["results"]

    for row in results:

        url = row.get('image')

        if url is not None:
            img_path = os.path.join(IMAGES_FOLDER, os.path.basename(url))
            download_image(url, img_path)

download_images = PythonOperator(
    task_id="download_images",
    python_callable=get_pictures,
    dag=dag
)

# 3- Notify

notify = BashOperator(
    task_id="notify",
    bash_command=f"echo There are $(ls -l TP-8/tmp/images | egrep -c '^-') images",
    dag=dag
)

### Tasks Ordering

download_launches >> download_images >> notify