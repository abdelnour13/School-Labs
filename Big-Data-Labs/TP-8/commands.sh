airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname Admin --role Admin --email your_mail@example.org

### Excute in tow seprate windows
airflow webserver
airflow scheduler