from kafka import KafkaProducer, KafkaConsumer

producer = KafkaProducer(bootstrap_servers=['localhost:29092'], api_version=(2, 0, 2))
producer.send('topic-example', bytes("hello kafka !", 'UTF-8'))

consumer = KafkaConsumer('topic-example', bootstrap_servers=['localhost:29092'], auto_offset_reset='earliest')

for message in consumer:
    print(message.value)
