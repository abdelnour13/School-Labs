docker network rm hadoop
docker network create hadoop

docker run -itd --net=hadoop -p 8080:8080 --expose 22 \
    --name hadoop-master --hostname hadoop-master \
    liliasfaxi/spark-hadoop:hv-2.7.2

docker run -itd --net=hadoop --expose 22 \
    --name hadoop-slave1 --hostname hadoop-slave1 \
    liliasfaxi/spark-hadoop:hv-2.7.2

docker run -itd --net=hadoop --expose 22 \
    --name hadoop-slave2 --hostname hadoop-slave2 \
    liliasfaxi/spark-hadoop:hv-2.7.2
