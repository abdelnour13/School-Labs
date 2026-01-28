package org.example;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;
import java.util.Iterator;
import java.io.IOException;

public class WordsCountReduce extends Reducer<Text,IntWritable,Text,Text> {

    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Reducer<Text, IntWritable, Text, Text>.Context context) throws IOException, InterruptedException {

        Iterator<IntWritable> i = values.iterator();
        int count = 0;

        while (i.hasNext()) {
            count += i.next().get();
        }

        context.write(key, new Text(count + " occurences"));

    }
}
