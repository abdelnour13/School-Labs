package org.example;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import java.util.StringTokenizer;
import org.apache.hadoop.mapreduce.Mapper;
import java.io.IOException;

public class WordsCountMap extends Mapper<Object, Text, Text, IntWritable> {

    private static final IntWritable ONE = new IntWritable(1);

    @Override
    protected void map(Object key, Text value, Mapper<Object, Text, Text, IntWritable>.Context context) throws IOException, InterruptedException {

        // create a word tokenizer
        StringTokenizer tokenizer = new StringTokenizer(value.toString());

        while (tokenizer.hasMoreTokens()) {
            Text word = new Text(tokenizer.nextToken());
            context.write(word, ONE);
        }

    }
}
