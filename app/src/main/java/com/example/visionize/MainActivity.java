package com.example.visionize;

import android.media.ImageReader;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;

import com.example.imageclassifier.ImageClassifier;

public abstract class MainActivity extends AppCompatActivity
        implements ImageReader.OnImageAvailableListener, View.OnClickListener, AdapterView.OnItemSelectedListener {

    private static final String MODEL_PATH = "mobilenet_v2_1.4_224.tflite";
    private static final String LABEL_PATH = "labels.txt";
    private static final boolean QUANT_MODEL = true;
    private static final int INPUT_SIZE = 224;

    ImageClassifier classifier;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
}
