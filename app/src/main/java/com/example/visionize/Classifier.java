package com.example.visionize;

import android.graphics.Bitmap;

import java.util.List;

public interface Classifier {

    List<DetectedObject> recognizeImage(Bitmap bitmap);

    void close();
}
