/**
 * A simple interface that defines all classifier that this application uses
 *
 * @author ndesai
 * @version 27th May 2019
 */
package com.example.visionize;

import android.graphics.Bitmap;

import java.util.List;

public interface Classifier {

    /**
     * Method to recognize objects given a particular image
     * @param bitmap
     * @return
     */
    List<DetectedObject> recognizeImage(Bitmap bitmap);

    /**
     * A method to close the Classifier model
     */
    void close();
}
