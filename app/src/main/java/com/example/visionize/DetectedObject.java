/**
 * A class to create objects for each prediction made for a given image.
 *
 * @author ndesai
 * @version 27th May 2019
 */
package com.example.visionize;

import java.util.Locale;

class DetectedObject {
    private final String id; // ID of object
    private final String title; // title of recognised image
    private final boolean quant; // Is the model being used a quantized one or a float one ?
    private final Float confidence; // Confidence score for predictions

    public DetectedObject(final String id, final String title, final boolean quant, final Float confidence) {
        this.id = id;
        this.title = title;
        this.quant = quant;
        this.confidence = confidence;
    }

    /**
     * Returns the ID of the prediction made (Accessor method)
     * @return
     */
    public String getId() {
        return id;
    }

    /**
     * Returns the name/title of the prediction made (Accessor method)
     * @return
     */
    public String getTitle() {
        return title;
    }

    /**
     * Returns the confidence of the prediction made (Accessor method)
     * @return
     */
    public Float getConfidence() {
        return confidence;
    }

    /**
     * Returns a string containing the information about the prediction made
     * @return
     */
    @Override
    public String toString() {
        String resultString = "";

        if (title != null) {
            resultString += title + " ";
        }

        if (confidence != null) {
            resultString += String.format(Locale.CANADA, "(%.1f%%) ", confidence * 100.0f);
        }

        return resultString.trim();
    }


}
