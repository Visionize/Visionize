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

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public Float getConfidence() {
        return confidence;
    }

    @Override
    public String toString() {
        String resultString = "";
//        if (id != null) {
//            resultString += "[" + id + "] ";
//        }

        if (title != null) {
            resultString += title + " ";
        }

        if (confidence != null) {
            resultString += String.format(Locale.CANADA, "(%.1f%%) ", confidence * 100.0f);
        }

        return resultString.trim();
    }


}
