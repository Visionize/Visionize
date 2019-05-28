/**
 * A TensorFlow Image Classifier (A Simple MobileNet v1.4 CNN classifier trained on the ImageNet
 * dataset)
 *
 * @author ndesai
 * @version 27th May 2019
 */
package com.example.visionize;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

public class ImageClassifier implements Classifier {
    private static final int MAX_RESULTS = 3;
    private static final int BATCH_SIZE = 1;
    private static final int PIXEL_SIZE = 3;
    private static final float THRESHOLD = 0.1f;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 120.0f;

    private Interpreter interpreter;
    private int inputSize;
    private List<String> labelList;
    private boolean quant; // Does the model used quantized variables?

    public ImageClassifier() {

    }

    /**
     * Creates the TensorFlow Classifier given the model's path and all the other properties of model.
     * This cannot be turned into a constructor as it is a static method
     * @param assetManager
     * @param modelPath
     * @param labelPath
     * @param inputSize
     * @param quant
     * @return
     * @throws IOException
     */
    static Classifier create(AssetManager assetManager,
                             String modelPath,
                             String labelPath,
                             int inputSize,
                             boolean quant) throws IOException {
        ImageClassifier classifier = new ImageClassifier();
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath), new Interpreter.Options());
        classifier.labelList = classifier.loadLabelList(assetManager, labelPath);
        classifier.inputSize = inputSize;
        classifier.quant = quant;

        return classifier;
    }

    /**
     * Classifies the Image taken into categories from the ImageNet dataset.
     * The image is passed in as a bitmap. Returns a list of DetectedObject objects
     * @param bitmap
     * @return
     */
    @Override
    public List<DetectedObject> recognizeImage(Bitmap bitmap) {
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        if(quant){
            byte[][] result = new byte[1][labelList.size()];
            interpreter.run(byteBuffer, result);
            return getSortedResultByte(result);
        } else {
            float [][] result = new float[1][labelList.size()];
            interpreter.run(byteBuffer, result);
            return getSortedResultFloat(result);
        }
    }

    /**
     * Loads the model file from the assets directory (the tflite file). Returns a MapByteBuffer
     * of the tflite file.
     * @param assetManager
     * @param modelPath
     * @return
     * @throws IOException
     */
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /**
     * Loads the labels from the labels file in the assets directory. Returns a list of Strings
     * of the labels from the ImageNet dataset
     * @param assetManager
     * @param labelPath
     * @return
     * @throws IOException
     */
    private List<String> loadLabelList(AssetManager assetManager, String labelPath) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader = new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    /**
     * A method to convert a Bitmap to a ByteBuffer. This is necessary because the Android Camera API
     * returns a Bitmap for each image taken by the camera. But the TensorFlow Lite model requires a
     * ByteBuffer as the input format for the image.
     * @param bitmap
     * @return
     */
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;

        if(quant) {
            byteBuffer = ByteBuffer.allocateDirect(BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        } else {
            byteBuffer = ByteBuffer.allocateDirect(4 * BATCH_SIZE * inputSize * inputSize * PIXEL_SIZE);
        }

        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                final int val = intValues[pixel++];
                if(quant){
                    byteBuffer.put((byte) ((val >> 16) & 0xFF));
                    byteBuffer.put((byte) ((val >> 8) & 0xFF));
                    byteBuffer.put((byte) (val & 0xFF));
                } else {
                    byteBuffer.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    byteBuffer.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                    byteBuffer.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
                }

            }
        }
        return byteBuffer;
    }

    /**
     * Sorts the label byte probability array. This makes for relevant predictions to be displayed
     * Returns a list of DetectedObject objects
     * @param labelProbArray
     * @return
     */
    private List<DetectedObject> getSortedResultByte(byte[][] labelProbArray) {

        PriorityQueue<DetectedObject> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<DetectedObject>() {
                            @Override
                            public int compare(DetectedObject lhs, DetectedObject rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = (labelProbArray[0][i] & 0xff) / 255.0f;
            if (confidence > THRESHOLD) {
                pq.add(new DetectedObject("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        quant, confidence));
            }
        }

        final ArrayList<DetectedObject> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

    /**
     * Sorts the label probability array (this method is used for float based models).
     * Returns a list of DetectedObject objects
     * @param labelProbArray
     * @return
     */
    private List<DetectedObject> getSortedResultFloat(float[][] labelProbArray) {

        PriorityQueue<DetectedObject> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<DetectedObject>() {
                            @Override
                            public int compare(DetectedObject first, DetectedObject second) {
                                return Float.compare(first.getConfidence(), second.getConfidence());
                            }
                        });

        for (int i = 0; i < labelList.size(); ++i) {
            float confidence = labelProbArray[0][i];
            if (confidence > THRESHOLD) {
                pq.add(new DetectedObject("" + i,
                        labelList.size() > i ? labelList.get(i) : "unknown",
                        quant, confidence));
            }
        }

        final ArrayList<DetectedObject> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }

    /**
     * Safely shuts down the TensorFlow classifier model
     */
    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }
}
