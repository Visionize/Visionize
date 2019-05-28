/**
 * Main Activity for Visionize
 *
 * Contains all of the applications primary functions and handles all user interactions
 *
 * @author ndesai
 *
 * @version 27th May 2019
 */
package com.example.visionize;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;

import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;


public class MainActivity extends AppCompatActivity {

    private static final String MODEL_PATH = "mobilenet_v2_1.4_224.tflite";
    private static final String LABEL_PATH = "labels.txt";
    private static final boolean QUANT_MODEL = false;
    private static final int INPUT_SIZE = 224;

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView predictionsTextView;
    private Button detectObjectButton;
    private ImageView resultImageView;
    private CameraView cameraView;


    /**
     * onCreate method. Called when the activity is created
     * @param savedInstanceState
     */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Button toggleCameraButton; // Restrict scope as much as possible

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Find all the widgets first
        cameraView = findViewById(R.id.cameraView);
        resultImageView = findViewById(R.id.resultImageView);
        predictionsTextView = findViewById(R.id.predictionsTextView);
        predictionsTextView.setMovementMethod(new ScrollingMovementMethod());

        toggleCameraButton = findViewById(R.id.toggleCameraButton);
        detectObjectButton = findViewById(R.id.detectObjectButton);

        // Configure the camera using the cameraKit module
        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {

            }

            @Override
            public void onError(CameraKitError cameraKitError) {

            }

            @Override
            public void onImage(CameraKitImage cameraKitImage) {

                Bitmap bitmap = cameraKitImage.getBitmap();

                bitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

                resultImageView.setImageBitmap(bitmap);

                final List<DetectedObject> results = classifier.recognizeImage(bitmap);
                String resultString = results.toString();
                resultString = resultString.substring(1, resultString.length() - 1); // Exclude the brackets from the results

                if (resultString.equals("")) {
                    resultString = "Unclassified Object";
                }

                predictionsTextView.setText(resultString);

            }

            @Override
            public void onVideo(CameraKitVideo cameraKitVideo) {

            }
        });

        toggleCameraButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.toggleFacing();
            }
        });

        detectObjectButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.captureImage();
            }
        });

        initTensorFlowAndLoadModel();
    }

    /**
     * Restart the cameraView when the activity is resumed
     */
    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    /**
     * Stop the cameraView when activity is paused
     */
    @Override
    protected void onPause() {
        cameraView.stop();
        super.onPause();
    }

    /**
     * Method that is called when the activity is closed
     * Makes sure that the TensorFlow process is safely destroyed
     */
    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }

    /**
     * Helper method that initializes TensorFlow Lite and loads the classifier model
     */
    private void initTensorFlowAndLoadModel() {
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = ImageClassifier.create(
                            getAssets(),
                            MODEL_PATH,
                            LABEL_PATH,
                            INPUT_SIZE,
                            QUANT_MODEL);
                    makeButtonVisible();
                } catch (final Exception e) {
                    throw new RuntimeException("Error initializing TensorFlow!", e);
                }
            }
        });
    }

    /**
     * Helper method that makes the "Detect Object" button visible once the classifier model is loaded
     */
    private void makeButtonVisible() {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                detectObjectButton.setVisibility(View.VISIBLE);
            }
        });
    }
}
