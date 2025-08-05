package cilabo.art;

import java.util.ArrayList;
import java.util.List;

import cilabo.ghng.Pattern;

public class MinMaxScaler {
    private double[] min;
    private double[] max;
    private int dimension;

    /**
     * Calculates the min and max for each feature from the data.
     * @param samples The list of samples.
     */
    public void fit(List<Pattern> samples) {
        if (samples == null || samples.isEmpty()) {
            return;
        }
        dimension = samples.get(0).features.length;
        min = new double[dimension];
        max = new double[dimension];
        
        // Initialize min and max with the first sample's values
        for (int i = 0; i < dimension; i++) {
            min[i] = samples.get(0).features[i];
            max[i] = samples.get(0).features[i];
        }

        // Iterate through all samples to find the actual min and max
        for (Pattern sample : samples) {
            for (int i = 0; i < dimension; i++) {
                if (sample.features[i] < min[i]) {
                    min[i] = sample.features[i];
                }
                if (sample.features[i] > max[i]) {
                    max[i] = sample.features[i];
                }
            }
        }
    }

    /**
     * Transforms the samples to a [0, 1] range based on the fitted min and max.
     * @param samples The list of samples to transform.
     * @return A new list of normalized samples.
     */
    public List<Pattern> transform(List<Pattern> samples) {
        if (samples == null || samples.isEmpty()) {
            return new ArrayList<>();
        }
        List<Pattern> normalizedSamples = new ArrayList<>();
        for (Pattern sample : samples) {
            double[] normalizedFeatures = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                double range = max[i] - min[i];
                if (range == 0) {
                    normalizedFeatures[i] = 0.0; // Avoid division by zero
                } else {
                    normalizedFeatures[i] = (sample.features[i] - min[i]) / range;
                }
            }
            normalizedSamples.add(new Pattern(normalizedFeatures, sample.label));
        }
        return normalizedSamples;
    }

    /**
     * Fits the scaler and transforms the data in one step.
     * @param samples The list of samples to fit and transform.
     * @return A new list of normalized samples.
     */
    public List<Pattern> fitTransform(List<Pattern> samples) {
        fit(samples);
        return transform(samples);
    }
}