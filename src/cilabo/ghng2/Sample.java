package cilabo.ghng2;

import java.util.Arrays;
import java.util.Objects;

/**
 * Represents one sample in the dataset.
 * Includes feature values (continuous) and a class label (discrete).
 */
public class Sample {
    private final double[] features;  // Attribute vector
    private final int label;          // Class label (int)

    // Optional: Add an ID for traceability
    private final int id;            // Optional sample ID (e.g. for debugging or tracking)
    
    public Sample(double[] features, int label) {
        this(features, label, -1); // default ID: -1
    }

    public Sample(double[] features, int label, int id) {
        this.features = Arrays.copyOf(features, features.length);
        this.label = label;
        this.id = id;
    }

    public int getDimension() {
        return features.length;
    }

    public double[] getFeatures() {
        return Arrays.copyOf(features, features.length);
    }

    public double getFeature(int index) {
        return features[index];
    }

    public int getLabel() {
        return label;
    }

    public int getId() {
        return id;
    }

    public Sample copy() {
        return new Sample(this.features, this.label, this.id);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Sample)) return false;
        Sample sample = (Sample) o;
        return label == sample.label &&
               Arrays.equals(features, sample.features);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(label);
        result = 31 * result + Arrays.hashCode(features);
        return result;
    }

    @Override
    public String toString() {
        return "Sample{" +
               "features=" + Arrays.toString(features) +
               ", label=" + label +
               ", id=" + id +
               '}';
    }
}
