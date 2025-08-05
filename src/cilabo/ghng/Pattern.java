package cilabo.ghng;

import java.util.Arrays;
import java.util.Objects;

public class Pattern {
    public double[] features;
    public int label;

    public Pattern(double[] features, int label) {
        this.features = features;
        this.label = label;
    }

    public int getDimension() {
        return features.length;
    }

    // ディープコピーメソッド
    public Pattern copy() {
        return new Pattern(Arrays.copyOf(this.features, this.features.length), this.label);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Pattern sample = (Pattern) o;
        return label == sample.label && Arrays.equals(features, sample.features);
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
               '}';
    }
}