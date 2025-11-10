package cilabo.hca;
import java.util.List;

public class DataLoadResult {
    public int numSamples;
    public int numDims;
    public int numClasses;
    public List<double[]> dataLines;

    public DataLoadResult(int numSamples, int numDims, int numClasses, List<double[]> dataLines) {
        this.numSamples = numSamples;
        this.numDims = numDims;
        this.numClasses = numClasses;
        this.dataLines = dataLines;
    }
}