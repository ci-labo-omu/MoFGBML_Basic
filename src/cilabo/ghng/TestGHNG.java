package cilabo.ghng;

import java.util.Arrays;

public class TestGHNG {

    // A simple class to hold the results of the testing function
    public static class TestResult {
        public int[] winners; // Winning neurons for each input sample (0-indexed)
        public double[] errors; // Squared distance (error) for each input sample

        public TestResult(int numSamples) {
            this.winners = new int[numSamples];
            this.errors = new double[numSamples];
        }
    }

    /**
     * Tests a GHNG model by determining the closest prototype for each sample.
     * Corresponds to the MATLAB function `TestGHNG`.
     *
     * @param prototypes Planar prototypes of the GHNG model (dimension x num_neurons).
     * Each column is a prototype.
     * @param samples Matrix of input data (dimension x num_samples).
     * Each column is a sample.
     * @return A TestResult object containing the winning neurons and their squared distances (errors).
     */
    public static TestResult testGHNG(double[][] prototypes, double[][] samples) {
        int dimension = samples.length;
        int numSamples = samples[0].length;
        int numPrototypes = prototypes[0].length; // Number of neurons/prototypes

        TestResult result = new TestResult(numSamples);

        // Main loop
        for (int ndxSample = 0; ndxSample < numSamples; ndxSample++) {
            // Get the current sample
            double[] currentSample = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                currentSample[d] = samples[d][ndxSample];
            }

            // Calculate squared distances from currentSample to all prototypes
            double minimumSquaredDistance = Double.POSITIVE_INFINITY;
            int winnerIndex = -1;

            for (int ndxProto = 0; ndxProto < numPrototypes; ndxProto++) {
                // Skip prototypes that are NaN (inactive/uninitialized neurons from GNG)
                if (Double.isNaN(prototypes[0][ndxProto])) {
                    continue;
                }

                double squaredDistance = 0.0;
                for (int d = 0; d < dimension; d++) {
                    squaredDistance += Math.pow(currentSample[d] - prototypes[d][ndxProto], 2);
                }

                if (squaredDistance < minimumSquaredDistance) {
                    minimumSquaredDistance = squaredDistance;
                    winnerIndex = ndxProto;
                }
            }

            result.winners[ndxSample] = winnerIndex;
            result.errors[ndxSample] = minimumSquaredDistance;
        }

        return result;
    }

    // Example Usage
    public static void main(String[] args) {
        // Dummy Prototypes (e.g., from a trained GHNG model)
        // Dimension x NumPrototypes
        double[][] prototypes = {
            {1.0, 5.0, 10.0, Double.NaN}, // Prototype 0, 1, 2, 3 (inactive)
            {1.0, 5.0, 10.0, Double.NaN}
        };

        // Dummy Samples (test data)
        // Dimension x NumSamples
        double[][] samples = {
            {1.1, 5.2, 9.8, 1.3, 5.0, 10.1, 18},
            {1.2, 5.1, 9.9, 1.4, 4.9, 10.2, 12}
        };

        System.out.println("Testing GHNG model...");
        TestResult testResult = testGHNG(prototypes, samples);

        System.out.println("\nTest Results:");
        for (int i = 0; i < samples[0].length; i++) {
            System.out.printf("Sample %d (%.2f, %.2f): Winner = Neuron %d, Error = %.4f%n",
                              i, samples[0][i], samples[1][i], testResult.winners[i], testResult.errors[i]);
        }
    }
}