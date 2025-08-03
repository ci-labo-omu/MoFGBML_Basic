import java.util.*;

public class GHNG {

    // Helper class to hold the GNG/GHNG model data
    public static class GHNGModel {
        public double[][] means; // Neuron weights/prototypes
        public int[] winners; // Stores the index of the winning neuron for each sample
        public Map<Integer, GHNGModel> children; // For hierarchical levels

        // GNG-specific fields (assuming these are part of the GNG model)
        public int[][] connections; // Adjacency matrix for connections
        public double[] errors; // Error counter for each neuron
        public int[] ages; // Age of connections

        public GHNGModel(int dimension, int maxNeurons) {
            this.means = new double[dimension][maxNeurons];
            this.winners = null; // Will be populated after training
            this.children = new HashMap<>();

            // Initialize GNG-specific fields
            this.connections = new int[maxNeurons][maxNeurons];
            this.errors = new double[maxNeurons];
            this.ages = new int[maxNeurons];
        }
    }

    /**
     * Trains a Growing Hierarchical Neural Gas (GHNG) model.
     * Corresponds to the MATLAB function `TrainGHNG`.
     *
     * @param samples Input samples (each column is a sample).
     * @param epochs Number of epochs to train the data.
     * @param maxNeurons Maximum number of neurons in each graph.
     * @param tau Learning parameter, 0 < Tau < 1.
     * @param lambda Number of steps between unit creations.
     * @param epsilonB Learning rate for the best matching unit.
     * @param epsilonN Learning rate for the neighbors of the best matching unit.
     * @param alpha Factor for reducing the value of the error counter in case of unit creation.
     * @param aMax Maximum admissible age of a connection.
     * @param D Factor for decreasing the value of the error counter each step.
     * @param level Current level of the hierarchy.
     * @return The trained GHNG model.
     */
    public static GHNGModel trainGHNG(double[][] samples, int epochs, int maxNeurons,
                                      double tau, int lambda, double epsilonB, double epsilonN,
                                      double alpha, int aMax, double D, int level) {

        int dimension = samples.length;
        int numSamples = samples[0].length;
        int maxLevels = 4; // Corresponds to MaxLevels in MATLAB code

        // Pruning condition from MATLAB code
        if (((numSamples < (dimension + 1)) && (level > 1)) || (level > maxLevels)) {
            return null; // Return null if pruning condition met
        }

        System.out.printf("LEVEL=%d%n", level);

        // Growing Process
        int numSteps = epochs * numSamples;
        // Assuming TrainGNG returns a GHNGModel instance with populated means and winners
        GHNGModel model = trainGNG(samples, maxNeurons, lambda, epsilonB, epsilonN, alpha, aMax, D, numSteps, tau);

        if (model == null) {
            return null; // GNG training might return null in some edge cases or if it fails
        }

        // Expansion Process
        // NdxNeurons in MATLAB is equivalent to finding non-NaN/finite means.
        // In Java, we track active neurons by keeping a count or using a list.
        // For simplicity, let's assume `model.means` might have uninitialized (e.g., all zeros or a specific placeholder)
        // neurons that are not "active" in the GNG sense.
        // A more robust GNG implementation would likely have an `activeNeurons` list or similar.
        List<Integer> ndxNeurons = new ArrayList<>();
        for (int i = 0; i < maxNeurons; i++) {
            // Check if the neuron mean is initialized (not all zeros, for example)
            // A more robust check would involve a dedicated `isActive` flag in the neuron object.
            boolean isInitialized = false;
            for (int d = 0; d < dimension; d++) {
                if (model.means[d][i] != 0.0) { // Simple check for non-zero means
                    isInitialized = true;
                    break;
                }
            }
            if (isInitialized) {
                ndxNeurons.add(i);
            }
        }

        System.out.printf("Final Graph Neurons: %d%n", ndxNeurons.size());

        // PRUNE THE GRAPHS WITH ONLY 2 NEURONS. THIS IS TO SIMPLIFY THE HIERARCHY
        if (ndxNeurons.size() == 2) {
            return null; // Return null if pruning condition met
        }

        // Recursively train child models
        for (int ndxNeuro : ndxNeurons) {
            List<double[]> childSamplesList = new ArrayList<>();
            for (int i = 0; i < numSamples; i++) {
                if (model.winners[i] == ndxNeuro) {
                    double[] sample = new double[dimension];
                    for (int d = 0; d < dimension; d++) {
                        sample[d] = samples[d][i];
                    }
                    childSamplesList.add(sample);
                }
            }

            if (!childSamplesList.isEmpty()) {
                double[][] childSamples = new double[dimension][childSamplesList.size()];
                for (int i = 0; i < childSamplesList.size(); i++) {
                    for (int d = 0; d < dimension; d++) {
                        childSamples[d][i] = childSamplesList.get(i)[d];
                    }
                }
                model.children.put(ndxNeuro, trainGHNG(childSamples, epochs, maxNeurons,
                        tau, lambda, epsilonB, epsilonN, alpha, aMax, D, level + 1));
            }
        }

        return model;
    }

    /**
     * A placeholder for the TrainGNG function.
     * This implementation provides a basic GNG training logic based on common GNG algorithms.
     * It's a simplification and might need adjustments based on the exact
     * behavior of the original MATLAB TrainGNG.
     *
     * @param samples Input samples (each column is a sample).
     * @param maxNeurons Maximum number of neurons.
     * @param lambda Number of steps between unit creations.
     * @param epsilonB Learning rate for the best matching unit.
     * @param epsilonN Learning rate for the neighbors of the best matching unit.
     * @param alpha Factor for reducing the value of the error counter.
     * @param aMax Maximum admissible age of a connection.
     * @param D Factor for decreasing the value of the error counter each step.
     * @param numSteps Total number of training steps.
     * @param tau Not directly used in the common GNG adaptation, but included for signature compatibility.
     * @return A trained GHNGModel representing the GNG.
     */
    private static GHNGModel trainGNG(double[][] samples, int maxNeurons, int lambda,
                                      double epsilonB, double epsilonN, double alpha,
                                      int aMax, double D, int numSteps, double tau) {

        int dimension = samples.length;
        int numSamples = samples[0].length;
        GHNGModel model = new GHNGModel(dimension, maxNeurons);

        // Initialize with two random neurons
        Random rand = new Random();
        int numActiveNeurons = 0;
        List<Integer> activeNeurons = new ArrayList<>();

        // Pick two distinct random samples to initialize the first two neurons
        if (numSamples < 2) {
            // Not enough samples to initialize 2 neurons, handle this case
            return null;
        }

        int idx1 = rand.nextInt(numSamples);
        int idx2;
        do {
            idx2 = rand.nextInt(numSamples);
        } while (idx2 == idx1);

        for (int d = 0; d < dimension; d++) {
            model.means[d][0] = samples[d][idx1];
            model.means[d][1] = samples[d][idx2];
        }
        activeNeurons.add(0);
        activeNeurons.add(1);
        numActiveNeurons = 2;

        model.winners = new int[numSamples]; // To store the BMU for each sample

        // Main GNG training loop
        for (int step = 0; step < numSteps; step++) {
            int sampleIndex = rand.nextInt(numSamples);
            double[] currentSample = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                currentSample[d] = samples[d][sampleIndex];
            }

            // Find best-matching unit (BMU) and second BMU
            int bmuIdx = -1;
            int sBmuIdx = -1;
            double minDist1 = Double.MAX_VALUE;
            double minDist2 = Double.MAX_VALUE;

            for (int i = 0; i < numActiveNeurons; i++) {
                int neuronIdx = activeNeurons.get(i);
                double dist = 0.0;
                for (int d = 0; d < dimension; d++) {
                    dist += Math.pow(model.means[d][neuronIdx] - currentSample[d], 2);
                }

                if (dist < minDist1) {
                    minDist2 = minDist1;
                    sBmuIdx = bmuIdx;
                    minDist1 = dist;
                    bmuIdx = neuronIdx;
                } else if (dist < minDist2) {
                    minDist2 = dist;
                    sBmuIdx = neuronIdx;
                }
            }

            if (bmuIdx == -1) {
                continue; // Should not happen if there are active neurons
            }

            model.winners[sampleIndex] = bmuIdx; // Store the winning neuron for this sample

            // 1. Increment error counter of BMU
            model.errors[bmuIdx] += minDist1;

            // 2. Adapt BMU and its neighbors
            // Move BMU
            for (int d = 0; d < dimension; d++) {
                model.means[d][bmuIdx] += epsilonB * (currentSample[d] - model.means[d][bmuIdx]);
            }

            // Move neighbors
            for (int i = 0; i < numActiveNeurons; i++) {
                int neighborIdx = activeNeurons.get(i);
                if (model.connections[bmuIdx][neighborIdx] == 1) { // If connected
                    for (int d = 0; d < dimension; d++) {
                        model.means[d][neighborIdx] += epsilonN * (currentSample[d] - model.means[d][neighborIdx]);
                    }
                }
            }

            // 3. Increment age of all edges emanating from BMU
            for (int i = 0; i < numActiveNeurons; i++) {
                int neighborIdx = activeNeurons.get(i);
                if (model.connections[bmuIdx][neighborIdx] == 1) {
                    model.ages[bmuIdx] = model.ages[bmuIdx] + 1; // Age the connection
                    model.ages[neighborIdx] = model.ages[neighborIdx] + 1; // Age the connection
                }
            }


            // 4. Create new connection or reset age
            // If connection exists, reset age. Else, create new connection.
            if (sBmuIdx != -1) {
                model.connections[bmuIdx][sBmuIdx] = 1;
                model.connections[sBmuIdx][bmuIdx] = 1;
                model.ages[bmuIdx] = 0; // Reset age of connection
                model.ages[sBmuIdx] = 0; // Reset age of connection
            }


            // 5. Remove connections that are too old
            Iterator<Integer> it = activeNeurons.iterator();
            while (it.hasNext()) {
                int nIdx = it.next();
                List<Integer> neighborsToRemove = new ArrayList<>();
                int numConnections = 0;
                for (int i = 0; i < numActiveNeurons; i++) {
                    int neighborIdx = activeNeurons.get(i);
                    if (model.connections[nIdx][neighborIdx] == 1) {
                        numConnections++;
                        if (model.ages[nIdx] > aMax || model.ages[neighborIdx] > aMax) { // Check if age exceeds max
                            neighborsToRemove.add(neighborIdx);
                        }
                    }
                }
                for (int neighborToRemove : neighborsToRemove) {
                    model.connections[nIdx][neighborToRemove] = 0;
                    model.connections[neighborToRemove][nIdx] = 0;
                    numConnections--;
                }

                if (numConnections == 0 && nIdx != bmuIdx) { // Remove neuron if it has no connections (and is not the current BMU)
                    // In a more complete GNG, you'd mark this neuron as inactive
                    // and potentially compact the `means` array.
                    // For this basic example, we'll just remove it from activeNeurons.
                    it.remove();
                    numActiveNeurons--;
                }
            }

            // 6. Insert new neuron
            if ((step + 1) % lambda == 0 && numActiveNeurons < maxNeurons) {
                // Find neuron with largest error
                double maxError = -1.0;
                int maxErrorNeuronIdx = -1;
                for (int i = 0; i < numActiveNeurons; i++) {
                    int neuronIdx = activeNeurons.get(i);
                    if (model.errors[neuronIdx] > maxError) {
                        maxError = model.errors[neuronIdx];
                        maxErrorNeuronIdx = neuronIdx;
                    }
                }

                if (maxErrorNeuronIdx != -1) {
                    // Find neighbor with largest error
                    double maxNeighborError = -1.0;
                    int maxErrorNeighborIdx = -1;
                    for (int i = 0; i < numActiveNeurons; i++) {
                        int neighborIdx = activeNeurons.get(i);
                        if (model.connections[maxErrorNeuronIdx][neighborIdx] == 1) {
                            if (model.errors[neighborIdx] > maxNeighborError) {
                                maxNeighborError = model.errors[neighborIdx];
                                maxErrorNeighborIdx = neighborIdx;
                            }
                        }
                    }

                    if (maxErrorNeighborIdx != -1) {
                        // Create new neuron
                        int newNeuronIdx = -1;
                        // Find the first available slot for a new neuron in the means array
                        for (int i = 0; i < maxNeurons; i++) {
                            boolean isTaken = false;
                            for (int activeIdx : activeNeurons) {
                                if (activeIdx == i) {
                                    isTaken = true;
                                    break;
                                }
                            }
                            if (!isTaken) {
                                newNeuronIdx = i;
                                break;
                            }
                        }

                        if (newNeuronIdx != -1) {
                            for (int d = 0; d < dimension; d++) {
                                model.means[d][newNeuronIdx] = (model.means[d][maxErrorNeuronIdx] + model.means[d][maxErrorNeighborIdx]) / 2.0;
                            }
                            activeNeurons.add(newNeuronIdx);
                            numActiveNeurons++;

                            // Update connections
                            model.connections[maxErrorNeuronIdx][maxErrorNeighborIdx] = 0;
                            model.connections[maxErrorNeighborIdx][maxErrorNeuronIdx] = 0;

                            model.connections[maxErrorNeuronIdx][newNeuronIdx] = 1;
                            model.connections[newNeuronIdx][maxErrorNeuronIdx] = 1;
                            model.ages[maxErrorNeuronIdx] = 0;
                            model.ages[newNeuronIdx] = 0;

                            model.connections[maxErrorNeighborIdx][newNeuronIdx] = 1;
                            model.connections[newNeuronIdx][maxErrorNeighborIdx] = 1;
                            model.ages[maxErrorNeighborIdx] = 0;
                            model.ages[newNeuronIdx] = 0;

                            // Decrease error counters
                            model.errors[maxErrorNeuronIdx] *= alpha;
                            model.errors[maxErrorNeighborIdx] *= alpha;
                            model.errors[newNeuronIdx] = model.errors[maxErrorNeuronIdx];
                        }
                    }
                }
            }

            // 7. Decrease all error counters
            for (int i = 0; i < numActiveNeurons; i++) {
                int neuronIdx = activeNeurons.get(i);
                model.errors[neuronIdx] *= D;
            }
        }
        return model;
    }

    // Example Usage (main method for testing)
    public static void main(String[] args) {
        // Create some dummy data (e.g., 2D points)
        double[][] samples = {
            {1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2}, // Dimension 1
            {1.0, 1.1, 1.2, 5.0, 5.1, 5.2, 10.0, 10.1, 10.2}  // Dimension 2
        };

        int epochs = 20;
        int maxNeurons = 20;
        double tau = 0.01;
        int lambda = 100;
        double epsilonB = 0.05;
        double epsilonN = 0.001;
        double alpha = 0.5;
        int aMax = 50;
        double D = 0.995;
        int level = 1;

        GHNGModel trainedModel = trainGHNG(samples, epochs, maxNeurons,
                                         tau, lambda, epsilonB, epsilonN,
                                         alpha, aMax, D, level);

        if (trainedModel != null) {
            System.out.println("\nGHNG Training Complete!");
            // You can add logic here to inspect the trained model
            // e.g., print neuron positions at the top level
            // and recursively print child models if they exist.
        } else {
            System.out.println("\nGHNG Training did not produce a model (pruned or error).");
        }
    }
}