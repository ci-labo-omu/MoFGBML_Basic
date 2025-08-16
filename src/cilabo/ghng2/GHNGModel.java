package cilabo.ghng2;

import java.io.Serializable;
import java.util.*;

/**
 * GHNGModel: Growing Hierarchical Neural Gas Model.
 * Represents one level of the hierarchy.
 */
public class GHNGModel implements Serializable {
    private static final long serialVersionUID = 1L;

    /** GNG / GHNG Parameters **/
    public int maxUnits;
    public int lambda;
    public double epsilonB;
    public double epsilonN;
    public double alpha;
    public int aMax;
    public double D;
    public int numSteps;
    public double[] errors; // Error counters for each neuron

    /** Neurons **/
    private final List<Neuron> neurons = new ArrayList<>();

    /** Connections: [i][j] = age between neuron i and j **/
    public int[][] connections;

    /** Each training sample’s winner index **/
    public int[] winners;

    /** MQE per step (optional) **/
    public double[] mqe;

    /** GHNG children: neuron index → child GHNGModel **/
    public Map<Integer, GHNGModel> children = new HashMap<>();

    /** Constructor **/
    public GHNGModel(int dimension, int maxUnits, int numSamples) {
        this.maxUnits = maxUnits;
        this.connections = new int[maxUnits][maxUnits];
        this.winners = new int[numSamples];
        this.mqe = new double[maxUnits + 1];

        // Initialize empty neurons with NaN
        for (int i = 0; i < maxUnits; i++) {
            neurons.add(new Neuron(dimension));
        }
    }

    /** Return the dimension of feature space **/
    public int getDimension() {
        return neurons.get(0).getWeights().length;
    }

    /** Get neuron by index **/
    public Neuron getNeuron(int index) {
        return neurons.get(index);
    }

    /** Get number of valid (finite) neurons **/
    public int getNumActiveNeurons() {
        int count = 0;
        for (Neuron n : neurons) {
            if (n.isActive()) count++;
        }
        return count;
    }

    /** Get all neurons **/
    public List<Neuron> getNeurons() {
        return neurons;
    }

    /** Deep copy (excluding children) **/
    public GHNGModel deepCopy() {
        GHNGModel copy = new GHNGModel(getDimension(), maxUnits, winners.length);

        copy.lambda = this.lambda;
        copy.epsilonB = this.epsilonB;
        copy.epsilonN = this.epsilonN;
        copy.alpha = this.alpha;
        copy.aMax = this.aMax;
        copy.D = this.D;
        copy.numSteps = this.numSteps;

        for (int i = 0; i < this.neurons.size(); i++) {
            copy.neurons.set(i, this.neurons.get(i).copy());
        }

        for (int i = 0; i < this.connections.length; i++) {
            copy.connections[i] = Arrays.copyOf(this.connections[i], this.connections[i].length);
        }

        copy.winners = Arrays.copyOf(this.winners, this.winners.length);
        copy.mqe = Arrays.copyOf(this.mqe, this.mqe.length);

        return copy;
    }

    /**
     * Neuron class inside GHNGModel.
     * Represents one unit in the model.
     */
    public static class Neuron implements Serializable {
        private static final long serialVersionUID = 1L;

        private double[] weights;      // Position in feature space
        private double error = 0.0;    // Error counter
        private int winCount = 0;      // Number of wins

        public Neuron(int dimension) {
            this.weights = new double[dimension];
            Arrays.fill(weights, Double.NaN); // inactive by default
        }

        public double[] getWeights() {
            return weights;
        }

        public void setWeights(double[] newWeights) {
            this.weights = Arrays.copyOf(newWeights, newWeights.length);
        }

        public double getWeight(int index) {
            return weights[index];
        }

        public void setWeight(int index, double value) {
            this.weights[index] = value;
        }

        public double getError() {
            return error;
        }

        public void setError(double error) {
            this.error = error;
        }

        public int getWinCount() {
            return winCount;
        }

        public void incrementWin() {
            this.winCount++;
        }

        public void resetWinCount() {
            this.winCount = 0;
        }

        public boolean isActive() {
            for (double w : weights) {
                if (!Double.isNaN(w)) return true;
            }
            return false;
        }

        public Neuron copy() {
            Neuron n = new Neuron(weights.length);
            n.weights = Arrays.copyOf(this.weights, this.weights.length);
            n.error = this.error;
            n.winCount = this.winCount;
            return n;
        }

        @Override
        public String toString() {
            return "Neuron{" +
                    "weights=" + Arrays.toString(weights) +
                    ", error=" + error +
                    ", winCount=" + winCount +
                    '}';
        }
    }
}
