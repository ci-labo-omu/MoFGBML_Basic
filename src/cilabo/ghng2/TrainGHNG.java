package cilabo.ghng2;

import java.util.*;
import java.util.stream.Collectors;
/*
public class TrainGHNG {

    private static final int MAX_LEVELS = 4;

    public static GHNGModel trainGHNG(double[][] samples, int epochs, int maxNeurons, double tau, int lambda,
                                       double epsilonB, double epsilonN, double alpha, int aMax, double d, int level) {

        GHNGModel model = new GHNGModel();

        int dimension = samples.length;
        int numSamples = samples[0].length;

        if (((numSamples < (dimension + 1)) && (level > 1)) || (level > MAX_LEVELS)) {
            return null;
        }

        System.out.println("\nLEVEL=" + level);

        // Growing process
        int numSteps = epochs * numSamples;
        GNGModel gngModel = TrainGNG.trainGNG(samples, maxNeurons, lambda, epsilonB, epsilonN,
                                              alpha, aMax, d, numSteps, tau);

        model.gngModel = gngModel;

        // Expansion process
        int[] winners = gngModel.getWinners();
        List<Integer> ndxNeurons = gngModel.getActiveNeuronIndices();

        System.out.println("Final Graph Neurons: " + ndxNeurons.size());

        // Prune the graphs with only 2 neurons
        if (ndxNeurons.size() == 2) {
            return null;
        }

        model.child = new HashMap<>();

        for (int neuronIndex : ndxNeurons) {
            List<double[]> childSampleList = new ArrayList<>();
            for (int i = 0; i < winners.length; i++) {
                if (winners[i] == neuronIndex) {
                    double[] sample = new double[dimension];
                    for (int j = 0; j < dimension; j++) {
                        sample[j] = samples[j][i];
                    }
                    childSampleList.add(sample);
                }
            }

            if (!childSampleList.isEmpty()) {
                double[][] childSamples = new double[dimension][childSampleList.size()];
                for (int i = 0; i < childSampleList.size(); i++) {
                    for (int j = 0; j < dimension; j++) {
                        childSamples[j][i] = childSampleList.get(i)[j];
                    }
                }
                GHNGModel childModel = trainGHNG(childSamples, epochs, maxNeurons, tau, lambda,
                                                 epsilonB, epsilonN, alpha, aMax, d, level + 1);
                model.child.put(neuronIndex, childModel);
            }
        }

        return model;
    }
}
*/