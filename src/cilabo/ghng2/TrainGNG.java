package cilabo.ghng2;

import java.util.*;
import java.util.stream.IntStream;
/*
public class TrainGNG {

    public static GHNGModel train(List<Sample> samples,
                                   int maxUnits,
                                   int lambda,
                                   double epsilonB,
                                   double epsilonN,
                                   double alpha,
                                   int aMax,
                                   double d,
                                   int numSteps,
                                   double tau) {

        int dimension = samples.get(0).getDimension();
        int numSamples = samples.size();

        GHNGModel model = new GHNGModel(dimension, maxUnits, numSamples);
        model.maxUnits = maxUnits;
        model.lambda = lambda;
        model.epsilonB = epsilonB;
        model.epsilonN = epsilonN;
        model.alpha = alpha;
        model.aMax = aMax;
        model.D = d;
        model.numSteps = numSteps;

        Random rand = new Random();
        int[] initIndices = {rand.nextInt(numSamples), rand.nextInt(numSamples)};
        for (int i = 0; i < 2; i++) {
            double[] feat = samples.get(initIndices[i]).features;
            for (int j = 0; j < dimension; j++) {
                model.means[j][i] = feat[j];
            }
        }
        model.connections[0][1] = aMax;
        model.connections[1][0] = aMax;

        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < numSamples; i++) indices.add(i);
        Collections.shuffle(indices);

        boolean growing = true;
        GHNGModel oldModel = null;
        int oldNumNeurons = 2;

        for (int t = 0; t < numSteps; t++) {
            int sampleIndex = indices.get(t % numSamples);
            Sample sample = samples.get(sampleIndex);

            double[] distances = new double[maxUnits];
            Arrays.fill(distances, Double.POSITIVE_INFINITY);

            for (int i = 0; i < maxUnits; i++) {
                if (!Double.isNaN(model.means[0][i])) {
                    distances[i] = squaredEuclidean(sample.features, getNeuron(model.means, i));
                }
            }

            int[] sortedIdx = IntStream.range(0, maxUnits)
                    .boxed()
                    .sorted(Comparator.comparingDouble(i -> distances[i]))
                    .mapToInt(i -> i)
                    .toArray();

            int s1 = sortedIdx[0];
            int s2 = sortedIdx[1];
            model.winners[sampleIndex] = s1;

            for (int i = 0; i < maxUnits; i++) {
                model.connections[s1][i] = Math.max(0, model.connections[s1][i] - 1);
                model.connections[i][s1] = Math.max(0, model.connections[i][s1] - 1);
            }

            model.errors[s1] += distances[s1];
            updateNeuron(model.means, s1, sample.features, epsilonB);

            for (int i = 0; i < maxUnits; i++) {
                if (model.connections[s1][i] > 0) {
                    updateNeuron(model.means, i, sample.features, epsilonN);
                }
            }

            model.connections[s1][s2] = aMax;
            model.connections[s2][s1] = aMax;

            pruneIsolated(model);

            if (t % lambda == 0 && growing) {
                oldNumNeurons = countNeurons(model);
                model.mqe[oldNumNeurons] = Arrays.stream(model.errors).sum() / oldNumNeurons;
                oldModel = model.deepCopy();

                int maxErrIdx = maxIndex(model.errors);
                int neighborIdx = findMaxErrorNeighbor(model, maxErrIdx);

                int newIdx = findFreeNeuron(model);
                if (newIdx != -1) {
                    for (int j = 0; j < dimension; j++) {
                        model.means[j][newIdx] = 0.5 * (model.means[j][maxErrIdx] + model.means[j][neighborIdx]);
                    }

                    model.connections[maxErrIdx][neighborIdx] = 0;
                    model.connections[neighborIdx][maxErrIdx] = 0;

                    model.connections[newIdx][maxErrIdx] = aMax;
                    model.connections[newIdx][neighborIdx] = aMax;
                    model.connections[maxErrIdx][newIdx] = aMax;
                    model.connections[neighborIdx][newIdx] = aMax;

                    model.errors[maxErrIdx] *= alpha;
                    model.errors[neighborIdx] *= alpha;
                    model.errors[newIdx] = model.errors[maxErrIdx];
                }
            }

            if (t % (2 * lambda) == (3 * lambda) / 2) {
                int currentNeurons = countNeurons(model);
                model.mqe[currentNeurons] = Arrays.stream(model.errors).sum() / currentNeurons;
                if (currentNeurons > oldNumNeurons) {
                    double improvement = (model.mqe[oldNumNeurons] - model.mqe[currentNeurons]) /
                            Math.abs(model.mqe[oldNumNeurons]);
                    if (improvement < tau) {
                        model = oldModel;
                        growing = false;
                    }
                }
            }

            for (int i = 0; i < maxUnits; i++) model.errors[i] *= d;
        }

        return model;
    }

    private static double[] getNeuron(double[][] means, int index) {
        return Arrays.stream(means).mapToDouble(dim -> dim[index]).toArray();
    }

    private static void updateNeuron(double[][] means, int idx, double[] input, double rate) {
        for (int d = 0; d < input.length; d++) {
            means[d][idx] = (1 - rate) * means[d][idx] + rate * input[d];
        }
    }

    private static double squaredEuclidean(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sum;
    }

    private static int countNeurons(GHNGModel model) {
        return (int) Arrays.stream(model.means[0]).filter(x -> !Double.isNaN(x)).count();
    }

    private static void pruneIsolated(GHNGModel model) {
        for (int i = 0; i < model.maxUnits; i++) {
            int sum = Arrays.stream(model.connections[i]).sum();
            if (sum == 0) {
                for (int d = 0; d < model.getDimension(); d++) model.means[d][i] = Double.NaN;
                model.errors[i] = 0;
            }
        }
    }

    private static int maxIndex(double[] array) {
        return IntStream.range(0, array.length)
                .reduce((i, j) -> array[i] > array[j] ? i : j).orElse(0);
    }

    private static int findMaxErrorNeighbor(GHNGModel model, int idx) {
        int best = -1;
        double maxErr = -1;
        for (int i = 0; i < model.maxUnits; i++) {
            if (model.connections[idx][i] > 0 && model.errors[i] > maxErr) {
                maxErr = model.errors[i];
                best = i;
            }
        }
        return best;
    }

    private static int findFreeNeuron(GHNGModel model) {
        for (int i = 0; i < model.maxUnits; i++) {
            if (Double.isNaN(model.means[0][i])) return i;
        }
        return -1;
    }
}
*/