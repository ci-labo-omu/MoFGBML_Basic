package cilabo.art;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cilabo.ghng.Sample;

public class ARTNetTrainer {
	
    public ARTNetTrainer() {}

    public void artClusteringTrain(HCAplusNet model, List<Sample> data) {
        List<Sample> originData = new ArrayList<>(data);

        int lambda = model.lambda;
        double minCIM = model.minCIM;
        model.winners.clear();
        
        for (int sampleNum = 0; sampleNum < originData.size(); sampleNum++) {
            Sample currentSample = originData.get(sampleNum);
            double[] inputData = currentSample.features;
            int winnerIndex = -1;
            double estSigCA;
            if (model.weights.isEmpty() || sampleNum % lambda == 0) {
                estSigCA = sigmaEstimation(originData, sampleNum, lambda);
            } else {
                estSigCA = mean(model.adaptiveSigs);
                if (estSigCA == 0.0) estSigCA = 1.0E-6;
            }
            if (model.weights.isEmpty()) {
                model.numNodes++;
                model.weights.add(Arrays.copyOf(inputData, inputData.length));
                model.countNodes.add(1);
                model.adaptiveSigs.add(estSigCA);
                model.labelClusters.add(currentSample.label);
                winnerIndex = model.numNodes;
            } else {
                double[] globalCIM = cim(inputData, model.weights, estSigCA);

                double Lcim_s1 = Double.MAX_VALUE;
                int s1 = -1;
                for (int i = 0; i < globalCIM.length; i++) {
                    if (globalCIM[i] < Lcim_s1) {
                        Lcim_s1 = globalCIM[i];
                        s1 = i;
                    }
                }

                double Lcim_s2 = Double.MAX_VALUE;
                int s2 = -1;
                if (model.weights.size() > 1) {
                    for (int i = 0; i < globalCIM.length; i++) {
                        if (i == s1) continue;
                        if (globalCIM[i] < Lcim_s2) {
                            Lcim_s2 = globalCIM[i];
                            s2 = i;
                        }
                    }
                }
                
                if (minCIM < Lcim_s1) {
                    model.numNodes++;
                    model.weights.add(Arrays.copyOf(inputData, inputData.length));
                    model.countNodes.add(1);
                    model.adaptiveSigs.add(estSigCA);
                    model.labelClusters.add(currentSample.label);
                    winnerIndex = model.numNodes;
                } else {
                    int countS1 = model.countNodes.get(s1) + 1;
                    model.countNodes.set(s1, countS1);
                    double[] weightS1 = model.weights.get(s1);
                    for (int i = 0; i < weightS1.length; i++) {
                        weightS1[i] += (1.0 / countS1) * (inputData[i] - weightS1[i]);
                    }

                    if (minCIM >= Lcim_s2 && s2 != -1) {
                        int countS2 = model.countNodes.get(s2);
                        //model.countNodes.set(s2, countS2);
                        double[] weightS2 = model.weights.get(s2);
                        for (int i = 0; i < weightS2.length; i++) {
                            weightS2[i] += (1.0 / (100.0 * countS2)) * (inputData[i] - weightS2[i]);
                        }
                    }
                    winnerIndex = s1 + 1;
                }
            }
            if (winnerIndex != -1) {
        		model.winners.add(winnerIndex);
        	} else {
        		System.err.println("Error: Winner index was not assigned properly.");
        	}
        }
    }
    

    private double sigmaEstimation(List<Sample> data, int sampleNum, int lambda) {
        List<Sample> exNodes;
        int dataSize = data.size();
        if (dataSize < lambda) {
            exNodes = data;
        } else if (sampleNum - lambda <= 0) {
            exNodes = data.subList(0, lambda);
        } else {
            exNodes = data.subList(sampleNum + 1 - lambda, sampleNum);
        }

        if (exNodes.isEmpty()) {
            //System.out.println(">>> [DEBUG] No samples for sigma estimation. Returning default.");
            return 1.0E-6;
        }

        int d = exNodes.get(0).features.length;
        double[] qStd = new double[d];

        for (int i = 0; i < d; i++) {
            double sum = 0;
            for (Sample s : exNodes) {
                sum += s.features[i];
            }
            double mean = sum / exNodes.size();
            double sumSqDiff = 0;
            for (Sample s : exNodes) {
                sumSqDiff += Math.pow(s.features[i] - mean, 2);
            }
            qStd[i] = Math.sqrt(sumSqDiff / exNodes.size());
            if (qStd[i] == 0) {
                qStd[i] = 1.0E-6;
            }
        }

        //System.out.printf(">>> [DEBUG] qStd = %s\n", Arrays.toString(qStd));

        double medianStd = median(qStd);
        double n = exNodes.size();
        double estSig = Math.pow(4 / (2.0 + d), 1.0 / (4.0 + d)) * medianStd * Math.pow(n, -1.0 / (4.0 + d));

        //System.out.printf(">>> [DEBUG] sigma = %.6e, d = %d, n = %d, median(qStd) = %.6e\n",
        //    estSig, d, (int)n, medianStd);

        return estSig;
    }

    private double[] cim(double[] x, List<double[]> y, double sig) {
        int n = y.size();
        int att = x.length;
        double[][] gKernel = new double[n][att];

        for (int i = 0; i < n; i++) {
            double[] y_i = y.get(i);
            for (int j = 0; j < att; j++) {
                gKernel[i][j] = gaussKernel(x[j] - y_i[j], sig);
            }
        }

        double[] ret1 = new double[n];
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < att; j++) {
                sum += gKernel[i][j];
            }
            ret1[i] = sum / att;
        }

        double[] cim = new double[n];
        for (int i = 0; i < n; i++) {
            cim[i] = Math.sqrt(1.0 - ret1[i]);
        }
        return cim;
    }

    private double gaussKernel(double sub, double sig) {
        return Math.exp(-Math.pow(sub, 2) / (2 * Math.pow(sig, 2)));
    }

    private double median(double[] array) {
        Arrays.sort(array);
        int middle = array.length / 2;
        if (array.length % 2 == 1) {
            return array[middle];
        } else {
            return (array[middle - 1] + array[middle]) / 2.0;
        }
    }

    private double mean(List<Double> list) {
        if (list.isEmpty()) {
            return 0.0;
        }
        double sum = 0;
        for (double d : list) {
            sum += d;
        }
        return sum / list.size();
    }
}
