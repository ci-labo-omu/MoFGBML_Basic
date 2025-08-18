// src/com/yourpackage/HCAplusTrainer.java
package cilabo.hca;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.ArrayList;

public class HCAplusTrainer {

    private final Random rand;
	private int maxLevels;

    public HCAplusTrainer(int maxLevels, long seed) {
        this.rand = new Random(seed);
        this.maxLevels = maxLevels;
    }

    /**
     * Train an HCAplus classification model.
     * @param samples Input samples (Dimension x NumSamples)
     * @param net The HCAplusNetModel to be trained
     * @param level Current level
     * @param sampleLabels The labels of the samples
     * @param maxLabel Max label value
     * @return The trained model and a flag indicating if pruning occurred
     */
    public HCAplusNet trainHCAplus(double[][] samples, HCAplusNet net, int level, int[] sampleLabels, int maxLabel) {

        int dimension = samples.length;
        int numSamples = samples[0].length;
        
        if (level > net.MaxLevel) {
            return null; // Prune
        }

        // Growing Process
        int numSteps = net.Epochs * numSamples;
        // The original MATLAB code calls TrainCAplus_Classification recursively here.
        // We'll implement that logic here for a single level.
        
        HCAplusNet model = trainCAplus(samples, net, level, numSteps, sampleLabels, maxLabel);

        if (model == null) {
            return null;
        }
        
        // Expansion Process
        int[] winners = model.Winners;
        model.Means = transpose(model.weight);
        
        List<Integer> neuronsIndex = new ArrayList<>();
        for (int i = 0; i < model.Means[0].length; i++) {
            if (!Double.isNaN(model.Means[0][i])) {
                neuronsIndex.add(i);
            }
        }
        int numNeurons = neuronsIndex.size();
        System.out.printf("  [HCAplusTrainer] Level %d: numSamples=%d, numNeurons=%d%n", level, numSamples, numNeurons);

        // Prune small clusters
        if (numNeurons <= 2 || numSamples == numNeurons) {
            return null; // Prune
        } else {
            for (int neuronIndex : neuronsIndex) {
                
                List<double[]> childSamplesList = new ArrayList<>();
                List<Integer> childSampleLabelsList = new ArrayList<>();
                
                for (int i = 0; i < numSamples; i++) {
                    if (winners[i] == neuronIndex) {
                        childSamplesList.add(getColumn(samples, i));
                        childSampleLabelsList.add(sampleLabels[i]);
                    }
                }
                
                if (!childSamplesList.isEmpty()) {
                    double[][] childSamples = listToMatrix(childSamplesList);
                    int[] childSampleLabels = new int[childSampleLabelsList.size()];
                    for (int i = 0; i < childSampleLabels.length; i++) {
                        childSampleLabels[i] = childSampleLabelsList.get(i);
                    }
                    model.Child.put(neuronIndex, trainHCAplus(childSamples, net, level + 1, childSampleLabels, maxLabel));
                }
            }
        }
        return model;
    }
    
    
    // TrainCAplus_Classification.m のロジックを再現する public メソッド
    public HCAplusNet trainCAplus(double[][] samples, HCAplusNet net, int level, int numSteps, int[] sampleLabels, int maxLabel) {
        
        HCAplusNet model = new HCAplusNet(samples.length, samples[0].length, maxLabel, net.MaxLevel, net.Epochs);
        model.Level = level;
        model.Samples = samples;
        model.SampleLabels = sampleLabels;
        model.NumSteps = numSteps;
        
        double[][] data = transpose(samples);
        
        int numNodes = net.numNodes;
        double[][] weight = net.weight;
        int[] countNode = net.CountNode;
        double[] adaptiveSig = net.adaptiveSig;
        double V_thres_ = net.V_thres_;
        List<Integer> activeNodeIdx = new ArrayList<>(net.activeNodeIdx);
        int[][] countLabel = net.CountLabel;
        int numSample = net.numSample;
        boolean flag_set_lambda = net.flag_set_lambda;
        int numActiveNode = net.numActiveNode;
        double[][] divMat = net.divMat;
        double[] sigma = net.sigma;
        
        double div_threshold = 1.0e-6;
        int n_init_data = 10;
        
        if (weight.length == 0) {
            countLabel = new int[1][maxLabel + 1];
        }
        
        if (numSample == 0) {
            double[][] initData = Arrays.copyOfRange(data, 0, Math.min(data.length, n_init_data));
            sigma = new double[]{sigmaEstimationByNode(initData, IntStream.range(0, initData.length).toArray())};
        }
        
        for (int sampleNum = 0; sampleNum < data.length; sampleNum++) {
            double[] input = data[sampleNum];
            int label = sampleLabels[sampleNum];
            numSample++;
            
            if (!flag_set_lambda || numNodes < numActiveNode) {
                numNodes++;
                weight = resizeMatrix(weight, numNodes, input.length);
                weight[numNodes - 1] = input;
                
                countNode = resizeArray(countNode, numNodes);
                countNode[numNodes - 1] = 1;
                
                if (adaptiveSig.length != 0) {
                    adaptiveSig = resizeArray(adaptiveSig, numNodes);
                    adaptiveSig[numNodes - 1] = adaptiveSig[0];
                }
                
                countLabel = resizeMatrix(countLabel, numNodes, maxLabel + 1);
                countLabel[numNodes - 1][label] = 1;
                model.Winners[sampleNum] = numNodes;
                
                if (numNodes >= n_init_data && !flag_set_lambda) {
                    double[] corr = new double[numNodes];
                    for (int i = 0; i < numNodes; i++) {
                        corr[i] = 1.0 - CIM(weight[numNodes - 1], new double[][]{weight[i]}, mean(sigma))[0];
                    }
                    divMat = resizeMatrix(divMat, numNodes, numNodes);
                    for (int i = 0; i < numNodes; i++) {
                        divMat[numNodes - 1][i] = corr[i];
                        divMat[i][numNodes - 1] = corr[i];
                    }
                    double detDiv = determinant(exp(divMat));
                    if (detDiv < div_threshold && numNodes >= n_init_data) {
                        numActiveNode = numNodes;
                        model.div_lambda = numActiveNode * 2; // div_lambdaをモデルに設定
                    }
                }
                
                if (numNodes == numActiveNode && !flag_set_lambda) {
                    flag_set_lambda = true;
                    int numAN = activeNodeIdx.size();
                    double[][] initWeight = Arrays.copyOfRange(weight, 0, Math.min(numAN, numActiveNode));
                    double initSig = sigmaEstimationByNode(weight, activeNodeIdx.stream().mapToInt(Integer::intValue).toArray());
                    adaptiveSig = new double[numNodes];
                    Arrays.fill(adaptiveSig, initSig);
                    
                    double[] tmpTh = new double[numActiveNode];
                    for (int k = 0; k < numActiveNode; k++) {
                        double[] tmpCIMs1 = CIM(weight[k], weight, mean(adaptiveSig));
                        int s1 = minIndex(tmpCIMs1);
                        tmpCIMs1[s1] = Double.POSITIVE_INFINITY;
                        tmpTh[k] = min(tmpCIMs1);
                    }
                    V_thres_ = mean(tmpTh);
                }
            } else {
                double meanAdaptiveSig = mean(adaptiveSig);
                double[] globalCIM = CIM(input, weight, meanAdaptiveSig);
                
                double vs1 = min(globalCIM);
                int s1 = minIndex(globalCIM);
                
                double[] tempCIM = Arrays.copyOf(globalCIM, globalCIM.length);
                tempCIM[s1] = Double.POSITIVE_INFINITY;
                double vs2 = min(tempCIM);
                int s2 = minIndex(tempCIM);
                
                if (V_thres_ < vs1 || numNodes < numActiveNode) {
                    numNodes++;
                    weight = resizeMatrix(weight, numNodes, input.length);
                    weight[numNodes - 1] = input;
                    activeNodeIdx = updateActiveNode(activeNodeIdx, numNodes);
                    countNode = resizeArray(countNode, numNodes);
                    countNode[numNodes - 1] = 1;
                    
                    countLabel = resizeMatrix(countLabel, numNodes, maxLabel + 1);
                    countLabel[numNodes - 1][label] = 1;
                    model.Winners[sampleNum] = numNodes;
                    
                    int numAN = activeNodeIdx.size();
                    double[][] subWeight = new double[numAN][weight[0].length];
                    for(int i = 0; i < numAN; i++){
                        subWeight[i] = weight[activeNodeIdx.get(i)-1];
                    }
                    adaptiveSig = resizeArray(adaptiveSig, numNodes);
                    adaptiveSig[numNodes - 1] = sigmaEstimationByNode(weight, activeNodeIdx.stream().mapToInt(Integer::intValue).toArray());
                } else {
                    countNode[s1] = countNode[s1] + 1;
                    for (int i = 0; i < input.length; i++) {
                        weight[s1][i] += (1.0 / countNode[s1]) * (input[i] - weight[s1][i]);
                    }
                    activeNodeIdx = updateActiveNode(activeNodeIdx, s1 + 1);
                    model.Winners[sampleNum] = s1 + 1;
                    countLabel[s1][label] = countLabel[s1][label] + 1;

                    if (V_thres_ >= vs2) {
                        for(int i = 0; i < input.length; i++){
                            weight[s2][i] += (1.0/(100*countNode[s2]))*(input[i]-weight[s2][i]);
                        }
                    }
                }
            }
        }
        
        if (adaptiveSig.length == 0) {
            numActiveNode = numNodes;
            model.div_lambda = numActiveNode * 2; // div_lambdaをモデルに設定
            int numAN = activeNodeIdx.size();
            double[][] subWeight = new double[numAN][weight[0].length];
            for(int i = 0; i < numAN; i++){
                subWeight[i] = weight[activeNodeIdx.get(i)-1];
            }
            double initSig = sigmaEstimationByNode(subWeight, IntStream.range(0, subWeight.length).toArray());
            adaptiveSig = new double[numNodes];
            Arrays.fill(adaptiveSig, initSig);
            
            double[] tmpTh = new double[numActiveNode];
            for (int k = 0; k < numActiveNode; k++) {
                double[] tmpCIMs1 = CIM(weight[k], weight, mean(adaptiveSig));
                int s1 = minIndex(tmpCIMs1);
                tmpCIMs1[s1] = Double.POSITIVE_INFINITY;
                tmpTh[k] = min(tmpCIMs1);
            }
            V_thres_ = mean(tmpTh);
        }
        
        model.numNodes = numNodes;
        model.weight = weight;
        model.CountNode = countNode;
        model.adaptiveSig = adaptiveSig;
        model.V_thres_ = V_thres_;
        model.activeNodeIdx = activeNodeIdx;
        model.CountLabel = countLabel;
        model.CL = transpose(countLabel);
        model.numSample = numSample;
        model.flag_set_lambda = flag_set_lambda;
        model.numActiveNode = numActiveNode;
        model.divMat = divMat;
        model.sigma = sigma;
        
        return model;
    }
    
 // 既存の double[][] を転置するメソッド
    private double[][] transpose(double[][] matrix) {
        if (matrix.length == 0) return new double[0][0];
        double[][] transposed = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }
        return transposed;
    }

    // ★★★ 新規追加：int[][] を転置して double[][] を返すメソッド ★★★
    private double[][] transpose(int[][] matrix) {
        if (matrix.length == 0) return new double[0][0];
        double[][] transposed = new double[matrix[0].length][matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                transposed[j][i] = (double) matrix[i][j]; // intをdoubleにキャスト
            }
        }
        return transposed;
    }
    
    private double[][] resizeMatrix(double[][] original, int rows, int cols) {
        double[][] newMatrix = new double[rows][cols];
        for(int i=0; i < original.length && i < rows; i++){
            System.arraycopy(original[i], 0, newMatrix[i], 0, Math.min(original[i].length, cols));
        }
        return newMatrix;
    }
    
    private int[][] resizeMatrix(int[][] original, int rows, int cols) {
        int[][] newMatrix = new int[rows][cols];
        for(int i=0; i < original.length && i < rows; i++){
            System.arraycopy(original[i], 0, newMatrix[i], 0, Math.min(original[i].length, cols));
        }
        return newMatrix;
    }
    
    private int[] resizeArray(int[] original, int length) {
        return Arrays.copyOf(original, length);
    }
    
    private double[] resizeArray(double[] original, int length) {
        return Arrays.copyOf(original, length);
    }

    private double[][] exp(double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++) {
                result[i][j] = Math.exp(matrix[i][j]);
            }
        }
        return result;
    }

    private double determinant(double[][] matrix) {
        // Simple determinant for small matrices (e.g., 2x2, 3x3)
        // For larger matrices, a more robust implementation or a library is needed.
        if (matrix.length == 1 && matrix[0].length == 1) {
            return matrix[0][0];
        }
        if (matrix.length == 2 && matrix[0].length == 2) {
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
        }
        // Dummy return for larger matrices
        return 0.0;
    }

    private double[] CIM(double[] x, double[][] y, double sig) {
        int m = y.length;
        int d = x.length;
        double[][] gKernel = new double[m][d];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++) {
                gKernel[i][j] = gaussKernel(x[j] - y[i][j], sig);
            }
        }
        
        double[] ret1 = new double[m];
        for (int i = 0; i < m; i++) {
            double sum = 0;
            for (int j = 0; j < d; j++) {
                sum += gKernel[i][j];
            }
            ret1[i] = sum / d;
        }

        double[] cim = new double[m];
        for (int i = 0; i < m; i++) {
            cim[i] = Math.sqrt(1.0 - ret1[i]);
        }
        return cim;
    }
    
    private double sigmaEstimationByNode(double[][] weight, int[] activeNodeIdx) {
    	double[][] exNodes = new double[activeNodeIdx.length][weight[0].length];
        
        for (int i = 0; i < activeNodeIdx.length; i++) {
            exNodes[i] = weight[activeNodeIdx[i]];
        }
        System.out.printf(">>> [DEBUG] sigmaEstimationByNode: activeNodeIdx = %s, exNodes = %s\n", 
			Arrays.toString(activeNodeIdx), Arrays.deepToString(exNodes));
        double[] qStd = new double[exNodes[0].length];
        for (int i = 0; i < exNodes[0].length; i++) {
            double sum = 0;
            for (int j = 0; j < exNodes.length; j++) {
                sum += exNodes[j][i];
            }
            double mean = sum / exNodes.length;
            double sumSqDiff = 0;
            for (int j = 0; j < exNodes.length; j++) {
                sumSqDiff += Math.pow(exNodes[j][i] - mean, 2);
            }
            qStd[i] = Math.sqrt(sumSqDiff / exNodes.length);
        }
        
        for (int i = 0; i < qStd.length; i++) {
            if (qStd[i] == 0) {
                qStd[i] = 1.0E-6;
            }
        }
        
        double medianStd = median(qStd);
        int n = exNodes.length;
        int d = exNodes[0].length;
        
        return medianStd * Math.pow(n, -1.0/(4.0+d)) * Math.pow(4.0/(2.0+d), 1.0/(4.0+d));
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
    
    private double mean(double[] array) {
        return Arrays.stream(array).average().orElse(0.0);
    }
    
    private int minIndex(double[] array) {
        int minIndex = -1;
        double minValue = Double.POSITIVE_INFINITY;
        for (int i = 0; i < array.length; i++) {
            if (array[i] < minValue) {
                minValue = array[i];
                minIndex = i;
            }
        }
        return minIndex;
    }
    
    private double min(double[] array) {
        double minValue = Double.POSITIVE_INFINITY;
        for (double v : array) {
            if (v < minValue) {
                minValue = v;
            }
        }
        return minValue;
    }
    
    private List<Integer> updateActiveNode(List<Integer> activeNodeIdx, int winnerIdx) {
        // 1. activeNodeIdxからwinnerIdxを削除する
        //    remove()は最初の要素しか削除しないため、複数のwinnerIdxが存在する場合には不十分
        //    リストをイテレータで安全に削除するか、単純なループを使う
        
        // 以下はMATLABのロジックを忠実に再現する修正案
        List<Integer> newList = new ArrayList<>();
        newList.add(winnerIdx); // winnerIdxをリストの先頭に追加
        
        for (Integer idx : activeNodeIdx) {
            if (!idx.equals(Integer.valueOf(winnerIdx))) {
                newList.add(idx); // winnerIdxと異なる要素をリストに追加
            }
        }
        
        return newList;
    } 
    
    // Javaのヘルパーメソッド
    
    private double[] getColumn(double[][] matrix, int colIndex) {
        double[] column = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            column[i] = matrix[i][colIndex];
        }
        return column;
    }
    
    private double[][] listToMatrix(List<double[]> list) {
        if (list.isEmpty()) return new double[0][0];
        int dimension = list.get(0).length;
        double[][] matrix = new double[dimension][list.size()];
        for (int i = 0; i < list.size(); i++) {
            for (int j = 0; j < dimension; j++) {
                matrix[j][i] = list.get(i)[j];
            }
        }
        return matrix;
    }
}