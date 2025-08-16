package cilabo.hca;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.Collections;
import java.util.stream.Collectors;

public class TrainCAplus {
    private final Random rand;

    public TrainCAplus(long seed) {
        this.rand = new Random(seed);
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
                    double initSig = sigmaEstimationByNode(initWeight, IntStream.range(0, initWeight.length).toArray());
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
                    adaptiveSig[numNodes - 1] = sigmaEstimationByNode(subWeight, IntStream.range(0, subWeight.length).toArray());
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
        List<Integer> newList = new ArrayList<>();
        newList.add(winnerIdx);
        for(int idx : activeNodeIdx){
            if(idx != winnerIdx){
                newList.add(idx);
            }
        }
        return newList;
    }
}