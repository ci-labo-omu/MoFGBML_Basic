package cilabo.hca;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

public class HCAplusNet implements Serializable {
    private static final long serialVersionUID = 1L;

    // TrainCAplus_Classification.m の Model 構造体に対応するフィールド
    public int Level;
    public double[][] Samples;
    public int[] SampleLabels;
    public int NumSteps;
    public int[] Winners;

    public int numNodes;
    public double[][] weight;
    public int[] CountNode;
    public double[] adaptiveSig;
    public double V_thres_;
    public List<Integer> activeNodeIdx;
    public int[][] CountLabel;
    public int numSample;
    public boolean flag_set_lambda;
    public int numActiveNode;
    public double[][] divMat;
    public double[] sigma;
    public double[][] CL;
    public double div_lambda; // λ (lambda) の値
    
    // HCA+ の階層構造のためのフィールド
    public int MaxLevel;
    public int Epochs;
    public Map<Integer, HCAplusNet> Child;
    public double[][] Means;
    
    // コンストラクタ
    public HCAplusNet(int dimension, int numSamples, int maxLabel, int maxLevel, int epochs) {
        this.Level = 0;
        this.Samples = new double[dimension][numSamples];
        this.SampleLabels = new int[numSamples];
        this.NumSteps = 0;
        this.Winners = new int[numSamples];

        this.numNodes = 0;
        this.weight = new double[0][0];
        this.CountNode = new int[0];
        this.adaptiveSig = new double[0];
        this.V_thres_ = 0.0;
        this.activeNodeIdx = new ArrayList<>();
        this.CountLabel = new int[0][maxLabel+1];
        this.numSample = 0;
        this.flag_set_lambda = false;
        this.numActiveNode = -1;
        this.divMat = new double[0][0];
        this.sigma = new double[0];
        this.CL = new double[0][0];

        this.MaxLevel = maxLevel;
        this.Epochs = epochs;
        this.Child = new HashMap<>();
        this.Means = new double[0][0];
    }
}