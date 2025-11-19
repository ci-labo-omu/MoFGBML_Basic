package cilabo.art;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * HCA+ネットワークの状態と階層構造を保持するモデルクラスです。
 * MATLABのHCAplusNet構造体を忠実に移植します。
 */
public class HCAplusNet implements Serializable {
    private static final long serialVersionUID = 1L;

    // --- CA+ 基礎属性 (TrainCAplus_Classification.m の Model に対応) ---
    public int numNodes;                    // the number of nodes
    public List<double[]> weights;          // node position (weight, MATLABではMeansに相当)
    public List<Integer> countNodes;         // winner counter for each node (CountNode)
    public List<Double> adaptiveSigs;       // kernel bandwidth for CIM in each node (adaptiveSig)
    public List<Integer> labelClusters;      // Cluster label for connected nodes (LabelCluster - 代表クラス)

    public double V_thres;                  // similarlity thresholds (V_thres_)
    public List<Integer> activeNodeIdx;      // indexes of active nodes (activeNodeIdx)
    public int numSample;                   // number of samples (numSample)
    public boolean flag_set_lambda;         // a flag for setting lambda
    public int numActiveNode;               // number of active nodes
    public double div_lambda;               // lambda determined by diversity
    public double[] sigma;                  // an estimated sigma for CIM
    public List<List<Double>> divMat;       // a matrix for diversity via determinants
    
    // --- HCA+ 階層構造属性 (TrainHCAplus_Classification.m に追加) ---
    public final double minCIM;             
    public final int lambda;                
    public final int maxLevel;              // StopLevel
    public int level;                       // Current Level
    
    /**
     * 💡 修正点: Child属性。このフィールドが不足していました。
     * MATLABの Model.Child{NeuronIndex} に対応し、子ネットワークを格納します。
     * ノードインデックスに対応するため、nullを含むリストとして定義します。
     */
    public List<HCAplusNet> children;       
    
    public List<int[]> countLabelMatrix;    // CountLabel (ノード数 x 最大ラベル数)
    public List<Integer> winners;           // Winners (各サンプルの勝者ノードインデックス)
    
    

    public HCAplusNet(int lambda, double minCIM, int maxLevel) {
        this.numNodes = 0;
        this.weights = new ArrayList<>();
        this.countNodes = new ArrayList<>();
        this.adaptiveSigs = new ArrayList<>();
        this.labelClusters = new ArrayList<>();
        
        this.V_thres = minCIM; // 暫定初期値
        this.activeNodeIdx = new ArrayList<>();
        this.numSample = 0;
        this.flag_set_lambda = false;
        this.numActiveNode = Integer.MAX_VALUE; 
        this.div_lambda = Integer.MAX_VALUE;   
        this.sigma = null;
        this.divMat = new ArrayList<>();
        
        this.lambda = lambda;
        this.minCIM = minCIM;
        this.maxLevel = maxLevel;
        this.level = 1;
        
        // 💡 修正点: childrenの初期化
        this.children = new ArrayList<>(); 
        
        this.countLabelMatrix = new ArrayList<>();
        this.winners = new ArrayList<>();
    }
}