package cilabo.art;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class ARTNetModel implements Serializable {
    private static final long serialVersionUID = 1L;

    public int numNodes;
    public List<double[]> weights; // クラスターの重心 (weight)
    public List<Integer> countNodes; // 各ノードに属するサンプルの数 (密度情報)
    public List<Double> adaptiveSigs; // 各ノードのシグマ値
    public List<Integer> labelClusters; // 各クラスターのラベル

    public final int lambda;
    public final double minCIM;

    public ARTNetModel(int lambda, double minCIM) {
        this.numNodes = 0;
        this.weights = new ArrayList<>();
        this.countNodes = new ArrayList<>();
        this.adaptiveSigs = new ArrayList<>();
        this.labelClusters = new ArrayList<>();
        this.lambda = lambda;
        this.minCIM = minCIM;
    }
    //ARTNerModelのweightsの内容を表示するメソッド
    public void printWeights() {
		System.out.println("Weights (Cluster Centers):");
		for (int i = 0; i < weights.size(); i++) {
			System.out.println("Cluster " + i + ": " + java.util.Arrays.toString(weights.get(i)));
		}
	}
}