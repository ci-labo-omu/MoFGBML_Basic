package cilabo.ghng;

import java.io.Serializable;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class GHNGModel implements Serializable {
    private static final long serialVersionUID = 1L; // シリアライズID

    // GNGおよびGHNGのパラメータ
    public int maxUnits;
    public int lambda;
    public double epsilonB;
    public double epsilonN;
    public double alpha;
    public int aMax;
    public double D;
    public int numSteps;

    // GNGの訓練結果
    public double[][] means; // ニューロンの座標 (Dimension x MaxUnits)
    public double[] errors; // 各ニューロンの誤差カウンタ
    public int[][] connections; // ニューロン間の接続情報 (age)
    public int[] winners; // 各訓練サンプルが勝者としたニューロンのインデックス

    // 訓練中に記録されるMQE
    public double[] mqe;

    // GHNGの階層構造
    public Map<Integer, GHNGModel> children; // 親ニューロンのインデックス -> 子モデル

    // コンストラクタ
    public GHNGModel(int dimension, int maxUnits, int numSamples) {
        this.maxUnits = maxUnits;
        this.means = new double[dimension][maxUnits];
        // MATLABのnan*onesのようにDouble.NaNで初期化
        for (int i = 0; i < dimension; i++) {
            for (int j = 0; j < maxUnits; j++) {
                this.means[i][j] = Double.NaN;
            }
        }
        this.errors = new double[maxUnits]; // Javaでは0で初期化される
        this.connections = new int[maxUnits][maxUnits]; // Javaでは0で初期化される
        this.winners = new int[numSamples]; // Javaでは0で初期化される
        this.children = new HashMap<>();
        this.mqe = new double[maxUnits + 1]; // 1-based index for neuron count in MQE
    }

    // ゲッターメソッド（必要に応じて追加）
    public int getDimension() {
        return means.length;
    }

    // ディープコピーメソッド (TrainGNGのOldModel用)
    public GHNGModel deepCopy() {
        GHNGModel copy = new GHNGModel(this.means.length, this.maxUnits, this.winners.length);

        copy.maxUnits = this.maxUnits;
        copy.lambda = this.lambda;
        copy.epsilonB = this.epsilonB;
        copy.epsilonN = this.epsilonN;
        copy.alpha = this.alpha;
        copy.aMax = this.aMax;
        copy.D = this.D;
        copy.numSteps = this.numSteps;

        copy.winners = Arrays.copyOf(this.winners, this.winners.length);
        for (int i = 0; i < this.means.length; i++) {
            copy.means[i] = Arrays.copyOf(this.means[i], this.means[i].length);
        }
        copy.errors = Arrays.copyOf(this.errors, this.errors.length);
        for (int i = 0; i < this.connections.length; i++) {
            copy.connections[i] = Arrays.copyOf(this.connections[i], this.connections[i].length);
        }
        copy.mqe = Arrays.copyOf(this.mqe, this.mqe.length);

        // 子供のモデルはここではコピーしない (GNGスナップショットの目的のため)
        return copy;
    }
}