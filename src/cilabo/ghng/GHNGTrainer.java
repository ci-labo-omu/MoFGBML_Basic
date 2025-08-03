// src/com/yourpackage/GHNGTrainer.java
package cilabo.ghng;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class GHNGTrainer {

    private final int maxLevels; // 最大階層レベル
    private final Random rand;

    public GHNGTrainer(int maxLevels, long seed) {
        this.maxLevels = maxLevels;
        this.rand = new Random(seed); // 乱数シードをコンストラクタで設定
    }

    /**
     * Growing Hierarchical Neural Gas (GHNG) モデルを訓練します。
     *
     * @param features      入力サンプルデータの特徴量 (次元 x サンプル数)
     * @param originalLabels 入力サンプルデータの元のクラスラベル (GHNGModelに直接は格納しないが、winnersの紐付けに必要)
     * @param epochs        エポック数
     * @param maxNeurons    各GHNGレベルでの最大ニューロン数
     * @param tau           学習パラメータ (成長停止閾値)
     * @param lambda        ユニット生成ステップ間隔
     * @param epsilonB      BMU学習率
     * @param epsilonN      近傍学習率
     * @param alpha         誤差削減率 (ユニット生成時)
     * @param aMax          接続の最大寿命
     * @param D             誤差減衰率
     * @param level         現在の階層レベル (初期呼び出しは0)
     * @return 訓練済みGHNGModel
     */
    public GHNGModel trainGHNG(double[][] features, int[] originalLabels, int epochs, int maxNeurons,
                               double tau, int lambda, double epsilonB, double epsilonN,
                               double alpha, int aMax, double D, int level) {

        int dimension = features.length;
        int numSamples = features[0].length;

        if (((numSamples < (dimension + 1)) && (level > 0)) || (level >= maxLevels)) { // level > 0 for prune condition
            System.out.printf("  [LEVEL %d] Pruned: Not enough samples (%d) or max level reached (%d).%n", level, numSamples, maxLevels);
            return null;
        }

        System.out.printf("---%nLEVEL %d: Starting GNG training for %d samples (Dimension %d) ---%n", level, numSamples, dimension);

        int numSteps = epochs * numSamples;
        GHNGModel model = trainGNG(features, maxNeurons, lambda, epsilonB, epsilonN, alpha, aMax, D, numSteps, tau);

        if (model == null) {
            System.out.printf("  [LEVEL %d] GNG training returned null (e.g., not enough samples for initial neurons).%n", level);
            return null;
        }

        // GNGモデルのパラメータをGHNGModelオブジェクトにセット
        model.maxUnits = maxNeurons;
        model.lambda = lambda;
        model.epsilonB = epsilonB;
        model.epsilonN = epsilonN;
        model.alpha = alpha;
        model.aMax = aMax;
        model.D = D;
        model.numSteps = numSteps;

        // アクティブなニューロンのインデックスを特定
        List<Integer> activeNeuronIndices = new ArrayList<>();
        for (int i = 0; i < model.maxUnits; i++) {
            if (!Double.isNaN(model.means[0][i])) {
                activeNeuronIndices.add(i);
            }
        }

        System.out.printf("  [LEVEL %d] GNG Training Complete. Final active neurons: %d%n", level, activeNeuronIndices.size());

        // 各ニューロンの勝利回数を計算（この階層での密度情報）
        Map<Integer, Integer> currentLevelWinCounts = calculateNeuronWinCounts(model);

        System.out.printf("  [LEVEL %d] Neuron Details:%n", level);
        for (int neuronIdx : activeNeuronIndices) {
            StringBuilder coords = new StringBuilder("(");
            for (int d = 0; d < dimension; d++) {
                coords.append(String.format("%.2f", model.means[d][neuronIdx]));
                if (d < dimension - 1) {
                    coords.append(", ");
                }
            }
            coords.append(")");
            System.out.printf("    Neuron %d: Coords=%s, Wins=%d%n",
                              neuronIdx, coords.toString(), currentLevelWinCounts.getOrDefault(neuronIdx, 0));
        }

        // PRUNE THE GRAPHS WITH ONLY 2 NEURONS. THIS IS TO SIMPLIFY THE HIERARCHY
        // MATLABコードのLevel==2でのプルーニング条件 (numel(NdxNeurons)==2) に対応
        // ただし、階層GHNGとして、もし二つのニューロンしかなく、かつ子レベルへの拡張が無意味な場合。
        // ここでは、GNG結果が2つのニューロンになった場合、それ以上階層を掘り下げないという単純な解釈。
        if (activeNeuronIndices.size() == 2) {
             System.out.printf("  [LEVEL %d] Pruned: Graph has exactly 2 active neurons. Hierarchy simplified/stopped.%n", level);
             // MATLABコードのように、Modelを空にする代わりに、子モデルを生成しないことで階層を停止させる
             return model; // モデル自体は返す (上位階層がその存在を知るため)
        }


        // Expansion Process (子モデルの再帰的訓練)
        System.out.printf("  [LEVEL %d] Expanding to child levels...%n", level);
        for (int ndxNeuro : activeNeuronIndices) {
            List<double[]> childFeaturesList = new ArrayList<>();
            int[] childOriginalLabels = new int[numSamples]; // Child samples will inherit their original labels
            int childOriginalLabelCount = 0;

            for (int i = 0; i < numSamples; i++) {
                if (model.winners[i] == ndxNeuro) {
                    double[] feature = new double[dimension];
                    for (int d = 0; d < dimension; d++) {
                        feature[d] = features[d][i];
                    }
                    childFeaturesList.add(feature);
                    childOriginalLabels[childOriginalLabelCount++] = originalLabels[i];
                }
            }

            if (!childFeaturesList.isEmpty()) {
                double[][] childFeatures = new double[dimension][childFeaturesList.size()];
                int[] actualChildLabels = new int[childOriginalLabelCount]; // Resize to actual count
                for (int i = 0; i < childFeaturesList.size(); i++) {
                    for (int d = 0; d < dimension; d++) {
                        childFeatures[d][i] = childFeaturesList.get(i)[d];
                    }
                    actualChildLabels[i] = childOriginalLabels[i];
                }
                // 再帰呼び出し: 次のレベルへ
                model.children.put(ndxNeuro, trainGHNG(childFeatures, actualChildLabels, epochs, maxNeurons,
                        tau, lambda, epsilonB, epsilonN, alpha, aMax, D, level + 1));
            } else {
                System.out.printf("    [LEVEL %d] Neuron %d has no associated samples for child level.%n", level, ndxNeuro);
            }
        }
        System.out.printf("---%nLEVEL %d: Expansion Complete.---%n", level);
        return model;
    }


    /**
     * GNG (Growing Neural Gas) モデルを訓練します。
     * GHNGTrainerクラスのプライベートメソッドとして、GHNGTrainerからのみ呼び出される。
     */
    private GHNGModel trainGNG(double[][] features, int maxUnits, int lambda,
                               double epsilonB, double epsilonN, double alpha,
                               int aMax, double D, int numSteps, double tau) {

        int dimension = features.length;
        int numSamples = features[0].length;

        GHNGModel model = new GHNGModel(dimension, maxUnits, numSamples);
        // パラメータをモデルに保存（GHNGModelが単独でも情報を持つように）
        model.lambda = lambda;
        model.epsilonB = epsilonB;
        model.epsilonN = epsilonN;
        model.alpha = alpha;
        model.aMax = aMax;
        model.D = D;
        model.numSteps = numSteps;

        // 初期化 (2つのユニットとそれらの間の接続)
        if (numSamples < 2) { // 2つ以上のサンプルがないと初期化できない
            System.err.println("  [GNG] Not enough samples to initialize 2 neurons.");
            return null;
        }
        int sampleIdx1 = rand.nextInt(numSamples);
        int sampleIdx2;
        do {
            sampleIdx2 = rand.nextInt(numSamples);
        } while (sampleIdx2 == sampleIdx1);

        for (int d = 0; d < dimension; d++) {
            model.means[d][0] = features[d][sampleIdx1];
            model.means[d][1] = features[d][sampleIdx2];
        }
        model.connections[0][1] = aMax;
        model.connections[1][0] = aMax;

        // サンプルのランダム順列 (エポックごとにシャッフル)
        List<Integer> sampleNdxs = IntStream.range(0, numSamples).boxed().collect(Collectors.toList());
        Collections.shuffle(sampleNdxs, rand);

        boolean growing = true; // 成長停止フラグ
        GHNGModel oldModel = null; // 成長停止チェック用のモデルスナップショット

        // メインループ
        for (int ndxStep = 1; ndxStep <= numSteps; ndxStep++) {
            int currentSampleIndex = sampleNdxs.get((ndxStep - 1) % numSamples);
            double[] currentFeature = new double[dimension];
            for (int d = 0; d < dimension; d++) {
                currentFeature[d] = features[d][currentSampleIndex];
            }

            // 最も近いユニット (S1) と2番目に近いユニット (S2) を決定
            double[] squaredDistances = new double[maxUnits];
            for (int i = 0; i < maxUnits; i++) {
                if (Double.isNaN(model.means[0][i])) {
                    squaredDistances[i] = Double.POSITIVE_INFINITY;
                    continue;
                }
                double dist = 0.0;
                for (int d = 0; d < dimension; d++) {
                    dist += Math.pow(model.means[d][i] - currentFeature[d], 2);
                }
                squaredDistances[i] = dist;
            }

            int s1 = -1, s2 = -1;
            double minDist1 = Double.POSITIVE_INFINITY;
            double minDist2 = Double.POSITIVE_INFINITY;

            for (int i = 0; i < maxUnits; i++) {
                if (squaredDistances[i] < minDist1) {
                    minDist2 = minDist1;
                    s2 = s1;
                    minDist1 = squaredDistances[i];
                    s1 = i;
                } else if (squaredDistances[i] < minDist2) {
                    minDist2 = squaredDistances[i];
                    s2 = i;
                }
            }

            if (s1 == -1) { // アクティブなニューロンが一つも見つからなかった場合（初期段階などで）
                continue;
            }
            if (s2 == -1 && activeNeuronCount(model) >= 2) { // 2番目のBMUが見つからなかったが、複数ニューロンがある場合
                // これは発生すべきではないが、頑健性のためにチェック
                continue;
            }


            model.winners[currentSampleIndex] = s1;

            // S1から伸びる全てのエッジの時間を減少
            for (int i = 0; i < maxUnits; i++) {
                if (model.connections[s1][i] > 0) {
                    model.connections[s1][i]--;
                    if (model.connections[s1][i] == 0) { // 寿命が尽きた接続は双方向で削除
                        model.connections[i][s1] = 0;
                    }
                }
                // connections[i][s1] は上のifで双方向で処理されるので不要
            }


            // S1の誤差カウンタに誤差を加算
            model.errors[s1] += squaredDistances[s1];

            // S1とそのトポロジカルな隣接ユニットをサンプルに向かって移動
            for (int d = 0; d < dimension; d++) {
                model.means[d][s1] = (1 - epsilonB) * model.means[d][s1] + epsilonB * currentFeature[d];
            }

            List<Integer> neighbors = new ArrayList<>();
            for (int i = 0; i < maxUnits; i++) {
                if (model.connections[s1][i] > 0) { // S1の有効な隣接ユニットを見つける
                    neighbors.add(i);
                }
            }

            for (int neighborIdx : neighbors) {
                for (int d = 0; d < dimension; d++) {
                    model.means[d][neighborIdx] = (1 - epsilonN) * model.means[d][neighborIdx] + epsilonN * currentFeature[d];
                }
            }

            // S1とS2間の接続を生成または更新
            if (s2 != -1) { // 2番目のBMUが存在する場合のみ接続
                model.connections[s1][s2] = aMax;
                model.connections[s2][s1] = aMax;
            }

            // エッジを持たないユニットを削除
            for (int i = 0; i < maxUnits; i++) {
                boolean hasEdges = false;
                for (int j = 0; j < maxUnits; j++) {
                    if (model.connections[i][j] > 0 || model.connections[j][i] > 0) { // iからj、jからi両方向をチェック
                        hasEdges = true;
                        break;
                    }
                }
                if (!hasEdges && !Double.isNaN(model.means[0][i])) { // エッジがなく、かつアクティブなニューロンなら削除
                    for (int d = 0; d < dimension; d++) {
                        model.means[d][i] = Double.NaN;
                    }
                    model.errors[i] = 0;
                }
            }


            // ユニット生成 (グラフが成長可能でLambdaステップごと)
            if (ndxStep % lambda == 0 && growing) {
                int currentActiveNeurons = activeNeuronCount(model);
                if (currentActiveNeurons > 0 && currentActiveNeurons <= maxUnits) {
                    double currentTotalError = 0;
                    for (int i = 0; i < maxUnits; i++) {
                        if (!Double.isNaN(model.means[0][i])) {
                            currentTotalError += model.errors[i];
                        }
                    }
                    model.mqe[currentActiveNeurons] = currentTotalError / currentActiveNeurons;
                }
                oldModel = model.deepCopy(); // 現在のモデルをスナップショット

                // 最も誤差の大きいユニットを見つける
                double maxError = -1.0;
                int ndxMaxError = -1;
                for (int i = 0; i < maxUnits; i++) {
                    if (!Double.isNaN(model.means[0][i]) && model.errors[i] > maxError) {
                        maxError = model.errors[i];
                        ndxMaxError = i;
                    }
                }

                if (ndxMaxError != -1) {
                    // 最も誤差の大きいユニットの、最も誤差の大きい隣接ユニットを見つける
                    double maxNeighborError = -1.0;
                    int ndxNeighbor = -1;
                    for (int i = 0; i < maxUnits; i++) {
                        // 接続があり、かつ有効なニューロンで、誤差が最大
                        if (model.connections[ndxMaxError][i] > 0 && !Double.isNaN(model.means[0][i]) && model.errors[i] > maxNeighborError) {
                            maxNeighborError = model.errors[i];
                            ndxNeighbor = i;
                        }
                    }

                    // 新しいユニットを生成（可能な場合）
                    int ndxNewUnit = -1;
                    for (int i = 0; i < maxUnits; i++) {
                        if (Double.isNaN(model.means[0][i])) { // 最初の空きスロットを見つける
                            ndxNewUnit = i;
                            break;
                        }
                    }

                    if (ndxNewUnit != -1 && ndxNeighbor != -1 && currentActiveNeurons < maxUnits) { // 新しいユニットのスロットがあり、隣接ユニットも有効で、最大ユニット数に達していない
                        // 新しいプロトタイプベクトルを設定
                        for (int d = 0; d < dimension; d++) {
                            model.means[d][ndxNewUnit] = 0.5 * (model.means[d][ndxMaxError] + model.means[d][ndxNeighbor]);
                        }

                        // 古い2つのユニット間の接続を削除
                        model.connections[ndxMaxError][ndxNeighbor] = 0;
                        model.connections[ndxNeighbor][ndxMaxError] = 0;

                        // 新しいユニットと古い2つのユニット間に接続を作成
                        model.connections[ndxNewUnit][ndxMaxError] = aMax;
                        model.connections[ndxMaxError][ndxNewUnit] = aMax;
                        model.connections[ndxNewUnit][ndxNeighbor] = aMax;
                        model.connections[ndxNeighbor][ndxNewUnit] = aMax;

                        // 古いユニットの誤差を減らし、新しいユニットの誤差を設定
                        model.errors[ndxMaxError] *= alpha;
                        model.errors[ndxNeighbor] *= alpha;
                        model.errors[ndxNewUnit] = model.errors[ndxMaxError]; // MATLABの動作に合わせる
                    }
                }
            }

            // 成長チェック (2*Lambdaステップごと)
            if (ndxStep % (2 * lambda) == (int) Math.floor(3.0 * lambda / 2.0)) {
                int numNeurons = activeNeuronCount(model);
                if (numNeurons > 0 && numNeurons <= maxUnits) {
                    double currentTotalError = 0;
                    for (int i = 0; i < maxUnits; i++) {
                        if (!Double.isNaN(model.means[0][i])) {
                            currentTotalError += model.errors[i];
                        }
                    }
                    model.mqe[numNeurons] = currentTotalError / numNeurons;
                }

                int oldNumNeurons = 0;
                if (oldModel != null) {
                    oldNumNeurons = activeNeuronCount(oldModel);
                }

                if (numNeurons > oldNumNeurons && oldModel != null && oldNumNeurons > 0) {
                    double improvement = (oldModel.mqe[oldNumNeurons] - model.mqe[numNeurons]) / Math.abs(oldModel.mqe[oldNumNeurons]);
                    if (improvement < tau) {
                        System.out.printf("  [GNG-GrowthCheck] Improvement (%.4f) below Tau (%.4f). Stopping growth.%n", improvement, tau);
                        System.out.printf("  [GNG-GrowthCheck] Old Num Neurons: %d, Current Num Neurons: %d%n", oldNumNeurons, numNeurons);
                        model = oldModel; // モデルを以前のスナップショットに戻す
                        growing = false; // 成長を停止
                    }
                }
            }

            // 全ての誤差変数をDで減衰
            for (int i = 0; i < maxUnits; i++) {
                model.errors[i] *= D;
            }
        }
        return model;
    }

    /**
     * アクティブなニューロンの数をカウントするヘルパーメソッド。
     * @param model GHNGModel
     * @return アクティブなニューロンの数
     */
    private int activeNeuronCount(GHNGModel model) {
        int count = 0;
        for (int i = 0; i < model.maxUnits; i++) {
            if (!Double.isNaN(model.means[0][i])) {
                count++;
            }
        }
        return count;
    }

    /**
     * 指定されたGHNGModelの、各アクティブニューロンの勝利回数を計算します。
     * @param model 計算対象のGHNGModel。trainGNGまたはtrainGHNGの出力。
     * @return 各ニューロンのインデックスとその勝利回数をマッピングしたMap。
     */
    public Map<Integer, Integer> calculateNeuronWinCounts(GHNGModel model) {
        Map<Integer, Integer> winCounts = new HashMap<>();

        if (model == null || model.winners == null) {
            return winCounts;
        }

        Set<Integer> activeNeurons = new HashSet<>();
        for (int i = 0; i < model.maxUnits; i++) {
            if (!Double.isNaN(model.means[0][i])) {
                activeNeurons.add(i);
            }
        }

        for (int winnerNeuronIndex : model.winners) {
            if (activeNeurons.contains(winnerNeuronIndex)) {
                winCounts.put(winnerNeuronIndex, winCounts.getOrDefault(winnerNeuronIndex, 0) + 1);
            }
        }
        return winCounts;
    }
}