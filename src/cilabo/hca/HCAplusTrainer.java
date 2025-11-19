package cilabo.hca;

import java.util.ArrayList;
import java.util.List;
// 依存クラス (適切なパッケージに修正してください)
import cilabo.ghng.Sample;
import cilabo.art.ARTNetTrainer;
import cilabo.art.HCAplusNet; 

/**
 * HCA+の再帰的トレーニングを実装するクラス。
 * MATLABのTrainHCAplus_Classification.mに対応します。
 */
public class HCAplusTrainer {
    private final ARTNetTrainer caPlusTrainer;

    public HCAplusTrainer(ARTNetTrainer caPlusTrainer) {
        this.caPlusTrainer = caPlusTrainer;
    }

    /**
     * HCA+の再帰的な訓練メソッド。
     * @param samples 現在のノードに割り当てられたサンプル (特徴量とラベルを含む)
     * @param net ネットワークパラメータと現在のレベル情報を含むHCAplusNet (Modelの基盤)
     * @return 訓練済みのHCAplusNet。階層が停止または枝刈りされた場合は null を返します。
     */
    public HCAplusNet trainRecursive(List<Sample> samples, HCAplusNet net) {
        // 1. 最大レベルチェック (MATLAB: if Level > MaxLevel)
        if (net.level > net.maxLevel) {
            return null;
        }

        // 2. Growing Process (単一層のCA+学習)
        // netオブジェクトに直接結果を書き込む (weights, numNodes, winnersなど)
        // ARTNetTrainer.artClusteringTrain は List<Sample> を受け取る実装に変更済みと仮定
        caPlusTrainer.artClusteringTrain(net, samples);
        HCAplusNet currentModel = net;
        if (currentModel.winners.isEmpty() && currentModel.numNodes > 0) {
            // ノードが生成されたのに winners が空であれば、ARTNetTrainerにバグがある
            System.err.println("FATAL: ARTNetTrainer failed to record winners!");
            // エラーを投げるか、nullを返して処理を中止すべき
            return null; 
        }
        // 3. Prune small clusters (枝刈り - MATLAB: if NumNeurons <= 2 || NumSamples == NumNeurons)
        int numNeurons = currentModel.numNodes;       // MATLAB: NumNeurons
        int numSamples = samples.size();              // MATLAB: NumSamples
        
        // 枝刈りロジック: ニューロンが少なすぎるか、サンプル数とニューロン数が一致する場合
        if (numNeurons <= 2 || numSamples == numNeurons) {
        //    System.out.println("  [Pruning] Level " + currentModel.level + " pruned: NumNeurons=" + numNeurons + ", NumSamples=" + numSamples);
           return null; // ノードを破棄
        }
        System.out.println("  [Training] Level " + currentModel.level + " completed: NumNeurons=" + numNeurons + ", NumSamples=" + numSamples);
        
        // 4. Expansion Process (再帰的展開)
        List<Integer> winners = currentModel.winners;
        currentModel.children = new ArrayList<>();
        // numNeuronsは1-basedのインデックスを意味する (1から開始)
        for (int neuronIndex = 1; neuronIndex <= numNeurons; neuronIndex++) {
            
            // 子ノードに割り当てるサンプルを抽出
            List<Sample> childSamples = new ArrayList<>();
            for (int i = 0; i < numSamples; i++) {
                // winnersは1-basedで保存されている
                if (winners.get(i) == neuronIndex) { 
                    childSamples.add(samples.get(i));
                }
            }

            // MATLAB: if ~isempty(ChildSamples)
            if (!childSamples.isEmpty()) {
                // 次のレベルの訓練のための新しいネットワークオブジェクトを作成
                HCAplusNet childNet = new HCAplusNet(currentModel.lambda, currentModel.minCIM, currentModel.maxLevel);
                childNet.level = currentModel.level + 1; // レベルを進める
               
                
                // 再帰呼び出し
                HCAplusNet childModel = trainRecursive(childSamples, childNet);
                
                // 子ノードの結果を保存 (nullもそのまま追加し、階層構造を維持)
                currentModel.children.add(childModel);
            } else {
                // サンプルが割り当てられなかったニューロンには null を子として設定
                currentModel.children.add(null); 
            }
        }
        
        return currentModel;
    }
}