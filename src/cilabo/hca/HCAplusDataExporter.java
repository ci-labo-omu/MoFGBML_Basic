package cilabo.hca;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;

public class HCAplusDataExporter {
    
    public int countNumNodes(HCAplusNet net) {
        if (net == null) {
            return 0;
        }

        int count = net.numNodes;
        
        Queue<HCAplusNet> queue = new LinkedList<>();
        for (HCAplusNet child : net.Child.values()) {
            queue.add(child);
        }
        
        while (!queue.isEmpty()) {
            HCAplusNet currentModel = queue.poll();
            count += currentModel.numNodes;
            
            for (HCAplusNet child : currentModel.Child.values()) {
                queue.add(child);
            }
        }
        return count;
    }
    public void exportLeavesNet(HCAplusNet hcaNet, String filePath, int numClasses) {
        if (hcaNet == null) {
            System.err.println("Error: Input HCAplusNet model is null.");
            return;
        }

        List<LeafNodeInfo> leaves = new ArrayList<>();
        collectLeavesRecursive(hcaNet, leaves);

        if (leaves.isEmpty()) {
            System.out.println("Warning: No leaf nodes found in the model. Skipping export.");
            return;
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            // ヘッダ行: パターン数, 属性数, クラス数
            int numLeaves = leaves.size();
            int numDims = leaves.get(0).weight.length;
            writer.write(String.format("%d,%d,%d", numLeaves, numDims, numClasses));
            writer.newLine();

            // データ行: 属性, クラス, 密度情報
            for (LeafNodeInfo leaf : leaves) {
                StringBuilder line = new StringBuilder();
                // 属性
                for (double feature : leaf.weight) {
                    line.append(feature).append(",");
                }
                // クラス
                line.append(leaf.label).append(",");
                // 密度情報
                line.append(leaf.density);
                writer.write(line.toString());
                writer.newLine();
            }
            System.out.printf("Exported %d leaf nodes to %s.%n", numLeaves, filePath);

        } catch (IOException e) {
            System.err.printf("Error exporting leaf nodes to %s: %s%n", filePath, e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * HCA+モデルを再帰的に探索し、葉ノードを収集するヘルパーメソッド。
     * * @param model 現在の階層のモデル
     * @param leaves 収集した葉ノードを格納するリスト
     */
    private void collectLeavesRecursive(HCAplusNet model, List<LeafNodeInfo> leaves) {
        if (model == null) {
            return;
        }

        // Childマップが空、または子モデルがすべてnullの場合、現在のモデルのノードが葉ノード
        if (model.Child == null || model.Child.isEmpty() || model.Child.values().stream().allMatch(c -> c == null)) {
            // 現在のモデルのノードを葉ノードとして収集
            for (int i = 0; i < model.numNodes; i++) {
                // 有効なノードのみを対象
                if (model.CountNode != null && i < model.CountNode.length && model.CountNode[i] > 0) {
                    // クラスラベルはCountLabelから多数決で決定
                    int classLabel = -1;
                    int maxCount = -1;
                    if (model.CountLabel != null && i < model.CountLabel.length) {
                        for (int j = 0; j < model.CountLabel[i].length; j++) {
                            if (model.CountLabel[i][j] > maxCount) {
                                maxCount = model.CountLabel[i][j];
                                classLabel = j;
                            }
                        }
                    }

                    if (classLabel != -1) {
                         leaves.add(new LeafNodeInfo(model.weight[i], classLabel, model.CountNode[i]));
                    }
                }
            }
        } else {
            // 子モデルが存在する場合、再帰的に探索
            for (HCAplusNet childModel : model.Child.values()) {
                collectLeavesRecursive(childModel, leaves);
            }
        }
    }

    /**
     * 葉ノードの情報を格納するためのヘルパークラス。
     */
    private static class LeafNodeInfo {
        double[] weight;
        int label;
        int density;

        public LeafNodeInfo(double[] weight, int label, int density) {
            this.weight = weight;
            this.label = label;
            this.density = density;
        }
    }
    
    /**
     * HCA+モデルの全階層のニューロンを抽出し、レベルごとに個別のファイルに出力します。
     * @param hcaNet HCA+の訓練済みモデル
     * @param baseFilePath 出力ファイルパスのベース名
     * @param numClasses クラスの総数
     */
    public void exportAllLevels(HCAplusNet hcaNet, String baseFilePath, int numClasses) {
        System.out.println("\n--- Exporting neurons for all levels ---");
        
        // 全レベルのニューロンを一時的に格納するマップ (Level -> List of NeuronInfo)
        Map<Integer, List<LeafNodeInfo>> neuronsByLevel = new HashMap<>();
        
        collectNeuronsPerLevelRecursive(hcaNet, 0, neuronsByLevel, numClasses);
        
        // レベルごとにCSVファイルに出力
        for (Map.Entry<Integer, List<LeafNodeInfo>> entry : neuronsByLevel.entrySet()) {
            int level = entry.getKey();
            List<LeafNodeInfo> neurons = entry.getValue();
            
            String filePath = String.format("%s_level%d.csv", baseFilePath, level);
            
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
                if (neurons.isEmpty()) {
                    System.out.printf("  Level %d has no neurons. Skipping file %s%n", level, filePath);
                    continue;
                }
                
                // ヘッダ行
                int numDims = neurons.get(0).weight.length;
                writer.write(String.format("%d,%d,%d", neurons.size(), numDims, numClasses));
                writer.newLine();

                // データ行
                for (LeafNodeInfo neuron : neurons) {
                    StringBuilder line = new StringBuilder();
                    for (double feature : neuron.weight) {
                        line.append(feature).append(",");
                    }
                    line.append(neuron.label).append(",");
                    line.append(neuron.density);
                    writer.write(line.toString());
                    writer.newLine();
                }
                System.out.printf("  Exported %d neurons for Level %d to %s%n", neurons.size(), level, filePath);
            } catch (IOException e) {
                System.err.printf("Error exporting neurons for Level %d: %s%n", level, e.getMessage());
                e.printStackTrace();
            }
        }
        
        System.out.println("--- Exporting neurons for all levels complete ---");
    }

    /**
     * HCA+モデルを再帰的に探索し、各レベルのノードを収集するヘルパーメソッド。
     * @param model 現在の階層のモデル
     * @param currentLevel 現在の階層レベル
     * @param neuronsByLevel 各レベルのノードを格納するマップ
     * @param numClasses クラス数
     */
    private void collectNeuronsPerLevelRecursive(HCAplusNet model, int currentLevel, Map<Integer, List<LeafNodeInfo>> neuronsByLevel, int numClasses) {
        if (model == null) {
            return;
        }

        // 現在のレベルのノードを収集
        List<LeafNodeInfo> currentLevelNeurons = neuronsByLevel.computeIfAbsent(currentLevel, k -> new ArrayList<>());
        for (int i = 0; i < model.numNodes; i++) {
            if (model.CountNode != null && i < model.CountNode.length && model.CountNode[i] > 0) {
                int classLabel = -1;
                int maxCount = -1;
                if (model.CountLabel != null && i < model.CountLabel.length) {
                    for (int j = 0; j < model.CountLabel[i].length; j++) {
                        if (model.CountLabel[i][j] > maxCount) {
                            maxCount = model.CountLabel[i][j];
                            classLabel = j;
                        }
                    }
                }

                if (classLabel != -1) {
                    currentLevelNeurons.add(new LeafNodeInfo(model.weight[i], classLabel, model.CountNode[i]));
                }
            }
        }

        // 子モデルが存在する場合、再帰的に探索
        if (model.Child != null) {
            for (HCAplusNet childModel : model.Child.values()) {
                collectNeuronsPerLevelRecursive(childModel, currentLevel + 1, neuronsByLevel, numClasses);
            }
        }
    }
    
    // GenerateExcelFromRecordALL.m のロジックもここに実装可能ですが、
    // まずはHCAplusのコア機能である訓練と評価に焦点を当てます。
}