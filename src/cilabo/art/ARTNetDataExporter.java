package cilabo.art;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ARTNetDataExporter {

    // Helper class to store collected node info
    private static class CollectedNodeInfo {
        double[] position;
        int classLabel;
        int countNode;

        public CollectedNodeInfo(double[] position, int classLabel, int countNode) {
            this.position = position;
            this.classLabel = classLabel;
            this.countNode = countNode;
        }
    }

    public ARTNetDataExporter() {
        // コンストラクタはARTNetTrainerのインスタンスを受け取らないため変更
    }

    /**
     * 各minCIM値ごとにファイルを生成し、そのファイルには全クラスのノードデータを含めます。
     * ノードの座標、クラスラベル、カウント数を指定された形式で出力します。
     *
     * @param modelsByMinCIMAndClass Map<minCIM, Map<classLabel, ARTNetModel>> 形式の訓練済みモデル
     * @param dirPath The directory path to save the file.
     * @param baseFileName The base name for the output files (e.g., "vehicle_nodes_minCIM")
     */
    public void exportCombinedNodeDataByMinCIM(Map<Double, Map<Integer, ARTNetModel>> modelsByMinCIMAndClass, String dirPath, String baseFileName) {
        File dir = new File(dirPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        System.out.println("\n--- Exporting combined ARTNet node data by minCIM ---");

        for (Map.Entry<Double, Map<Integer, ARTNetModel>> minCIMEntry : modelsByMinCIMAndClass.entrySet()) {
            double currentMinCIM = minCIMEntry.getKey();
            Map<Integer, ARTNetModel> modelsForThisMinCIM = minCIMEntry.getValue();
            String filePath = dirPath + File.separator + String.format("%s_%d.csv", baseFileName, (int)(currentMinCIM * 100));

            try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
                int dimension = 0;
                if (!modelsForThisMinCIM.isEmpty()) {
                    ARTNetModel firstModel = modelsForThisMinCIM.values().iterator().next();
                    if (firstModel != null && !firstModel.weights.isEmpty()) {
                        dimension = firstModel.weights.get(0).length;
                    }
                }

                if (dimension == 0) {
                    System.out.printf("Warning: No dimensions found for minCIM %.2f. Skipping file %s%n", currentMinCIM, filePath);
                    continue;
                }

                // ノードを収集
                List<CollectedNodeInfo> allNodesForThisMinCIM = new ArrayList<>();
                Set<Integer> classLabels = new HashSet<>();

                for (Map.Entry<Integer, ARTNetModel> classModelEntry : modelsForThisMinCIM.entrySet()) {
                    int classLabel = classModelEntry.getKey();
                    classLabels.add(classLabel);
                    ARTNetModel model = classModelEntry.getValue();

                    if (model == null) continue;

                    for (int i = 0; i < model.weights.size(); i++) {
                        double[] position = model.weights.get(i);
                        int count = model.countNodes.get(i);
                        if (count > 0) {
                            allNodesForThisMinCIM.add(new CollectedNodeInfo(position, classLabel, count));
                        }
                    }
                }

                // ヘッダー: [ノード数],[次元数],[クラス数]
                writer.write(String.format("%d,%d,%d", allNodesForThisMinCIM.size(), dimension, classLabels.size()));
                writer.newLine();

                // データ行
                for (CollectedNodeInfo node : allNodesForThisMinCIM) {
                    StringBuilder sb = new StringBuilder();
                    for (int j = 0; j < node.position.length; j++) {
                        sb.append(String.format("%.9f", node.position[j])).append(',');
                    }
                    sb.append(node.classLabel).append(',');
                    sb.append(node.countNode);
                    writer.write(sb.toString());
                    writer.newLine();
                }

                System.out.printf("  Exported %d nodes for minCIM %.2f to %s%n", allNodesForThisMinCIM.size(), currentMinCIM, filePath);

            } catch (IOException e) {
                System.err.printf("Error exporting nodes for minCIM %.2f: %s%n", currentMinCIM, e.getMessage());
                e.printStackTrace();
            }
        }

        System.out.println("--- Exporting combined ARTNet node data by minCIM complete ---");
    }


    // --- 以前のARTNetDataExporterのメソッドは役割が異なるため削除または変更 ---
    // exportNodeData (minCIM_class_file) は不要になる
    // exportMoFGBMLPatternFile (minCIM_class_file) も不要になる

    // ARTNetTrainerインスタンスが不要になるためコンストラクタは引数なしに変更。
    // そのため ARTNetTrainer trainer; は削除
    // また、TrainerのcalculateNeuronWinCountsに相当するものはARTNetTrainerにはなく、
    // model.countNodesに直接入っているので、ARTNetDataExporterからは不要です。
    // したがって ARTNetDataExporter のコンストラクタから trainer は不要です。
}