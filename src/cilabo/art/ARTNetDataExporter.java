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

    // Helper class to store collected node info (変更なし)
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
        // コンストラクタはARTNetTrainerのインスタンスを受け取らないため変更なし
    }

    /**
     * ARTNetモデルのニューロン情報（座標、クラスラベル、勝利回数）をCSVファイルに出力します。
     * 出力構造は `baseDirPath/minCIM_XX/filename_suffix.csv` となります。
     * ノードの座標、クラスラベル、カウント数を指定された形式で出力します。
     *
     * @param modelsForThisFile 各クラスのARTNetModel情報（classLabel -> ARTNetModel）
     * @param currentMinCIM     現在処理中のminCIM値
     * @param baseDirPath       出力ルートディレクトリ (例: "output_data")
     * @param filenameSuffix    ファイル名のサフィックス (例: "a0_0_vehicle_nodes_ART.csv")
     */
    public void exportNodesForSingleFile(
            Map<Integer, ARTNetModel> modelsForThisFile,
            double currentMinCIM,
            String baseDirPath,
            String filenameSuffix) {

        // minCIMごとのサブディレクトリを作成
        String minCIMSubDirPath = baseDirPath + File.separator + String.format("minCIM_%d", (int)(currentMinCIM * 100));
        File dir = new File(minCIMSubDirPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        String filePath = minCIMSubDirPath + File.separator + filenameSuffix;

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            // ヘッダーの準備: 最初のモデルから次元数を取得
            int dimension = 0;
            if (!modelsForThisFile.isEmpty()) {
                ARTNetModel firstModel = modelsForThisFile.values().iterator().next();
                if (firstModel != null && !firstModel.weights.isEmpty()) {
                    dimension = firstModel.weights.get(0).length;
                }
            }
            
            if (dimension == 0) {
                System.out.printf("Warning: No dimensions found for minCIM %.2f, file %s. Skipping export.%n", currentMinCIM, filePath);
                return; // 次元が取得できない場合はスキップ
            }


            
            Set<Integer> classLabels = new HashSet<>();
            // このminCIMにおける全クラスのノードを収集
            List<CollectedNodeInfo> allNodesForThisFile = new ArrayList<>();
            for (Map.Entry<Integer, ARTNetModel> classModelEntry : modelsForThisFile.entrySet()) {
                int classLabel = classModelEntry.getKey();
                classLabels.add(classLabel);
                ARTNetModel model = classModelEntry.getValue();

                if (model == null) {
                    continue;
                }

                for (int i = 0; i < model.weights.size(); i++) {
                    double[] position = model.weights.get(i);
                    int count = model.countNodes.get(i);
                    // 勝利カウントが0のノードは含めない
                    if (count > 0) {
                        allNodesForThisFile.add(new CollectedNodeInfo(position, classLabel, count));
                    }
                }
            }
            
            
            writer.write(String.format("%d,%d,%d", allNodesForThisFile.size(), dimension, classLabels.size()));
            writer.newLine();
            // 収集した全てのノードをファイルに書き出す
            for (CollectedNodeInfo node : allNodesForThisFile) {
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < node.position.length; j++) {
                    sb.append(node.position[j]);
                    sb.append(',');
                }
                sb.append(node.classLabel).append(',');
                sb.append(node.countNode);
                writer.write(sb.toString());
                writer.newLine();
            }
            System.out.printf("  Exported %d nodes to %s%n", allNodesForThisFile.size(), filePath);

        } catch (IOException e) {
            System.err.printf("Error exporting nodes to %s: %s%n", filePath, e.getMessage());
            e.printStackTrace();
        }
    }
}