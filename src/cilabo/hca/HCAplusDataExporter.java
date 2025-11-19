package cilabo.hca;

import cilabo.art.HCAplusNet;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.stream.Collectors;

/**
 * HCA+ネットワークのノード情報とツリー構造の統計をエクスポートするユーティリティクラス。
 * MATLABのGenerateExcelFromRecordALL.mや、階層レベルごとのCSV出力の要件を処理します。
 */
public class HCAplusDataExporter {

    private final String outputBaseDir;

    public HCAplusDataExporter(String outputBaseDir) {
        this.outputBaseDir = outputBaseDir;
    }

    /**
     * HCA+の特定のレベルにおけるノード情報（座標、代表クラス、勝利回数）をCSVファイルに出力します。
     * 出力構造は `baseDirPath/filename_suffix.csv` となります。
     *
     * @param currentModel HCA+の特定の階層レベルのモデル
     * @param filePrefix ファイル名のプレフィックス (例: "aN_M_dataset")
     */
    public void exportNodesForHCAplusLevel(HCAplusNet currentModel, String filePrefix) {

        // minCIMごとのサブディレクトリを作成
        String minCIMSubDirPath = this.outputBaseDir + File.separator + 
                                  String.format("minCIM_%d", (int)(currentModel.minCIM * 100));
        String outputFileName = String.format("%s_level%d_nodes_HCAplus.csv", filePrefix, currentModel.level);
        String filePath = minCIMSubDirPath + File.separator + outputFileName;

        File dir = new File(minCIMSubDirPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
            int numNodes = currentModel.numNodes;
            int dimension = currentModel.weights.isEmpty() ? 0 : currentModel.weights.get(0).length;
            
            if (dimension == 0) {
                System.out.printf("Warning: No dimensions found for HCA+ level %d. Skipping export.%n", currentModel.level);
                return;
            }
            
            // 最大クラスラベル数を取得 (CountLabelMatrixの列サイズ-1)
            int maxLabel = 0;
            if (!currentModel.countLabelMatrix.isEmpty()) {
                maxLabel = currentModel.countLabelMatrix.get(0).length - 1; 
            }

            // ヘッダー: ノード数, 次元数, クラスラベルの数
            writer.write(String.format("%d,%d,%d", numNodes, dimension, maxLabel)); 
            writer.newLine();

            // ノードデータ本体の書き出し
            for (int i = 0; i < numNodes; i++) {
                double[] position = currentModel.weights.get(i);
                // model.labelClustersは、CA+トレーニングの最後に決定された代表クラスです
                int classLabel = currentModel.labelClusters.get(i); 
                int count = currentModel.countNodes.get(i);        
                
                StringBuilder sb = new StringBuilder();
                // 属性値（重心座標）
                for (int j = 0; j < position.length; j++) {
                    sb.append(position[j]);
                    sb.append(',');
                }
                // クラスラベル, 勝利回数
                sb.append(classLabel).append(',');
                sb.append(count);
                writer.write(sb.toString());
                writer.newLine();
            }
            //System.out.printf("  Exported %d nodes for Level %d to %s%n", numNodes, currentModel.level, filePath);

        } catch (IOException e) {
            System.err.printf("Error exporting HCA+ nodes to %s: %s%n", filePath, e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * HCA+ツリー内の全ノード数をカウントします (MATLAB: CountNumNodes.mの目的に沿って)。
     * @param net ルートネットワークモデル
     * @return ツリー内のノード総数
     */
    public int countAllNodes(HCAplusNet net) {
        if (net == null) return 0;

        int totalNodes = 0;
        Queue<HCAplusNet> queue = new LinkedList<>();
        queue.add(net); 
        
        while (!queue.isEmpty()) {
            HCAplusNet currentModel = queue.remove();
            
            // このHCAplusNetオブジェクトが持つノードの数を加算
            totalNodes += currentModel.numNodes; 
            
            // 子ノードをキューに追加
            for (HCAplusNet child : currentModel.children) {
                 if (child != null) {
                    queue.add(child);
                 }
            }
        }
        return totalNodes;
    }
}