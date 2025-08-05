package cilabo.ghng;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GHNGDataExporter {

    private final GHNGTrainer trainer;

    public GHNGDataExporter(GHNGTrainer trainer) {
        this.trainer = trainer;
    }

    /**
     * GHNGモデルのニューロン情報（座標、クラスラベル、勝利回数）をCSVファイルに出力します。
     * - 最もニューロン数が多い階層レベルのニューロンのみを出力します。
     * - 勝利回数 (wins) が0のニューロンは出力しません。
     * - 列の順序は「属性値, ..., クラスラベル, wins数」です。
     *
     * @param modelInfosByClass 各クラスのGHNGModel情報（クラスラベル -> モデル）
     * @param filePath 出力ファイルパス
     */
    /**
     * 各クラスのGHNGモデルのニューロン情報を、階層レベルごとに個別のCSVファイルに出力します。
     * ファイル名は "[baseFileName]_class[label]_level[level].csv" の形式になります。
     * 勝利回数 (wins) が0のニューロンは出力しません。
     *
     * @param modelInfosByClass 各クラスのGHNGModel情報（クラスラベル -> モデル）
     * @param baseFilePath 基となるファイルパス (例: "output/ghng_neurons")
     */
    /**
     * 各クラスのGHNGモデルのニューロン情報を、階層レベルごとに個別のCSVファイルに出力します。
     * ファイル名は "[baseFileName]_class[label]_level[level].csv" の形式になります。
     * 勝利回数 (wins) が0のニューロンは出力しません。
     *
     * @param modelInfosByClass 各クラスのGHNGModel情報（クラスラベル -> モデル）
     * @param baseFilePath 基となるファイルパス (例: "output/ghng_neurons")
     */
    public void exportNeuronsByLevel(Map<Integer, GHNGModel> modelInfosByClass, String baseFilePath) {
        System.out.println("\n--- Exporting combined neurons for each level ---");

        // まず、全モデルから最大のレベル深さを特定
        int maxDepth = 0;
        for (GHNGModel model : modelInfosByClass.values()) {
            Map<Integer, Integer> neuronCountsByLevel = new HashMap<>();
            countNeuronsPerLevel(model, 0, neuronCountsByLevel);
            for (int level : neuronCountsByLevel.keySet()) {
                if (level > maxDepth) {
                    maxDepth = level;
                }
            }
        }

        // 各レベルごとにファイルを作成
        for (int level = 0; level <= maxDepth; level++) {
            String filePath = String.format("%s_level%d.csv", baseFilePath, level);
            List<NeuronInfo> neuronsForThisLevel = new ArrayList<>();
            int dimension = -1; // 次元数を初期化

            // 各クラスのモデルから、現在のレベルのニューロンを収集
            for (Map.Entry<Integer, GHNGModel> entry : modelInfosByClass.entrySet()) {
                int classLabel = entry.getKey();
                GHNGModel model = entry.getValue();

                if (model == null) {
                    continue;
                }
                
                // 最適なレベル（今回は 'level' 変数）のニューロンを収集
                // collectOptimalLevelNeurons は指定レベルのニューロンを収集するのに使える
                collectOptimalLevelNeurons(model, 0, level, classLabel, neuronsForThisLevel);

                // 次元数を一度取得（全てのモデルで次元は同じはず）
                if (dimension == -1 && model.means != null && model.means.length > 0) {
                    dimension = model.means.length;
                }
            }

            // このレベルのニューロンがあればファイルに出力
            if (!neuronsForThisLevel.isEmpty()) {
                try (BufferedWriter writer = new BufferedWriter(new FileWriter(filePath))) {
                    // ヘッダーの準備
                    StringBuilder header = new StringBuilder();
                    for (int d = 0; d < dimension; d++) {
                        header.append("attr").append(d + 1).append(",");
                    }
                    header.append("class_label,wins_count");
                    writer.write(header.toString());
                    writer.newLine();

                    for (NeuronInfo neuron : neuronsForThisLevel) {
                        StringBuilder line = new StringBuilder();
                        for (double coord : neuron.coordinates) {
                            line.append(coord).append(",");
                        }
                        line.append(neuron.classLabel).append(",");
                        line.append(neuron.winCount);

                        writer.write(line.toString());
                        writer.newLine();
                    }
                    System.out.printf("  Exported %d neurons for Level %d (all classes) to %s%n", neuronsForThisLevel.size(), level, filePath);

                } catch (IOException e) {
                    System.err.printf("Error exporting neurons for Level %d: %s%n", level, e.getMessage());
                    e.printStackTrace();
                }
            } else {
                System.out.printf("  No neurons found for Level %d (all classes). Skipping export for this level.%n", level);
            }
        }
        System.out.println("--- Exporting combined neurons by level complete ---");
    }



    /**
     * 各階層レベルのアクティブニューロン数を再帰的にカウントします。
     * @param model 現在のレベルのモデル
     * @param level 現在の階層レベル
     * @param countsByLevel 各レベルのカウントを格納するマップ
     */
    private void countNeuronsPerLevel(GHNGModel model, int level, Map<Integer, Integer> countsByLevel) {
        if (model == null) {
            return;
        }

        int activeCount = 0;
        for (int i = 0; i < model.maxUnits; i++) {
            if (!Double.isNaN(model.means[0][i])) {
                activeCount++;
            }
        }
        countsByLevel.put(level, countsByLevel.getOrDefault(level, 0) + activeCount);

        // 子モデルがあれば再帰的にカウント
        for (GHNGModel childModel : model.children.values()) {
            countNeuronsPerLevel(childModel, level + 1, countsByLevel);
        }
    }


    /**
     * 最適なレベルのニューロンのみをCSVファイルに書き込むヘルパーメソッド。
     * @param model 現在のレベルのモデル
     * @param writer ファイルライター
     * @param currentLevel 現在の階層レベル
     * @param targetLevel 出力対象の最適な階層レベル
     * @param classLabel ニューロンが訓練されたクラスラベル
     * @throws IOException
     */


   public void saveGHNGModels(Map<Integer, GHNGModel> modelInfosByClass, String filePath) {
       try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filePath))) {
           oos.writeObject(modelInfosByClass); // マップ全体をシリアライズ
           System.out.println("GHNG Models saved to: " + filePath);
       } catch (IOException e) {
           System.err.println("Error saving GHNG Models: " + e.getMessage());
           e.printStackTrace();
       }
   }
   private void collectOptimalLevelNeurons(GHNGModel model, int currentLevel, int targetLevel, int classLabel, List<NeuronInfo> collectedNeurons) {
       if (model == null) {
           return;
       }

       // 現在のレベルがターゲットレベルと一致する場合、ニューロンを収集
       if (currentLevel == targetLevel) {
           Map<Integer, Integer> currentLevelWinCounts = trainer.calculateNeuronWinCounts(model);

           for (int i = 0; i < model.maxUnits; i++) {
               if (!Double.isNaN(model.means[0][i])) { // アクティブなニューロンのみ
                   int winCount = currentLevelWinCounts.getOrDefault(i, 0);
                   if (winCount > 0) { // 勝利回数が0より大きいもののみ収集
                       double[] coords = new double[model.means.length];
                       for (int d = 0; d < model.means.length; d++) {
                           coords[d] = model.means[d][i];
                       }
                       collectedNeurons.add(new NeuronInfo(coords, classLabel, winCount));
                   }
               }
           }
       } else if (currentLevel < targetLevel) { // 現在のレベルがターゲットレベルより低い場合、子モデルを深く探索
           for (GHNGModel childModel : model.children.values()) {
               collectOptimalLevelNeurons(childModel, currentLevel + 1, targetLevel, classLabel, collectedNeurons);
           }
       }
       // currentLevel > targetLevel の場合は何もしない（ターゲットレベルより深い階層は探索しない）
   }
   private static class NeuronInfo {
       double[] coordinates;
       int classLabel;
       int winCount;

       public NeuronInfo(double[] coordinates, int classLabel, int winCount) {
           this.coordinates = coordinates;
           this.classLabel = classLabel;
           this.winCount = winCount;
       }
   }
    // GHNGModelオブジェクトをファイルに保存/読み込み (変更なし)
    @SuppressWarnings("unchecked") // 型安全でないキャスト警告を抑制
    public Map<Integer, GHNGModel> loadGHNGModels(String filePath) {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filePath))) {
            Map<Integer, GHNGModel> models = (Map<Integer, GHNGModel>) ois.readObject();
            System.out.println("GHNG Models loaded from: " + filePath);
            return models;
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading GHNG Models: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }}