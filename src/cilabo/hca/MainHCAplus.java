package cilabo.hca;
import java.util.LinkedList; // Queueの実装に使用
import java.util.Queue;      // ツリー探索に使用
import cilabo.ghng.Sample;
import cilabo.art.ARTNetTrainer; // 既存のCA+ロジック
import cilabo.art.HCAplusNet;      // 拡張されたHCA+モデル
// 必要に応じて他のHCA+関連ユーティリティをインポート
// import cilabo.hca.util.HCAplusDataExporter;
// import cilabo.hca.HCAplusEvaluator;
// import cilabo.hca.HCAplusManager;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class MainHCAplus {

    // --- 補助クラス: DataLoadResult (ヘッダ情報とデータを保持) ---
    private static class DataLoadResult {
        final int numSamples;
        final int numDims;
        final int numClasses;
        final List<double[]> dataLines;

        public DataLoadResult(int numSamples, int numDims, int numClasses, List<double[]> dataLines) {
            this.numSamples = numSamples;
            this.numDims = numDims;
            this.numClasses = numClasses;
            this.dataLines = dataLines;
        }
    }

    // --- 補助クラス: DataHelper (データ読み込み/変換ロジック) ---
    private static class DataHelper {
        public static DataLoadResult loadRawDataAsList(String filePath) throws IOException, NumberFormatException {
            List<double[]> lines = new ArrayList<>();
            int numSamples = 0;
            int numDims = 0;
            int numClasses = 0;

            try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
                String headerLine = reader.readLine();
                if (headerLine == null) throw new IOException("File is empty.");
                
                String[] headerParts = headerLine.trim().split("[\\s,]+");
                if (headerParts.length >= 3) {
                    numSamples = Integer.parseInt(headerParts[0]);
                    numDims = Integer.parseInt(headerParts[1]);
                    numClasses = Integer.parseInt(headerParts[2]);
                }

                String line;
                while ((line = reader.readLine()) != null) {
                    if (line.trim().isEmpty() || line.trim().startsWith("#")) continue;
                    String[] parts = line.trim().split("[\\s,]+"); 
                    List<String> validParts = new ArrayList<>();
                    for(String part : parts) { if (!part.isEmpty()) validParts.add(part); }
                    if (validParts.isEmpty()) continue;
                    double[] values = new double[validParts.size()];
                    for (int i = 0; i < validParts.size(); i++) values[i] = Double.parseDouble(validParts.get(i));
                    lines.add(values);
                }
            }
            return new DataLoadResult(numSamples, numDims, numClasses, lines);
        }

        public static List<Sample> convertRawDataToSamples(List<double[]> rawLines) throws IllegalArgumentException {
            List<Sample> samples = new ArrayList<>();
            if (rawLines.isEmpty()) return samples;
            
            int totalDims = rawLines.get(0).length;
            int numDims = totalDims - 1; 
            
            if (numDims < 1) throw new IllegalArgumentException("Data has too few values to separate features and label.");
            
            for (int i = 0; i < rawLines.size(); i++) {
                double[] line = rawLines.get(i);
                if (line.length != totalDims) { 
                    System.err.println("Warning: Skipping raw line at index " + i + " with incorrect dimensions.");
                    continue;
                }
                
                // 最後の要素がラベル
                int label = (int) line[line.length - 1]; 
                double[] features = new double[numDims];
                System.arraycopy(line, 0, features, 0, numDims);
                
                samples.add(new Sample(features, label));
            }
            return samples;
        }

        public static List<Sample> shuffleData(List<Sample> samples, long seed) {
            Collections.shuffle(samples, new Random(seed));
            return samples;
        }
    }
    
    // --- 評価のための補助関数（変更なし） ---

    private static List<double[]> convertSamplesToFeatures(List<Sample> samples) {
        List<double[]> featuresList = new ArrayList<>();
        for (Sample sample : samples) {
            featuresList.add(sample.features);
        }
        return featuresList;
    }

    private static int[] convertSamplesToLabels(List<Sample> samples) {
        return samples.stream().mapToInt(s -> s.label).toArray();
    }


    /**
     * JavaのHCA+アルゴリズムのメインエントリポイント
     */
    public static void main(String[] args) {
        // HCAplusEvaluator, HCAplusManager, HCAplusDataExporter は便宜上 MainHCAplusと同じパッケージにあるものと仮定
        // 実際のコードでは適切なインポートが必要です
        HCAplusEvaluator evaluator = new HCAplusEvaluator(); // 仮のインスタンス
        HCAplusManager manager = new HCAplusManager();       // 仮のインスタンス
        HCAplusDataExporter exporter = new HCAplusDataExporter("output_data"); // 仮のインスタンス

        // --- 実験パラメータ ---
        final int MaxLevel = 3;      
        final long SHUFFLE_SEED = 1;  
        final double MIN_CIM = 0.30;  
        final int LAMBDA = 50;        

        String dataset = "vehicle"; 
        String outputBaseDir = String.format("output_data/%s", dataset);
        
        // HCA+ トレーナーの初期化
        ARTNetTrainer caPlusTrainer = new ARTNetTrainer(); 
        HCAplusTrainer hcaPlusTrainer = new HCAplusTrainer(caPlusTrainer); 

        System.out.println("--- Starting HCAplus Training ---");
        
        // --- ループ (n=0..2, m=0..9) ---
        for (int n = 0; n < 3; n++) {
            for (int m = 0; m < 10; m++) {
                String baseFileName = String.format("a%d_%d_%s-10tra.dat", n, m, dataset);
                String filePath = String.format("dataset/%s/", dataset) + baseFileName;
                String filePrefix = String.format("a%d_%d_%s", n, m, dataset);

                System.out.printf("\nProcessing file: %s%n", filePath);

                DataLoadResult dataResult;
                try {
                    dataResult = DataHelper.loadRawDataAsList(filePath);
                } catch (IOException | NumberFormatException e) {
                    System.err.printf("Error loading raw data from %s: %s%n", filePath, e.getMessage());
                    continue;
                }
                // ファイルの中身を見る
                System.out.printf("  Loaded %d samples with %d dimensions and %d classes.%n",
								  dataResult.numSamples, dataResult.numDims, dataResult.numClasses);
                
                // データセット全体（特徴量とラベル）のリストを作成
                List<Sample> allData;
                try {
                    allData = DataHelper.convertRawDataToSamples(dataResult.dataLines);
                } catch (IllegalArgumentException e) {
                    System.err.printf("Error converting raw data from %s to Samples: %s%n", filePath, e.getMessage());
                    continue;
                }
                //ここで，クラスごとにデータを分割する，そして，for分で，クラスごとに！クラスタリングを実行する
                for (int classLabel = 0; classLabel < dataResult.numClasses; classLabel++) {
					List<Sample> classData = new ArrayList<>();
					for (Sample sample : allData) {
						if (sample.label == classLabel) {
							classData.add(sample);
						}
					}
					if (classData.isEmpty()) {
						System.out.printf("  No samples found for class %d in file %s. Skipping this class.%n", classLabel, baseFileName);
						continue;
					}
					// データをシャッフル
					List<Sample> shuffledClassData = DataHelper.shuffleData(classData, SHUFFLE_SEED);
					// --- HCAplusNetの初期化 ---
					HCAplusNet net = new HCAplusNet(
						LAMBDA, 
						MIN_CIM,
						MaxLevel
					);
					// --- 訓練 (trainRecursiveの引数を List<Sample> に修正) ---
					long startTime = System.nanoTime();
					HCAplusNet trainedNet = hcaPlusTrainer.trainRecursive(
						shuffledClassData, // List<Sample> をそのまま渡す
						net
					);
					long endTime = System.nanoTime();
					double timeTrain = (endTime - startTime) / 1_000_000.0;
					System.out.printf("  HCA+ Training for class %d in %s complete. Time: %.3f ms%n", classLabel, baseFileName, timeTrain);
					// --- 葉ノードの抽出と評価 ---
	                if (trainedNet != null && trainedNet.numNodes > 0) {
	                    
	                    // 葉ノードの抽出
	                    int[] maxLevelRef = {1}; 
	                    HCAplusNet leavesNet = manager.getLeavesNet(trainedNet, maxLevelRef);
	                    System.out.printf("  Extracted leaves net with %d nodes at max level %d.%n", 
										  leavesNet.numNodes, maxLevelRef[0]);
	                    // 評価
	                    /*double[] evaluationResults = evaluator.evaluate(
	                        convertSamplesToFeatures(shuffledClassData), 
	                        convertSamplesToLabels(shuffledClassData), 
	                        leavesNet
	                    );
	                    
	                    double ari = evaluationResults[0];
	                    double ami = evaluationResults[1];
	                    int numNodes = manager.countAllNodes(trainedNet);
	                    int numLeafNodes = leavesNet.numNodes;
	                    
	                    System.out.printf("    ARI: %.4f, AMI: %.4f, Total Nodes: %d, Leaf Nodes: %d, Max Level: %d%n",
	                                      ari, ami, numNodes, numLeafNodes, maxLevelRef[0]);
*/
	                 // 💡 修正点: exportHCAplusTreeNodes() を、ツリー巡回ロジックに置き換える
	                    Queue<HCAplusNet> queue = new LinkedList<>();
	                    queue.add(trainedNet); // ルートノードをキューに追加
	                    
	                    while (!queue.isEmpty()) {
	                        HCAplusNet currentModel = queue.poll(); // 先頭要素を取り出す
	                        
	                        // 1. ノード情報をCSVに出力
	                        // exporterは exportNodesForHCAplusLevel(model, filePrefix) を持っていると仮定
	                        exporter.exportNodesForHCAplusLevel(currentModel, filePrefix); 
	                        
	                        // 2. 子ノードをキューに追加 (再帰的展開)
	                        if (currentModel.children != null) {
	                            for (HCAplusNet child : currentModel.children) {
	                                // nullでない子ノードのみ次のレベルとして追加
	                                if (child != null) {
	                                    queue.add(child);
	                                }
	                            }
	                        }
	                    }
	                    
	                } else {
	                    System.out.printf("  Training failed or resulted in 0 nodes for %s. Skipping output/evaluation.%n", baseFileName);
	                }
                
                }
                
                
            }
        }
        System.out.println("\n--- All HCAplus Training and Evaluation complete ---");
    }
}