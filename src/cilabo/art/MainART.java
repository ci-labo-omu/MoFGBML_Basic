package cilabo.art;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.Collections;
import java.util.HashMap;

import cilabo.data.Input;
import cilabo.ghng.Sample;

public class MainART {

    private static class DataHelper {
        public static List<double[]> loadRawDataAsList(String filePath) throws IOException, NumberFormatException {
            List<double[]> lines = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
                String line;
                reader.readLine(); // ヘッダ行を読み飛ばす
                while ((line = reader.readLine()) != null) {
                    if (line.trim().isEmpty() || line.trim().startsWith("#")) continue;
                    String[] parts = line.trim().split("\\s+|,"); 
                    List<String> validParts = new ArrayList<>();
                    for(String part : parts) { if (!part.isEmpty()) validParts.add(part); }
                    if (validParts.isEmpty()) continue;
                    double[] values = new double[validParts.size()];
                    for (int i = 0; i < validParts.size(); i++) values[i] = Double.parseDouble(validParts.get(i));
                    lines.add(values);
                }
            }
            return lines;
        }

        public static List<Sample> convertRawDataToSamples(List<double[]> rawLines) throws IllegalArgumentException {
            List<Sample> samples = new ArrayList<>();
            if (rawLines.isEmpty()) return samples;
            int numDims = rawLines.get(0).length - 1; 
            if (numDims < 1) throw new IllegalArgumentException("First data line has too few values to separate features and label.");
            for (int i = 0; i < rawLines.size(); i++) {
                double[] line = rawLines.get(i);
                if (line.length != numDims + 1) { 
                    System.err.println("Warning: Skipping raw line at index " + i + " with incorrect number of dimensions (expected " + (numDims + 1) + " but got " + line.length + ").");
                    continue;
                }
                int label = (int) line[line.length - 1];
                double[] features = new double[line.length - 1];
                System.arraycopy(line, 0, features, 0, line.length - 1);
                samples.add(new Sample(features, label));
            }
            return samples;
        }

        public static List<Sample> shuffleData(List<Sample> samples, long seed) {
            Collections.shuffle(samples, new Random(seed));
            return samples;
        }
    }

    public static void main(String[] args) {
        final int TRIAL = 1; 

        double[] minCIMs = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75};
        final int LAMBDA = 50;
        final long SHUFFLE_SEED = 11; 

        ARTNetTrainer trainer = new ARTNetTrainer();
        ARTNetDataExporter exporter = new ARTNetDataExporter();

        System.out.println("--- Starting ARTNet Training for all 10-fold x 3 repetitions data ---");

        
        
        String dataset = "banknote"; // データセット名 (例: vehicle)
        // 出力ルートディレクトリ
        String outputBaseDir = "dataset_nodes/%s".formatted(dataset); // 例: dataset_nodes/vehicle/minCIM_XX/aN_M_vehicle_nodes_ART.csv
        // minCIMs の外側ループを先に行う
        for (double minCIM : minCIMs) {
            System.out.printf("\n=== Processing minCIM = %.2f ===\n", minCIM);

            // このminCIMにおける全てのフォールドの結果を格納するマップ (今は直接ファイルに出すので不要だが、もし後でまとめたいなら使える)
            // Map<String, Map<Integer, ARTNetModel>> modelsPerFoldForThisMinCIM = new HashMap<>();

            // 3回の繰り返し (n)
            for (int n = 0; n < 3; n++) {
                // 10-fold (m)
                for (int m = 0; m < 10; m++) {
                    String baseFileName = String.format("a%d_%d_%s-10tra.dat", n, m, dataset);
                    String filePath = "dataset/%s/".formatted(dataset) + baseFileName;

                    System.out.printf("  Processing file: %s%n", filePath);

                    List<double[]> rawLines;
                    try {
                        rawLines = DataHelper.loadRawDataAsList(filePath);
                    } catch (IOException | NumberFormatException e) {
                        System.err.printf("Error loading raw data from %s: %s%n", filePath, e.getMessage());
                        e.printStackTrace();
                        continue;
                    }
                    
                    List<Sample> allData;
                    try {
                        allData = DataHelper.convertRawDataToSamples(rawLines);
                    } catch (IllegalArgumentException e) {
                        System.err.printf("Error converting raw data from %s to Samples: %s%n", filePath, e.getMessage());
                        e.printStackTrace();
                        continue;
                    }

                    // System.out.printf("  Loaded %d samples from %s%n", allData.size(), filePath); // デバッグ出力は控えめに
                    
                    MinMaxScaler scaler = new MinMaxScaler();
                    List<Sample> normalizedData = scaler.fitTransform(allData);

                    // クラスごとに分割 (ARTNetはクラスごとに訓練するため)
                    Map<Integer, List<Sample>> dataByClass = normalizedData.stream()
                        .collect(Collectors.groupingBy(s -> s.label));

                    // この特定のファイル(aN_M)内で、各クラスを訓練したモデルを格納
                    Map<Integer, HCAplusNet> modelsForThisFile = new HashMap<>(); 

                    for (Map.Entry<Integer, List<Sample>> entry : dataByClass.entrySet()) {
                        int classLabel = entry.getKey();
                        List<Sample> classData = entry.getValue();

                        // System.out.printf("--- Training Class %d with minCIM=%.2f for %s ---\n", classLabel, minCIM, baseFileName); // デバッグ出力は控えめに
                        
                        List<Sample> shuffledData = DataHelper.shuffleData(classData, SHUFFLE_SEED); 
                        
                        HCAplusNet net = new HCAplusNet(LAMBDA, minCIM, 1); // maxlevel=1 でART相当
                        long timeTrain = 0;
                        for (int trial = 0; trial < TRIAL; trial++) {
                            long startTime = System.nanoTime();
                            trainer.artClusteringTrain(net, shuffledData);
                            timeTrain += System.nanoTime() - startTime;
                        }
                        
                        System.out.printf("    minCIM=%.2f, Class %d: Num. Clusters: %d, Time: %.3f ms%n", 
                                          minCIM, classLabel, net.numNodes, timeTrain / 1_000_000.0);
                        
                        modelsForThisFile.put(classLabel, net); // 訓練済みモデルを格納
                    }

                    // この特定のファイル(aN_M)の訓練が全て完了した後、ノードデータをエクスポート
                    // 出力ファイル名: a{n}_{m}_vehicle_nodes_ART.csv (minCIMはフォルダ名になる)
                    String outputFileNameSuffix = String.format("a%d_%d_%s_nodes_ART.csv", n, m, dataset);
                    exporter.exportNodesForSingleFile(
                        modelsForThisFile, 
                        minCIM, // 現在のminCIMを渡す
                        outputBaseDir, // ルート出力ディレクトリを渡す
                        outputFileNameSuffix
                    );
                }
            }
        }
        System.out.println("\n--- All ARTNet Training and Export complete ---");
    }
}

