package cilabo.hca;

import cilabo.ghng.Sample;
import cilabo.art.ARTNetDataExporter;
import cilabo.art.MinMaxScaler;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.Arrays;
import java.util.stream.IntStream;

public class MainHCAplus {

    // ARTNetのmainから抽出したDataHelperクラスをここに配置
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
        // GHNG パラメータ設定 (HCA+のパラメータに対応させる)
        final int Epochs = 1;
        final int MaxLevel = 10;
        final long SHUFFLE_SEED = 1; // main_HCAplus.m の rng(1) に対応
        final int MaxLabel = 4; // vehicle データセットのクラス数

        String dataset = "vehicle";
        String outputBaseDir = String.format("dataset_nodes/%s", dataset);
        
        HCAplusTrainer trainer = new HCAplusTrainer(MaxLevel, SHUFFLE_SEED);
        HCAplusEvaluation evaluator = new HCAplusEvaluation();
        ARTNetDataExporter exporter = new ARTNetDataExporter();

        System.out.println("--- Starting HCAplus Training for all 10-fold x 3 repetitions data ---");
        
        for (int n = 0; n < 3; n++) {
            for (int m = 0; m < 10; m++) {
                String baseFileName = String.format("a%d_%d_%s-10tra.dat", n, m, dataset);
                String filePath = String.format("dataset/%s/", dataset) + baseFileName;

                System.out.printf("\nProcessing file: %s%n", filePath);

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

                System.out.printf("  Loaded %d samples from %s%n", allData.size(), filePath);
                
                // データをシャッフル (Python main_HCAplus.mのrng(1)とrandpermに対応)
                List<Sample> shuffledData = DataHelper.shuffleData(allData, SHUFFLE_SEED);

                // HCAplusNetの初期化
                HCAplusNet net = new HCAplusNet(shuffledData.get(0).features.length, shuffledData.size(), MaxLabel, MaxLevel, Epochs);
                
                // 訓練
                long startTime = System.nanoTime();
                HCAplusNet trainedNet = trainer.trainHCAplus(
                    convertSamplesToFeatures(shuffledData), 
                    net, 
                    1, // レベル1から開始
                    convertSamplesToLabels(shuffledData), 
                    MaxLabel
                );
                long endTime = System.nanoTime();
                
                double timeTrain = (endTime - startTime) / 1_000_000.0;
                
                System.out.printf("  HCA+ Training for %s complete. Time: %.3f ms%n", baseFileName, timeTrain);
                
                // ここで評価や出力ロジックを追加
                if (trainedNet != null) {
                    // 出力ファイル名: a{n}_{m}_vehicle_nodes_HCAplus.csv
                    //String outputFileName = String.format("a%d_%d_%s_nodes_HCAplus.csv", n, m, dataset);
                    // exporter.exportNodes(trainedNet, outputBaseDir, outputFileName);
                } else {
                    System.out.printf("  Training failed for %s. Skipping output.%n", baseFileName);
                }
            }
        }
        System.out.println("\n--- All HCAplus Training and Export complete ---");
    }

    private static double[][] convertSamplesToFeatures(List<Sample> samples) {
        if (samples.isEmpty()) return new double[0][0];
        int dimension = samples.get(0).features.length;
        double[][] features = new double[dimension][samples.size()];
        for (int i = 0; i < samples.size(); i++) {
            for (int d = 0; d < dimension; d++) {
                features[d][i] = samples.get(i).features[d];
            }
        }
        return features;
    }

    private static int[] convertSamplesToLabels(List<Sample> samples) {
        return samples.stream().mapToInt(s -> s.label).toArray();
    }
}