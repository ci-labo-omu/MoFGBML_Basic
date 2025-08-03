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

    // Helper class for data loading and normalization, similar to Python's numpy/sklearn
	private static class DataHelper {
        public static List<double[]> loadRawDataAsList(String filePath) throws IOException, NumberFormatException {
            List<double[]> lines = new ArrayList<>();
            System.out.println("Attempting to read raw data from file: " + filePath);

            try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
                String line;
                int lineNumber = 0;
                String headerLine = reader.readLine(); 
                if (headerLine != null) {
                    System.out.println("Header line skipped: " + headerLine.trim());
                    lineNumber++;
                }

                while ((line = reader.readLine()) != null) {
                    lineNumber++;
                    if (line.trim().isEmpty() || line.trim().startsWith("#")) {
                        continue;
                    }
                    String[] parts = line.trim().split("\\s+|,"); 
                    List<String> validParts = new ArrayList<>();
                    for(String part : parts) {
                        if (!part.isEmpty()) {
                            validParts.add(part);
                        }
                    }
                    if (validParts.isEmpty()) {
                        System.out.println("Warning: Line " + lineNumber + " contains no valid data, skipping.");
                        continue;
                    }
                    double[] values = new double[validParts.size()];
                    for (int i = 0; i < validParts.size(); i++) {
                        values[i] = Double.parseDouble(validParts.get(i));
                    }
                    lines.add(values);
                }
            }
            System.out.printf("Finished reading file. %d data lines loaded into raw list.%n", lines.size());
            return lines;
        }

        public static List<Sample> convertRawDataToSamples(List<double[]> rawLines) throws IllegalArgumentException {
            List<Sample> samples = new ArrayList<>();
            if (rawLines.isEmpty()) {
                return samples;
            }
            int numDims = rawLines.get(0).length - 1; 
            if (numDims < 1) { 
                throw new IllegalArgumentException("First data line has too few values to separate features and label.");
            }
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
            System.out.printf("Converted %d raw lines to Sample objects.%n", samples.size());
            return samples;
        }

        public static List<Sample> shuffleData(List<Sample> samples, long seed) {
            Collections.shuffle(samples, new Random(seed));
            return samples;
        }
    }

    public static void main(String[] args) {
        final int TRIAL = 1;

        // Load data from file
        List<double[]> rawLines = new ArrayList<>(); 
        try {
            rawLines = DataHelper.loadRawDataAsList("dataset/vehicle/all_data.dat");
            System.out.println("\n--- Content of rawLines (all data lines from file) ---");
            for (int i = 0; i < rawLines.size(); i++) {
                System.out.println("Line " + (i+1) + ": " + Arrays.toString(rawLines.get(i)));
            }
            System.out.println("--- End of rawLines content ---\n");

        } catch (IOException | NumberFormatException e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
            return;
        }
        
        List<Sample> allData = new ArrayList<>();
        try {
            allData = DataHelper.convertRawDataToSamples(rawLines);
        } catch (IllegalArgumentException e) {
            System.err.println(e.getMessage());
            e.printStackTrace();
            return;
        }


        System.out.printf("Loaded %d samples from all_data.dat%n", allData.size());
        
        Map<Integer, Long> classCounts = allData.stream()
            .collect(Collectors.groupingBy(s -> s.label, Collectors.counting()));
        System.out.println("Samples per class: " + classCounts);
        
        MinMaxScaler scaler = new MinMaxScaler();
        List<Sample> normalizedData = scaler.fitTransform(allData);

        Map<Integer, List<Sample>> dataByClass = normalizedData.stream()
            .collect(Collectors.groupingBy(s -> s.label));

        double[] minCIMs = {0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75};
        final int LAMBDA = 50;

        ARTNetTrainer trainer = new ARTNetTrainer(); // コンストラクタ引数なしに変更
        ARTNetDataExporter exporter = new ARTNetDataExporter(); // コンストラクタ引数なしに変更

        // 訓練結果を格納するためのMapのMap (minCIM -> classLabel -> ARTNetModel)
        Map<Double, Map<Integer, ARTNetModel>> allTrainedModels = new HashMap<>();

        for (double minCIM : minCIMs) {
            Map<Integer, ARTNetModel> modelsForThisMinCIM = new HashMap<>(); // このminCIMにおける全クラスのモデル

            for (Map.Entry<Integer, List<Sample>> entry : dataByClass.entrySet()) {
                int classLabel = entry.getKey();
                List<Sample> classData = entry.getValue();

                System.out.printf("\n--- Training Class %d with minCIM=%.2f ---\n", classLabel, minCIM);
                
                // Shuffle data (Python script does this inside the loop)
                List<Sample> shuffledData = DataHelper.shuffleData(classData, 11); // Seed 11
                //shuffledDataの形状を表示
                System.out.printf("  Shuffled data size for class %d: %d samples%n", classLabel, shuffledData.size());
                ARTNetModel net = new ARTNetModel(LAMBDA, minCIM);
                
                long timeTrain = 0;
                for (int trial = 0; trial < TRIAL; trial++) {
                    System.out.printf("  Trial %d/%d%n", trial + 1, TRIAL);
                    
                    long startTime = System.nanoTime();
                    trainer.artClusteringTrain(net, shuffledData);
                    timeTrain += System.nanoTime() - startTime;
                    
                    System.out.printf("   Num. Clusters: %d%n", net.numNodes);
                    System.out.printf("  Processing Time: %.3f ms%n", timeTrain / 1_000_000.0);
                }
                modelsForThisMinCIM.put(classLabel, net); // 訓練済みモデルを格納
            }
            allTrainedModels.put(minCIM, modelsForThisMinCIM); // このminCIMの全クラスモデルを格納
        }

        // 全てのminCIMとクラスの訓練が完了した後、ARTNetDataExporterを呼び出す
        exporter.exportCombinedNodeDataByMinCIM(allTrainedModels, "dataset_nodes/vehicle", "vehicle_nodes_minCIM");
        // 例: dataset_nodes/vehicle/vehicle_nodes_minCIM_10.csv, vehicle_nodes_minCIM_15.csv などが出力されます
    }
}