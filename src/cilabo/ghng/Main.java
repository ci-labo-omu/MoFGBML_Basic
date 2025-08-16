package cilabo.ghng;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;


public class Main {
    public static void main(String[] args) {
        // GHNG パラメータ設定 (省略せず、前回のコードからコピーしてください)
        int epochs = 20;
        int maxNeurons = 1000;
        double tau = 0.01;
        int lambda = 50;
        double epsilonB = 0.05;
        double epsilonN = 0.001;
        double alpha = 0.5;
        int aMax = 50;
        double D = 0.995;
        int initialLevel = 0;

        // GHNGTrainer と GHNGDataExporter のインスタンス化
        GHNGTrainer trainer = new GHNGTrainer(4, 42); // 最大レベル4, 乱数シード42
        GHNGDataExporter exporter = new GHNGDataExporter(trainer);

        List<Sample> allTrainingSamples = new ArrayList<>();
        String datasetName = "magic"; // データセット名を指定
        // ファイル読み込み部分を直接ここに記述
        String fileName = "dataset/%s/%s.dat".formatted(datasetName, datasetName); // 正しいファイルパスであることを確認
        System.out.println("Attempting to read file: " + fileName); // 読み込み開始メッセージ

        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line;
            int lineNumber = 0; // 行数をカウント（デバッグ用）

            // 最初の行（パラメータ行を想定）をスキップする場合はコメントを外す
            reader.readLine(); 
            lineNumber++;



            while ((line = reader.readLine()) != null) {
                lineNumber++;
                // 空行やコメント行 (#などで始まる行) をスキップ
                if (line.trim().isEmpty() || line.trim().startsWith("#")) {
                    continue;
                }

                // スペース、タブ、カンマなどで区切られていると仮定して分割
                String[] parts = line.trim().split("\\s+|,"); 
                
                // 空文字列を除去して有効な数値部分だけを抽出
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

                // 数値配列に変換
                double[] values = new double[validParts.size()];
                for (int i = 0; i < validParts.size(); i++) {
                    values[i] = Double.parseDouble(validParts.get(i));
                }

                // 最後の要素をラベルとして扱い、それ以外を特徴量とする
                if (values.length < 2) { // 少なくとも特徴量とラベルが必要
                    System.out.println("Warning: Line " + lineNumber + " has too few values to separate features and label, skipping.");
                    continue;
                }
                int label = (int) values[values.length - 1];
                double[] features = new double[values.length - 1];
                System.arraycopy(values, 0, features, 0, values.length - 1);
                
                allTrainingSamples.add(new Sample(features, label));
            }
            System.out.printf("Loaded %d samples from %s%n", allTrainingSamples.size(), fileName);

        } catch (IOException e) {
            System.err.println("Error reading file: " + fileName + " - " + e.getMessage());
            e.printStackTrace();
            return; // ファイル読み込み失敗時は処理を終了
        } catch (NumberFormatException e) {
            System.err.println("Error parsing number in file at line " + (allTrainingSamples.size() + 1) + ": " + fileName + " - " + e.getMessage());
            System.err.println("Please check the data format in the file.");
            e.printStackTrace();
            return; // 数値変換失敗時は処理を終了
        }


        // 読み込んだサンプルの中身を表示（デバッグ用：最初の数件と最後の数件）
        System.out.println("Training samples (first 5 and last 5 if available):");
        for (int i = 0; i < Math.min(5, allTrainingSamples.size()); i++) {
            System.out.println(allTrainingSamples.get(i));
        }
        if (allTrainingSamples.size() > 10) { // データが十分にある場合のみ"..."を表示
            System.out.println("...");
            for (int i = Math.max(0, allTrainingSamples.size() - 5); i < allTrainingSamples.size(); i++) {
                System.out.println(allTrainingSamples.get(i));
            }
        } else { // データが少ない場合は全て表示
             for (Sample sample : allTrainingSamples) {
                 System.out.println(sample);
             }
        }


        // ここからGHNG訓練の既存のロジック
        Map<Integer, List<Sample>> samplesByClass = allTrainingSamples.stream()
                .collect(Collectors.groupingBy(s -> s.label));

        Map<Integer, GHNGModel> trainedModelsByClass = new HashMap<>();

        System.out.println("\n--- Starting Class-wise GHNG Training ---");
        long totalStartTime = System.nanoTime();
        
        for (Map.Entry<Integer, List<Sample>> entry : samplesByClass.entrySet()) {
            int classLabel = entry.getKey();
            List<Sample> classSamples = entry.getValue();

            if (classSamples.isEmpty()) {
                System.out.printf("Skipping Class %d: No samples found.%n", classLabel);
                continue;
            }

            // 特徴量配列 (double[][]) に変換
            // クラスの最初のサンプルから次元数を取得 (SampleクラスのgetDimension()を使用)
            int dimension = classSamples.get(0).getDimension(); 
            double[][] classFeatures = new double[dimension][classSamples.size()];
            int[] classOriginalLabels = new int[classSamples.size()];

            for (int i = 0; i < classSamples.size(); i++) {
                for (int d = 0; d < dimension; d++) {
                    classFeatures[d][i] = classSamples.get(i).features[d];
                }
                classOriginalLabels[i] = classSamples.get(i).label;
            }

            System.out.printf("\n>>> Training GHNG for Class %d (%d samples) <<< %n", classLabel, classSamples.size());
            GHNGModel trainedModelForClass = trainer.trainGHNG(classFeatures, classOriginalLabels, epochs, maxNeurons,
                                                     tau, lambda, epsilonB, epsilonN,
                                                     alpha, aMax, D, initialLevel);
            if (trainedModelForClass != null) {
                trainedModelsByClass.put(classLabel, trainedModelForClass);
            } else {
                System.out.printf(">>> Class %d model was pruned or training failed. Skipping. <<< %n", classLabel);
            }
        }

        long totalEndTime = System.nanoTime();
        System.out.printf("\n--- All Class-wise GHNG Training Complete! Total time: %.3f ms ---%n", (totalEndTime - totalStartTime) / 1_000_000.0);

        // 結果をCSVに出力
        System.out.println("\n--- Exporting GHNG Neurons by Level to CSV ---");
        //exporterに渡すとき，データセット名を入れる
        
        exporter.exportNeuronsByLevel(trainedModelsByClass, String.format("ghng_%s", datasetName));
        // オプション: モデルをシリアライズして保存
        //exporter.saveGHNGModels(trainedModelsByClass, "ghng_trained_models_by_class.ser");
    }
}