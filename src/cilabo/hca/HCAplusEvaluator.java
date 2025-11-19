package cilabo.hca;

import java.util.Arrays;
import java.util.List;

import cilabo.art.HCAplusNet;


// HCA+ の評価ロジック
public class HCAplusEvaluator {
    
    /**
     * HCA+の葉ノードを使用してテストデータを評価し、ARIとAMIを計算します (MATLAB: HCAplus_Evaluation_new.m)。
     * * @param testData テストデータ (N x D 行列)
     * @param testLabels テストデータの真のラベル
     * @param leavesNet 葉ノード情報を持つHCAplusNet
     * @return ARIとAMIを含む配列 {ARI, AMI}
     */
    public double[] evaluate(List<double[]> testData, int[] testLabels, HCAplusNet leavesNet) {
        
        List<double[]> weights = leavesNet.weights; // (k x d)
        if (weights == null || weights.isEmpty()) {
            // 葉ノードの重みが空の場合、すべてのクラスタ割り当てを 0 (未割り当て) として返す
            System.err.println("Evaluation Error: LeavesNet has no weights (0 nodes).");
            return new double[]{0.0, 0.0}; 
        }
        
        double adaptiveSig = util.meanDouble(leavesNet.adaptiveSigs); // CIM で使用する適応シグマ値
        
        int numTestData = testData.size();
        int numWeights = weights.size();
        System.out.println("Number of test data: " + numTestData);
        System.out.println("Number of clusters (leaves): " + numWeights);
        int[] assignedCluster = new int[numTestData]; // クラスタリング結果
        
        
        // 1. データ点に最も近い葉ノードを割り当てる (CIM距離を使用)
        for (int i = 0; i < numTestData; i++) {
            double[] data = testData.get(i);
            double minDistance = Double.MAX_VALUE;
            int closestCluster = 0;
            
            for (int j = 0; j < numWeights; j++) {
                double[] weight = weights.get(j);
                // CIM 距離の計算 (MATLAB: CIM(testData(i, :), weights(j, :), adaptiveSig))
                double distance = util.cim(data, Arrays.asList(weight), adaptiveSig)[0]; 
                if (distance < minDistance) {
                    minDistance = distance;
                    // MATLABのインデックスは1から始まるため、+1する
                    closestCluster = j + 1; 
                }
            }
            assignedCluster[i] = closestCluster;
        }

        // 2. 評価指標の計算
        double ari = adjustedRandIndex(testLabels, assignedCluster);
        double ami = adjustedMutualInformation(testLabels, assignedCluster);
        System.out.printf("Evaluation Results: ARI = %.4f, AMI = %.4f%n", ari, ami);
        return new double[]{ari, ami};
    }

    // --- 評価指標の補助関数 ---

    /**
     * Adjusted Rand Indexの計算 (MATLAB: AdjustedRandIndex)
     */
    private double adjustedRandIndex(int[] actual, int[] predicted) {
        int n = predicted.length;
        if (n <= 1) return 0.0; 

        int ku = Arrays.stream(predicted).max().orElse(0);
        int kv = Arrays.stream(actual).max().orElse(0);
     
     // 🛑 早期終了チェック: データがない、またはラベルがない場合は 0.0 を返す
        if (n == 0 || ku == 0 || kv == 0) {
        	System.err.println("ARI Calculation Error: No data or no labels.");
            return 0.0;
        }
        // predicted と actual の内容をデバッグ出力
        System.out.println("Predicted labels: ");
		System.out.println("Actual labels: ");
		System.out.println("DEBUG: Actual Labels (first 20): " + Arrays.toString(Arrays.copyOf(actual, actual.length)));
	    System.out.println("DEBUG: Predicted Labels (first 20): " + Arrays.toString(Arrays.copyOf(predicted, predicted.length)));
        // 1. 混同行列 m の作成
        int[][] m = new int[ku + 1][kv + 1]; // 1-based indexing
        for (int i = 0; i < n; i++) {
            m[predicted[i]][actual[i]]++;
        }
        
        // 2. 行和 mu と列和 mv の計算
        long[] mu = new long[ku + 1];
        long[] mv = new long[kv + 1];
        for (int i = 1; i <= ku; i++) {
            for (int j = 1; j <= kv; j++) {
                mu[i] += m[i][j];
                mv[j] += m[i][j];
            }
        }
        
        // 3. a, b1, b2, c の計算
        long a = 0;
        for (int i = 1; i <= ku; i++) {
            for (int j = 1; j <= kv; j++) {
                if (m[i][j] > 1) {
                    a += util.nchoosek(m[i][j], 2);
                }
            }
        }

        long b1 = 0;
        for (int i = 1; i <= ku; i++) {
            if (mu[i] > 1) {
                b1 += util.nchoosek(mu[i], 2);
            }
        }
        
        long b2 = 0;
        for (int i = 1; i <= kv; i++) {
            if (mv[i] > 1) {
                b2 += util.nchoosek(mv[i], 2);
            }
        }
        
        long c = util.nchoosek(n, 2);

        // 4. ARI の計算
        double expectedIndex = (double)b1 * b2 / c;
        double maxIndex = 0.5 * (b1 + b2);
        
        if (c == expectedIndex) { // Avoid division by zero, though unlikely
             return 0.0;
        }

        double ari = (a - expectedIndex) / (maxIndex - expectedIndex);
        
        return Math.max(0.0, ari); // MATLAB: if ARI<0, ARI = 0;
    }

    /**
     * Adjusted Mutual Informationの計算 (MATLAB: AdjustedMutualInformation)
     * *NOTE: この実装は非常に複雑なため、MATLABのロジックを完全にJavaの線形代数に変換する必要があります。*
     * *ここでは、MATLABのロジックの主要部分のみを示し、完全な再実装はutilに依存します。*
     */
private double adjustedMutualInformation(int[] trueMem, int[] mem) {
        
        int R = util.max(trueMem); 
        int C = util.max(mem);
        int N = trueMem.length;
        
        if (N == 0 || R == 0 || C == 0) return 0.0;

        // 1. 混同行列 T の構築
        int[][] T = util.contingency(trueMem, mem);
        
        // Tが空の場合（通常はR, C == 0で既にチェックされる）
        if (T.length == 0 || T[0].length == 0) return 0.0;
        
        // 2. 行和 a と列和 b の計算 (TはR+1 x C+1 サイズ)
        long[] a = new long[R + 1]; // 行和 (真のラベル)
        long[] b = new long[C + 1]; // 列和 (予測クラスタ)
        
        for (int i = 1; i <= R; i++) {
            for (int j = 1; j <= C; j++) {
                if (i < T.length && j < T[i].length) {
                    a[i] += T[i][j];
                }
            }
        }
        for (int j = 1; j <= C; j++) {
            for (int i = 1; i <= R; i++) {
                if (i < T.length && j < T[i].length) {
                    b[j] += T[i][j];
                }
            }
        }
        
        // 3. エントロピー Ha, Hb, MI (Unadjusted) の計算
        double Ha = 0;
        for(int i = 1; i <= R; i++) {
            if (a[i] > 0) {
                double pa = (double)a[i] / N;
                Ha -= pa * Math.log(pa);
            }
        }
        
        double Hb = 0;
        for(int j = 1; j <= C; j++) {
            if (b[j] > 0) {
                double pb = (double)b[j] / N;
                Hb -= pb * Math.log(pb);
            }
        }
        
        // MIの計算 (unadjusted)
        double MI = 0;
        for (int i = 1; i <= R; i++) {
            for (int j = 1; j <= C; j++) {
                if (i < T.length && j < T[i].length && T[i][j] > 0) {
                    double pij = (double)T[i][j] / N;
                    double pa = (double)a[i] / N;
                    double pb = (double)b[j] / N;
                    // T(i,j)*log(T(i,j)*n/(a(i)*b(j))) / n
                    MI += pij * Math.log(pij / (pa * pb));
                }
            }
        }
        
        // 4. 期待値補正 EMI (Expected Mutual Information) の計算 - 複雑な部分
        // MATLABのロジックに忠実な再現を試みる (二項係数と逐次確率計算)
        double EMI = 0;
        
        for (int i = 1; i <= R; i++) {
            for (int j = 1; j <= C; j++) {
                
                long ai = a[i];
                long bj = b[j];
                
                // 組み合わせの範囲 [max(1, a(i)+b(j)-N), min(a(i), b(j))]
                int nij_min = (int) Math.max(1, ai + bj - N);
                int nij_max = (int) Math.min(ai, bj);
                
                if (nij_min > nij_max) continue;
                
                // 最初の確率 p0 の計算 (nij = nij_min)
                // MATLABの prod(nom./dem)/N に相当する対数階乗による安定化が必要だが、
                // MATLABの逐次計算ロジックに合わせる
                
                

                // 最初の確率 p(nij) の対数 (ログ確率) を計算
                // log(p(nij)) = log( C(ai, nij) * C(N-ai, bj-nij) / C(N, bj) )
                
                // C(N, bj) は定数ではないため、ここでは AMI ロジックに従う (p0 の直接計算)
                
                // 忠実な再現のため、AMIのオリジナルの逐次計算ロジックに則る
                double sumPnij = 0.0;
                double EPLNP = 0.0;
                
                // 最初の nij = nij_min の確率 p(nij) を計算
                double logP_start = util.logGamma(ai + 1) + util.logGamma(N - ai + 1) 
                                  + util.logGamma(bj + 1) + util.logGamma(N - bj + 1)
                                  - util.logGamma(N + 1) - util.logGamma(nij_min + 1)
                                  - util.logGamma(ai - nij_min + 1) - util.logGamma(bj - nij_min + 1)
                                  - util.logGamma(N - ai - bj + nij_min + 1);

                double p_current = Math.exp(logP_start);
                
                for (int nij = nij_min; nij <= nij_max; nij++) {
                    
                    // 確率の更新 (p1 = p0 * (ai-nij)*(bj-nij) / (nij+1) / (N-ai-bj+nij+1))
                    if (nij > nij_min) {
                         p_current = p_current * (ai - (nij - 1)) * (bj - (nij - 1)) 
                                   / (nij) / (N - ai - bj + (nij - 1) + 1);
                    }
                    
                    // sumPnij = sumPnij + p(nij)
                    sumPnij += p_current;
                    
                    // EPLNP(i,j) = EPLNP(i,j) + nij * log(nij/N) * p(nij)
                    if (p_current > 0) {
                        EPLNP += (double)nij * Math.log((double)nij / N) * p_current;
                    }
                }
                
                // E3 = (AB/n^2).*log(AB/n^2);
                double E3 = ((double)ai * bj) / (N * N) * Math.log(((double)ai * bj) / (N * N));
                
                // EMI += sum(sum(EPLNP - E3))
                EMI += (EPLNP - E3);
            }
        }
        
        // 5. 最終計算
        
        // Ha=-(a/n)*log(a/n)'  (既に計算済み)
        // Hb=-(b/n)*log(b/n)'  (既に計算済み)
        
        double max_H = Math.max(Ha, Hb);
        
        if (max_H == EMI) {
            return 0.0; // ゼロ除算の回避
        }

        double AMI = (MI - EMI) / (max_H - EMI);
        
        // AMI < 0 の場合は 0 に丸める (AMIは通常0以上)
        if (AMI < 0) AMI = 0;

        // MATLABのロジック: EMIが小さすぎる場合のNMIへのフォールバックは省略する（utilにAMIロジックを追加しないため）
        
        return AMI;
    }
}