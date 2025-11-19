package cilabo.hca;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class util {

    // --- CIM 関連 ---
    public static double[] cim(double[] x, List<double[]> y, double sig) {
        // ... (MATLABのCIM(X,Y,sig)を移植)
        // HCAplus_Evaluation_new.m の CIM と異なり、TrainCAplus_Classification.m の CIM を使用
        // MATLAB TrainCAplus: cim = sqrt(1 - mean(exp(-(X-Y).^2/(2*sig^2)), 2))'
        
        int n = y.size();
        int att = x.length;
        double[] cim = new double[n];

        for (int i = 0; i < n; i++) {
            double[] y_i = y.get(i);
            double sumKernel = 0;
            for (int j = 0; j < att; j++) {
                double sub = x[j] - y_i[j];
                sumKernel += gaussKernel(sub, sig);
            }
            double meanKernel = sumKernel / att;
            cim[i] = Math.sqrt(1.0 - meanKernel);
        }
        return cim;
    }
    
    public static double gaussKernel(double sub, double sig) {
        return Math.exp(-Math.pow(sub, 2) / (2 * Math.pow(sig, 2)));
    }

    // --- 統計関連 ---
    
    public static double median(double[] array) {
        // ... (中央値の計算ロジック)
        return 0.0;
    }
    
    
    
    
    
   
    // --- 統計・配列操作 ---

    public static double meanDouble(List<Double> list) {
        if (list == null || list.isEmpty()) return 0.0;
        return list.stream().mapToDouble(d -> d).average().orElse(0.0);
    }
    
    public static int max(int[] array) {
        if (array == null || array.length == 0) return 0;
        return Arrays.stream(array).max().orElse(0);
    }

    // --- 組み合わせ論 ---

    /**
     * nCk (n choose k) を計算 (MATLAB: nchoosek(n, k))
     * @param n 総数
     * @param k 選択数
     * @return 組み合わせ数
     */
    public static long nchoosek(long n, int k) {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        if (k > n / 2) k = (int)n - k;
        
        long res = 1;
        for (int i = 1; i <= k; i++) {
            // オーバーフローを防ぐため、乗算と除算を組み合わせる
            res = res * (n - i + 1) / i;
        }
        return res;
    }
 // ログスケールで組み合わせ係数を計算するヘルパー関数
    double log_nCr_safe(long n, long r) {
        if (r < 0 || r > n) return Double.NEGATIVE_INFINITY;
        return util.logGamma(n + 1) - util.logGamma(r + 1) - util.logGamma(n - r + 1);
    }
    
    // --- AMI 補助関数 ---
    
    /**
     * 混同行列を作成 (MATLAB: Contingency(Mem1, Mem2))
     * @param mem1 クラスタリング結果1 (1-based label)
     * @param mem2 クラスタリング結果2 (1-based label)
     * @return 混同行列 T
     */
    public static int[][] contingency(int[] mem1, int[] mem2) {
        if (mem1.length != mem2.length || mem1.length == 0) return new int[0][0];

        int R = max(mem1); // Max label of mem1
        int C = max(mem2); // Max label of mem2
        
        if (R == 0 || C == 0) return new int[0][0];

        // 1-based indexingなので、サイズは R+1 x C+1
        int[][] T = new int[R + 1][C + 1];
        
        for (int i = 0; i < mem1.length; i++) {
            // mem1[i] や mem2[i] が 0の場合はスキップ (未割り当て)
            if (mem1[i] > 0 && mem2[i] > 0) {
                T[mem1[i]][mem2[i]]++;
            }
        }
        return T;

    }
    public static double logGamma(double z) {
        // MATLABのlgamma関数に相当する実装
        if (z < 0.5) {
            return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1.0 - z);
        }
        z -= 1.0;
        double x = 0.99999999999980993;
        double[] c = {
            676.5203681218851, -1259.1392167224028, 771.32342872914155, 
            -176.61502916214059, 12.507343278686905, -0.13857109526572012, 
            9.9843695780195716e-6, 1.5056327351493116e-7
        };
        for (int i = 0; i < c.length; i++) {
            x += c[i] / (z + i + 1);
        }
        double t = z + c.length - 0.5;
        return Math.log(2 * Math.PI) / 2.0 + (z + 0.5) * Math.log(t) - t + Math.log(x);
    }
}