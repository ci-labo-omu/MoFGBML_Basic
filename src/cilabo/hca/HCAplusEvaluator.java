// src/com/yourpackage/HCAplusEvaluation.java
package cilabo.hca;

import java.util.Arrays;

public class HCAplusEvaluation {

    /**
     * CIM (Correntropy induced Metric)
     * @param X 1 x d vector
     * @param Y m x d matrix
     * @param sig sigma value
     * @return 1 x m vector of CIM values
     */
    public double[] cim(double[] x, double[][] y, double sig) {
        int m = y.length;
        int d = x.length;
        double[][] gKernel = new double[m][d];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < d; j++) {
                gKernel[i][j] = gaussKernel(x[j] - y[i][j], sig);
            }
        }
        
        double[] ret1 = new double[m];
        for (int i = 0; i < m; i++) {
            double sum = 0;
            for (int j = 0; j < d; j++) {
                sum += gKernel[i][j];
            }
            ret1[i] = sum / d;
        }

        double[] cim = new double[m];
        for (int i = 0; i < m; i++) {
            cim[i] = Math.sqrt(1.0 - ret1[i]);
        }
        return cim;
    }
    
    private double gaussKernel(double sub, double sig) {
        if (sig == 0 || Double.isNaN(sig) || Double.isInfinite(sig)) {
            return 0.0;
        }
        return Math.exp(-Math.pow(sub, 2) / (2 * Math.pow(sig, 2)));
    }
    
    // Other evaluation metrics like ARI and AMI are complex and would require a lot of code.
    // For now, we will focus on the core HCA+ logic.
}