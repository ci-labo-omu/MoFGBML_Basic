package cilabo.fuzzy.rule.antecedent.factory.impl;

import cilabo.data.DataSet;
import cilabo.data.pattern.Pattern;
import cilabo.fuzzy.knowledge.Knowledge;
import cilabo.fuzzy.rule.antecedent.factory.AntecedentIndexFactory;
import cilabo.main.Consts;
import random.MersenneTwisterFast;

/**ランダムに決定されるヒューリスティックに基づいた前件部のFactory
 * @author Takigawa Hiroki
 */
public final class RandomInitialization implements AntecedentIndexFactory{
	/**	乱数生成器 */
	MersenneTwisterFast uniqueRnd;
	/** 学習用データ */
	private DataSet<?> train;
	/** 次元数 */
	private int dimension;
	/** シード値 */
	private int seed;

	/** 入力された情報を基にインスタンスを生成
	 * @param seed シード値
	 * @param train データセット
	 */
	public RandomInitialization(int seed, DataSet<?> train) {
		this.seed = seed;
		this.uniqueRnd = new MersenneTwisterFast(seed);
		this.train = train;
		this.dimension = Knowledge.getInstance().getNumberOfDimension();
	}

	@Override
	public int[] create() {
		double dcRate;
		System.out.println("Creating antecedent by Random Initialization in RandomInitialization...");
		System.out.println(Consts.DONT_CARE_RT);
		if(Consts.IS_PROBABILITY_DONT_CARE) {
			System.out.println("Probability Value");
			// Constant Value
			dcRate = Consts.DONT_CARE_RT;
		}
		else {
			// (dimension - Const) / dimension
			dcRate = (dimension - Consts.ANTECEDENT_NUMBER_DO_NOT_DONT_CARE)/(double)dimension;
		}

		int[] antecedentIndex = new int[dimension];

		Pattern<?> randomPattern = train.getPattern(uniqueRnd.nextInt(train.getDataSize()));

		for(int n = 0; n < dimension; n++) {
			if(uniqueRnd.nextBoolean(dcRate)) {
				// don't care
				antecedentIndex[n] = 0;
			}
			else {
				// Judge which dimension n is categorical or numerical.
				if(randomPattern.getAttributeValue(n) < 0) {
					// Categorical
					antecedentIndex[n] = (int)randomPattern.getAttributeValue(n);
				}
				else {
					// Numerical
					antecedentIndex[n] = uniqueRnd.nextInt(Knowledge.getInstance().getFuzzySetNum(n));
				}
			}
		}

		return antecedentIndex;
	}

	@Override
	public int[][] create(int numberOfGenerateRule) {
		System.out.println("Creating antecedents by Random Initialization in RandomInitialization...");
		int[][] antecedentIndexArray = new int[numberOfGenerateRule][this.dimension];
		for(int i=0; i<numberOfGenerateRule; i++) {
			antecedentIndexArray[i] = this.create();
		}
		return antecedentIndexArray;
	}

	@Override
	public String toString() {
		return "RandomInitialization [uniqueRnd=" + uniqueRnd + "]";
	}

	@Override
	public RandomInitialization copy() {
		return new RandomInitialization(this.seed, this.train);
	}

}
