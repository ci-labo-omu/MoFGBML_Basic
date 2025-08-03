package cilabo;

import java.util.List;
import java.util.Objects;

import org.apache.commons.lang3.tuple.Pair;

import cilabo.data.DataSet;
import cilabo.data.DataSetManager;
import cilabo.data.Input; // Input_Basic (従来のInput) を使用
import cilabo.data.InputDensity; // ★ InputDensity をインポート
import cilabo.data.pattern.impl.Pattern_Basic;
import cilabo.data.pattern.impl.PatternDensity; // ★ PatternDensity をインポート
import cilabo.fuzzy.classifier.Classifier;
import cilabo.fuzzy.classifier.classification.Classification;
import cilabo.fuzzy.classifier.classification.impl.SingleWinnerRuleSelection;
import cilabo.fuzzy.classifier.impl.Classifier_basic;
import cilabo.fuzzy.knowledge.Knowledge;
import cilabo.fuzzy.knowledge.factory.HomoTriangleKnowledgeFactory;
import cilabo.fuzzy.knowledge.membershipParams.Parameters;
import cilabo.fuzzy.rule.Rule.RuleBuilder;
import cilabo.fuzzy.rule.antecedent.factory.impl.HeuristicRuleGenerationMethod;
import cilabo.fuzzy.rule.consequent.factory.impl.MoFGBML_Learning;
import cilabo.fuzzy.rule.impl.Rule_Basic;
import cilabo.gbml.problem.pittsburghFGBML_Problem.impl.PittsburghFGBML_Basic;
import cilabo.gbml.solution.michiganSolution.AbstractMichiganSolution;
import cilabo.gbml.solution.michiganSolution.MichiganSolution.MichiganSolutionBuilder;
import cilabo.gbml.solution.michiganSolution.impl.MichiganSolution_Basic;
import cilabo.gbml.solution.pittsburghSolution.impl.PittsburghSolution_Basic;


// Density用に変更
// MakeTestObject クラスも PatternDensity を扱うように改修します。
public class MakeTestObject {
	// train フィールドの型を PatternDensity に変更
	DataSet<PatternDensity> train = null; // ★変更点: 型を PatternDensity に

	Parameters parameters = null;
	Knowledge knowledge = null;
	RuleBuilder<Rule_Basic, ?, ?> ruleBuilder = null;
	MichiganSolutionBuilder<MichiganSolution_Basic<Rule_Basic>> michiganSolutionBuilder = null;
	Classification<MichiganSolution_Basic<Rule_Basic>> classification = null;
	Classifier<MichiganSolution_Basic<Rule_Basic>> classifier = null;
	PittsburghFGBML_Basic<MichiganSolution_Basic<Rule_Basic>> problem = null;

	String dataSetName;
	int x;
	int y;

	// コンストラクタ
	public MakeTestObject(String dataSetName, int x, int y) {
		super();
		this.dataSetName = dataSetName;
		this.x = x;
		this.y = y;
		// getTrain() の呼び出しにより train フィールドが初期化される
		this.getTrain();
		this.getKnowledge();
	}

	// getTrain() メソッドの修正
	// 戻り値の型を DataSet<PatternDensity> に変更し、InputDensity を使用してロード
	public DataSet<PatternDensity> getTrain() { // ★変更点: 戻り値の型を PatternDensity に
		if(Objects.isNull(train)) {
			System.out.println("MakeTestObject.getTrain() called.");

			// ファイルパスの定義
			// 訓練データは密度情報を持つファイル (例: _node30.csv)
			// テストデータは従来のファイル (例: -10tst.dat)
			// 注意: "node30" の部分は、必要に応じてコンストラクタ引数などから動的に設定できるようにすると柔軟性が上がります。
			String trainFileName = String.format("dataset_nodes/%s/a%d_%d_%s_tra/a%d_%d_%s_node30.csv", this.dataSetName, this.x, this.y, this.dataSetName, this.x, this.y, this.dataSetName);
			String testFileName = String.format("dataset/%s/a%d_%d_%s-10tst.dat", this.dataSetName, this.x, this.y, this.dataSetName);

			/* Load Dataset ======================== */
			// DataSetManager は DataSet<? extends Pattern<?>> を受け入れるように修正済みである前提
			// 訓練データセットは InputDensity クラスを使用してロード
			DataSetManager.getInstance().addTrains(InputDensity.inputDataSet_WithDensity(trainFileName)); // ★変更点: InputDensity を使用

			// テストデータセットは Input (従来のInput) クラスを使用してロード
			DataSetManager.getInstance().addTests(Input.inputDataSet_Basic(testFileName)); // ★変更点: Input.inputDataSet_Basic を使用

			// DataSetManager から訓練データを取得する際も PatternDensity 型として取得
			// DataSetManager.getTrains() が List<DataSet<? extends Pattern<?>>> を返す前提でキャスト
			train = (DataSet<PatternDensity>) DataSetManager.getInstance().getTrains().get(0); // ★変更点: キャスト先を PatternDensity に
		}
		return train;
	}

	// setTrain() メソッドの修正
	public void setTrain(DataSet<PatternDensity> train) { // ★変更点: 引数の型を PatternDensity に
		this.train = train;
	}

	// getParameters() メソッド (getTrain() の戻り値の型変更により自動的に適合)
	public Parameters getParameters() {
		if(Objects.isNull(parameters)) {
			parameters = new Parameters(this.getTrain()); // getTrain() は DataSet<PatternDensity> を返す
			for(int dim_i=0; dim_i<train.getNdim(); dim_i++) {
				parameters.makeHomePartition(dim_i, new int[] {2, 3, 4, 5});
			}
		}
		return parameters;
	}

	public void setParameters(Parameters parameters) {
		this.parameters = parameters;
	}

	// getKnowledge() メソッド (getTrain() に依存しない、あるいは間接的に適合)
	public Knowledge getKnowledge() {
		if(Objects.isNull(knowledge)) {
		HomoTriangleKnowledgeFactory KnowledgeFactory = new HomoTriangleKnowledgeFactory(this.getParameters());
		KnowledgeFactory.create2_3_4_5();
			}
		return knowledge;
		}

	public void setKnowledge(Knowledge knowledge) {
		this.knowledge = knowledge;
	}

	// getRuleBuilder() メソッド
	public RuleBuilder<Rule_Basic, ?, ?> getRuleBuilder() {
		if(Objects.isNull(ruleBuilder)) {
			// MoFGBML_Learning のコンストラクタは DataSet<PatternDensity> を期待しているので、
			// this.getTrain() が DataSet<PatternDensity> を返すように修正されたため、ここは変更不要。
			ruleBuilder = new Rule_Basic.RuleBuilder_Basic(
				new HeuristicRuleGenerationMethod(this.getTrain()),
				new MoFGBML_Learning(this.getTrain()));
		}
		return ruleBuilder;
	}

	public void setRuleBuilder(RuleBuilder<Rule_Basic, ?, ?> ruleBuilder) {
		this.ruleBuilder = ruleBuilder;
	}

	// getMichiganSolutionBuilder() メソッド
	public MichiganSolutionBuilder<MichiganSolution_Basic<Rule_Basic>> getMichiganSolutionBuilder() {
		if(Objects.isNull(michiganSolutionBuilder)) {
			List<Pair<Integer, Integer>> bounds_Michigan = AbstractMichiganSolution.makeBounds();
			michiganSolutionBuilder = new MichiganSolution_Basic.MichiganSolutionBuilder_Basic<Rule_Basic>(
					bounds_Michigan, 2, 0, this.getRuleBuilder());
		}
		return michiganSolutionBuilder;
	}

	public void setMichiganSolutionBuilder(MichiganSolutionBuilder<MichiganSolution_Basic<Rule_Basic>> michiganSolutionBuilder) {
		this.michiganSolutionBuilder = michiganSolutionBuilder;
	}

	// getClassification() メソッド
	public Classification<MichiganSolution_Basic<Rule_Basic>> getClassification() {
		if(Objects.isNull(classification)) {
			classification = new SingleWinnerRuleSelection<MichiganSolution_Basic<Rule_Basic>>();
		}
		return classification;
	}

	public void setClassification(Classification<MichiganSolution_Basic<Rule_Basic>> classification) {
		this.classification = classification;
	}

	// getClassifier() メソッド
	public Classifier<MichiganSolution_Basic<Rule_Basic>> getClassifier() {
		if(Objects.isNull(classifier)) {
			classifier = new Classifier_basic<>(this.getClassification());
		}
		return classifier;
	}

	public void setClassifier(Classifier<MichiganSolution_Basic<Rule_Basic>> classifier) {
		this.classifier = classifier;
	}

	// getProblem() メソッド
	public PittsburghFGBML_Basic<MichiganSolution_Basic<Rule_Basic>> getProblem() {
		if(Objects.isNull(problem)) {
			// getTrain() が DataSet<PatternDensity> を返すように修正されたため、ここは変更不要
			problem = new PittsburghFGBML_Basic<MichiganSolution_Basic<Rule_Basic>>(
					60, 2, 0, this.getTrain(), this.getMichiganSolutionBuilder(), this.getClassifier());
		}
		return problem;
	}

	public void setProblem(PittsburghFGBML_Basic<MichiganSolution_Basic<Rule_Basic>> problem) {
		this.problem = problem;
	}

	// makePittsburghSolution() メソッド
	public PittsburghSolution_Basic<MichiganSolution_Basic<Rule_Basic>> makePittsburghSolution() {
		return this.getProblem().createSolution();
	}

	// makeMichiganSolution() メソッド
	public MichiganSolution_Basic<Rule_Basic> makeMichiganSolution(){
		if(Objects.isNull(michiganSolutionBuilder)) {
			this.getMichiganSolutionBuilder();
		}
		return this.michiganSolutionBuilder.createMichiganSolution();
	}

	// makeMichiganSolutionArray() メソッド
	public List<MichiganSolution_Basic<Rule_Basic>> makeMichiganSolutionArray(int numberOfGenerateRule){
		if(Objects.isNull(michiganSolutionBuilder)) {
			this.getMichiganSolutionBuilder();
		}
		return this.michiganSolutionBuilder.createMichiganSolutions(numberOfGenerateRule);
	}

}