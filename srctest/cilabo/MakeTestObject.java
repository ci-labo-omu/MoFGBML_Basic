package cilabo;

import java.util.List;
import java.util.Objects;

import org.apache.commons.lang3.tuple.Pair;

import cilabo.data.DataSet;
import cilabo.data.DataSetManager;
import cilabo.data.Input;
import cilabo.data.pattern.impl.Pattern_Basic;
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


//Density用に変更
public class MakeTestObject {
	DataSet<Pattern_Basic> train = null;
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

	public MakeTestObject(String dataSetName, int x, int y) {
		super();
		this.dataSetName = dataSetName;
		this.x = x;
		this.y = y;
		this.getTrain();
		this.getKnowledge();
	}

	public DataSet<Pattern_Basic> getTrain() {
		if(Objects.isNull(train)) {
			String trainFileName = String.format("dataset_nodes/%s/a%d_%d_%s_tra/a%d_%d_%s_node30.csv", this.dataSetName, this.x, this.y, this.dataSetName, this.x, this.y, this.dataSetName);
			String testFileName = String.format("dataset/%s/a%d_%d_%s-10tst.dat", this.dataSetName, this.x, this.y, this.dataSetName);
			/* Load Dataset ======================== */
			Input.loadTrainTestFiles_Basic(trainFileName, trainFileName);
			train = (DataSet<Pattern_Basic>) DataSetManager.getInstance().getTrains().get(0);
		}
		return train;
	}

	public void setTrain(DataSet<Pattern_Basic> train) {
		this.train = train;
	}

	public Parameters getParameters() {
		if(Objects.isNull(parameters)) {
			parameters = new Parameters(this.getTrain());
			for(int dim_i=0; dim_i<train.getNdim(); dim_i++) {
				parameters.makeHomePartition(dim_i, new int[] {2, 3, 4, 5});
			}
		}
		return parameters;
	}

	public void setParameters(Parameters parameters) {
		this.parameters = parameters;
	}

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

	public RuleBuilder<Rule_Basic, ?, ?> getRuleBuilder() {
		if(Objects.isNull(knowledge)) {
			ruleBuilder = new Rule_Basic.RuleBuilder_Basic(
				new HeuristicRuleGenerationMethod(this.getTrain()),
				new MoFGBML_Learning(this.getTrain()));
		}
		return ruleBuilder;
	}

	public void setRuleBuilder(RuleBuilder<Rule_Basic, ?, ?> ruleBuilder) {
		this.ruleBuilder = ruleBuilder;
	}

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

	public Classification<MichiganSolution_Basic<Rule_Basic>> getClassification() {
		if(Objects.isNull(classification)) {
			classification = new SingleWinnerRuleSelection<MichiganSolution_Basic<Rule_Basic>>();
		}
		return classification;
	}

	public void setClassification(Classification<MichiganSolution_Basic<Rule_Basic>> classification) {
		this.classification = classification;
	}

	public Classifier<MichiganSolution_Basic<Rule_Basic>> getClassifier() {
		if(Objects.isNull(classifier)) {
			classifier = new Classifier_basic<>(this.getClassification());
		}
		return classifier;
	}

	public void setClassifier(Classifier<MichiganSolution_Basic<Rule_Basic>> classifier) {
		this.classifier = classifier;
	}

	public PittsburghFGBML_Basic<MichiganSolution_Basic<Rule_Basic>> getProblem() {
		if(Objects.isNull(problem)) {
			problem = new PittsburghFGBML_Basic<MichiganSolution_Basic<Rule_Basic>>(
					60, 2, 0, this.getTrain(), this.getMichiganSolutionBuilder(), this.getClassifier());
		}
		return problem;
	}

	public void setProblem(PittsburghFGBML_Basic<MichiganSolution_Basic<Rule_Basic>> problem) {
		this.problem = problem;
	}

	public PittsburghSolution_Basic<MichiganSolution_Basic<Rule_Basic>> makePittsburghSolution() {
		return this.getProblem().createSolution();
	}

	public MichiganSolution_Basic<Rule_Basic> makeMichiganSolution(){
		if(Objects.isNull(michiganSolutionBuilder)) {
			this.getMichiganSolutionBuilder();
		}
		return this.michiganSolutionBuilder.createMichiganSolution();
	}

	public List<MichiganSolution_Basic<Rule_Basic>> makeMichiganSolutionArray(int numberOfGenerateRule){
		if(Objects.isNull(michiganSolutionBuilder)) {
			this.getMichiganSolutionBuilder();
		}
		return this.michiganSolutionBuilder.createMichiganSolutions(numberOfGenerateRule);
	}

}
