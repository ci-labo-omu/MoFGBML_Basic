package cilabo.gbml.component.initialsolutionscreation;

import java.io.File;
import java.util.List;

import org.apache.commons.lang3.tuple.Pair;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.uma.jmetal.component.initialsolutioncreation.InitialSolutionsCreation;

import cilabo.MakeTestObject;
import cilabo.data.DataSet;
import cilabo.data.DataSetManager;
import cilabo.data.pattern.impl.PatternDensity;
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
import cilabo.main.Consts;
import xml.XML_reader;

class InitialSolutionsCreationFromXMLTest {

	private static XML_reader XML_reader;
	private static DataSet<PatternDensity> train;
	private static InitialSolutionsCreation<PittsburghSolution_Basic<MichiganSolution_Basic<Rule_Basic>>>
		initialSolutionsCreation;
	private static List<PittsburghSolution_Basic<MichiganSolution_Basic<Rule_Basic>>> population;

	@BeforeAll
	static void setUpBeforeClass() throws Exception {
		MakeTestObject MTO = new MakeTestObject("pima", 0, 0);
		train = MTO.getTrain();

		Parameters parameters = new Parameters(train);
		HomoTriangleKnowledgeFactory KnowledgeFactory = new HomoTriangleKnowledgeFactory(parameters);
		KnowledgeFactory.create2_3_4_5();

		RuleBuilder<Rule_Basic, ?, ?> ruleBuilder = new Rule_Basic.RuleBuilder_Basic(
				new HeuristicRuleGenerationMethod(train),
				new MoFGBML_Learning(train));

		List<Pair<Integer, Integer>> bounds_Michigan = AbstractMichiganSolution.makeBounds();
		MichiganSolutionBuilder<MichiganSolution_Basic<Rule_Basic>> michiganSolutionBuilder
			= new MichiganSolution_Basic.MichiganSolutionBuilder_Basic<Rule_Basic>(
					bounds_Michigan, 2, 0, ruleBuilder);

		Classification<MichiganSolution_Basic<Rule_Basic>> classification = new SingleWinnerRuleSelection<MichiganSolution_Basic<Rule_Basic>>();

		Classifier<MichiganSolution_Basic<Rule_Basic>> classifier = new Classifier_basic<>(classification);

		/* MOP: Multi-objective Optimization Problem */
		PittsburghFGBML_Basic<MichiganSolution_Basic<Rule_Basic>> problem =
				new PittsburghFGBML_Basic<MichiganSolution_Basic<Rule_Basic>>(
						60, 2, 0, null, michiganSolutionBuilder, classifier);

		try {
			String ln = File.separator;
			XML_reader = new XML_reader("results" + ln + "default" + ln + "pima" + ln + "trial00" + ln + Consts.XML_FILE_NAME + ".xml");
		} catch (Exception e) {
			e.printStackTrace();
		}
		initialSolutionsCreation = new InitialSolutionsCreationFromXML<PittsburghSolution_Basic<MichiganSolution_Basic<Rule_Basic>>>(
				problem, XML_reader.getPopulation(Consts.TERMINATE_EVALUATION));
	}

	@Test
	void test() {
		population = initialSolutionsCreation.create();
//		System.out.println(population.toString());
	}

	@AfterAll
	static void afterClass() throws Exception {
		DataSetManager.getInstance().clear();
		Knowledge.getInstance().clear();
	}
}
