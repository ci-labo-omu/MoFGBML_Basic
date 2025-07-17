package cilabo.fuzzy.classifier.classification.impl;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import cilabo.MakeTestObject;
import cilabo.data.pattern.Pattern;
import cilabo.fuzzy.classifier.classification.Classification;
import cilabo.fuzzy.rule.impl.Rule_Basic;
import cilabo.gbml.solution.michiganSolution.MichiganSolution.MichiganSolutionBuilder;
import cilabo.gbml.solution.michiganSolution.impl.MichiganSolution_Basic;

class SingleWinnerRuleSelectionTest {

	static MakeTestObject testObject;
	@BeforeAll
	static void setUpBeforeClass() throws Exception {
		testObject = new MakeTestObject("vehicle", 0, 0);
	}

	@Test
	void test() {

		List<MichiganSolution_Basic<Rule_Basic>> michiganSolutionList =
			new ArrayList<MichiganSolution_Basic<Rule_Basic>>();

		for(MichiganSolution_Basic<Rule_Basic> solution_i: testObject.makeMichiganSolutionArray(20)) {
			michiganSolutionList.add(solution_i);
		}
		Classification<MichiganSolution_Basic<Rule_Basic>> classification = testObject.getClassification();
		Pattern<?> pattern = testObject.getTrain().getPattern(0);
		assertTrue(classification.classify(michiganSolutionList, pattern) instanceof MichiganSolution_Basic);
	}

	@Test
	void test2() {
		List<MichiganSolution_Basic<Rule_Basic>> michiganSolutionList =
				new ArrayList<MichiganSolution_Basic<Rule_Basic>>();
		MichiganSolutionBuilder<MichiganSolution_Basic<Rule_Basic>> michiganSolutionBuilder = testObject.getMichiganSolutionBuilder();
		michiganSolutionList.addAll(michiganSolutionBuilder.createMichiganSolutions(5));
		Classification<MichiganSolution_Basic<Rule_Basic>> classification = testObject.getClassification();
		Pattern<?> pattern = testObject.getTrain().getPattern(0);
		assertTrue(classification.classify(michiganSolutionList, pattern) instanceof MichiganSolution_Basic);
	}
}
