package cilabo.gbml.algorithm;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import org.uma.jmetal.algorithm.impl.AbstractEvolutionaryAlgorithm;
import org.uma.jmetal.component.densityestimator.DensityEstimator;
import org.uma.jmetal.component.densityestimator.impl.CrowdingDistanceDensityEstimator;
import org.uma.jmetal.component.evaluation.Evaluation;
import org.uma.jmetal.component.evaluation.impl.SequentialEvaluation;
import org.uma.jmetal.component.initialsolutioncreation.InitialSolutionsCreation;
import org.uma.jmetal.component.initialsolutioncreation.impl.RandomSolutionsCreation;
import org.uma.jmetal.component.ranking.Ranking;
import org.uma.jmetal.component.ranking.impl.FastNonDominatedSortRanking;
import org.uma.jmetal.component.replacement.Replacement;
import org.uma.jmetal.component.replacement.impl.RankingAndDensityEstimatorReplacement;
import org.uma.jmetal.component.selection.MatingPoolSelection;
import org.uma.jmetal.component.selection.impl.NaryTournamentMatingPoolSelection;
import org.uma.jmetal.component.termination.Termination;
import org.uma.jmetal.component.variation.Variation;
import org.uma.jmetal.operator.crossover.CrossoverOperator;
import org.uma.jmetal.operator.mutation.MutationOperator;
import org.uma.jmetal.operator.selection.SelectionOperator;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.util.JMetalLogger;
import org.uma.jmetal.util.SolutionListUtils;
import org.uma.jmetal.util.comparator.MultiComparator;
import org.uma.jmetal.util.fileoutput.impl.DefaultFileOutputContext;
import org.uma.jmetal.util.observable.Observable;
import org.uma.jmetal.util.observable.ObservableEntity;
import org.uma.jmetal.util.observable.impl.DefaultObservable;
import org.w3c.dom.Element;

import cilabo.fuzzy.knowledge.Knowledge;
import cilabo.gbml.component.variation.CrossoverAndMutationAndPittsburghLearningVariation;
import cilabo.gbml.problem.pittsburghFGBML_Problem.AbstractPittsburghFGBML;
import cilabo.gbml.solution.pittsburghSolution.PittsburghSolution;
import cilabo.util.fileoutput.PittsburghSolutionListOutput;
import xml.XML_TagName;
import xml.XML_manager;

public class HybridMoFGBMLwithNSGAII <S extends PittsburghSolution<?>>
	extends AbstractEvolutionaryAlgorithm<S, List<S>>
	implements ObservableEntity {

	private int evaluations;
	private int populationSize;
	private int offspringPopulationSize;
	private int frequency;
	private String outputRootDir;

	protected SelectionOperator<List<S>, S> selectionOperator;
	protected CrossoverOperator<S> crossoverOperator;
	protected MutationOperator<S> mutationOperator;
	private Termination termination;
	private Variation<S> variation;
	private InitialSolutionsCreation<S> initialSolutionsCreation;

	private Map<String, Object> algorithmStatusData;

	private Evaluation<S> evaluation;
	private Replacement<S> replacement;
	private MatingPoolSelection<S> selection;

	private long startTime;
	private long totalComputingTime;

	private Observable<Map<String, Object>> observable;

	/** Constructor */
	public HybridMoFGBMLwithNSGAII(
			/* Arguments */
			Problem<S> problem,
			int populationSize,
			int offspringPopulationSize,
			int frequency,
			String outputRootDir,
			CrossoverOperator<S> crossoverOperator,
			MutationOperator<S> mutationOperator,
			Termination termination) {
		/* Constructor Body */
		this.problem = problem;

		this.populationSize = populationSize;
		this.offspringPopulationSize = offspringPopulationSize;
		this.frequency = frequency;
		this.outputRootDir = outputRootDir;

		this.crossoverOperator = crossoverOperator;
		this.mutationOperator = mutationOperator;
		this.termination = termination;

		/* NSGA-II */
		DensityEstimator<S> densityEstimator = new CrowdingDistanceDensityEstimator<>();
		Ranking<S> ranking = new FastNonDominatedSortRanking<>();

		this.replacement =
				new RankingAndDensityEstimatorReplacement<>(
						ranking, densityEstimator, Replacement.RemovalPolicy.oneShot);

		this.variation =
				new CrossoverAndMutationAndPittsburghLearningVariation<S>(
						offspringPopulationSize, crossoverOperator, mutationOperator);

		this.selection =
				new NaryTournamentMatingPoolSelection<>(
						2,
						variation.getMatingPoolSize(),
						new MultiComparator<>(
								Arrays.asList(
										ranking.getSolutionComparator(), densityEstimator.getSolutionComparator())));

		this.initialSolutionsCreation = new RandomSolutionsCreation<S>(problem, populationSize);

		this.evaluation = new SequentialEvaluation<>();

		this.algorithmStatusData = new HashMap<>();
		this.observable = new DefaultObservable<>("Hybrid MoFGBML with NSGA-II algorithm");
	}

	@Override
	public void run() {
		startTime = System.currentTimeMillis();

		/* === START === */
		List<S> offspringPopulation;
		List<S> matingPopulation;

		/* Step 1. 初期個体群生成 - Initialization Population */
		population = createInitialPopulation();
		/* Step 2. 初期個体群評価 - Initial Population Evaluation */
		population = evaluatePopulation(population);
		/* 未勝利個体削除*/
		population = removeNoWinnerMichiganSolution(population);
		/* JMetal progress initialization */
		initProgress();

		Element population_ = XML_manager.getInstance().createElement(XML_TagName.population);
		for(S solution: this.getResult()) {
			XML_manager.getInstance().addElement(population_, solution.toElement());
		}
		Element generations_ = XML_manager.getInstance().createElement(XML_TagName.generations, XML_TagName.evaluation, String.valueOf(0));
		//knowlwdge出力用
		XML_manager.getInstance().addElement(generations_, Knowledge.getInstance().toElement());
		XML_manager.getInstance().addElement(generations_, population_);
    	XML_manager.getInstance().addElement(XML_manager.getInstance().getRoot(), generations_);

		/* GA loop */
		while(!isStoppingConditionReached()) {
			/* 親個体選択 - Mating Selection */
			matingPopulation = selection(population);
			/* 子個体群生成 - Offspring Generation */
			offspringPopulation = reproduction(matingPopulation);
			/* 子個体群評価 - Offsprign Evaluation */
			offspringPopulation = evaluatePopulation(offspringPopulation);
			/* 未勝利個体削除*/
			offspringPopulation = removeNoWinnerMichiganSolution(offspringPopulation);
			/* 個体群更新・環境選択 - Environmental Selection */
			population = replacement(population, offspringPopulation);

			/* JMetal progress update */
			updateProgress();
		}

		/* ===  END  === */
		totalComputingTime = System.currentTimeMillis() - startTime;
	}

	@Override
	protected void initProgress() {
		evaluations = populationSize;

	    algorithmStatusData.put("EVALUATIONS", evaluations);
	    algorithmStatusData.put("POPULATION", population);
	    algorithmStatusData.put("COMPUTING_TIME", System.currentTimeMillis() - startTime);

	    observable.setChanged();
	    observable.notifyObservers(algorithmStatusData);
	}


	@Override
	protected void updateProgress() {
		evaluations += offspringPopulationSize;
	    algorithmStatusData.put("EVALUATIONS", evaluations);
	    algorithmStatusData.put("POPULATION", population);
	    algorithmStatusData.put("COMPUTING_TIME", System.currentTimeMillis() - startTime);

	    observable.setChanged();
	    observable.notifyObservers(algorithmStatusData);

	    String sep = File.separator;
	    Integer evaluations = (Integer)algorithmStatusData.get("EVALUATIONS");
	    if(evaluations != null) {
	    	if(evaluations * 10 % frequency == 0 && evaluations % frequency != 0) System.out.print(". ");
	    	if(evaluations % frequency == 0) {
	    		System.out.print(" ->");
	    		for(int i=0; i<getPopulation().get(0).getNumberOfObjectives(); i++) {
	    			double tmp=0;
	    			for(int j=0; j<getPopulation().size(); j++) {
	    				tmp += getPopulation().get(j).getObjective(i);
	    			}
	    			tmp /= getPopulation().size();
		    		System.out.print(String.format("objectives[%d]: %.8f.. ", i, tmp));
	    		}
	    		System.out.println(); System.out.println();

	    	    /*new PittsburghSolutionListOutput((List<PittsburghSolution<?>>) this.getResult())
	            .setVarFileOutputContext(new DefaultFileOutputContext(outputRootDir + sep + String.format("VAR-%010d.csv", evaluations), ","))
	            .setFunFileOutputContext(new DefaultFileOutputContext(outputRootDir + sep + String.format("FUN-%010d.csv", evaluations), ","))
	            .print();*/

    	        new PittsburghSolutionListOutput((List<PittsburghSolution<?>>) this.getResult())
                .setVarFileOutputContext(new DefaultFileOutputContext(outputRootDir + sep + String.format("VAR-%d.csv", evaluations), ","))
                .setFunFileOutputContext(new DefaultFileOutputContext(outputRootDir + sep + String.format("FUN-%d.csv", evaluations), ","))
                .print();

	    		Element population = XML_manager.getInstance().createElement(XML_TagName.population);

	    		for(S solution: this.getResult()) {
	    			Element pittsburghSolution = solution.toElement();
	    			XML_manager.getInstance().addElement(population, pittsburghSolution);
	    		}

	    		Element generations = XML_manager.getInstance().createElement(XML_TagName.generations, XML_TagName.evaluation, String.valueOf(evaluations));

	    		//knowlwdge出力用
	    		XML_manager.getInstance().addElement(generations, Knowledge.getInstance().toElement());
	    		XML_manager.getInstance().addElement(generations, population);
		    	XML_manager.getInstance().addElement(XML_manager.getInstance().getRoot(), generations);
	    	}
	    }
		else {
			JMetalLogger.logger.warning(getClass().getName()
			+ ": The algorithm has not registered yet any info related to the EVALUATIONS key");
		}
	}

	@Override
	protected boolean isStoppingConditionReached() {
		return termination.isMet(algorithmStatusData);
	}

	@Override
	protected List<S> createInitialPopulation() {
		return initialSolutionsCreation.create();
	}

	@Override
	protected List<S> evaluatePopulation(List<S> population) {
		return  evaluation.evaluate(population, getProblem());
	}

	@Override
	protected List<S> selection(List<S> population) {
		return this.selection.select(population);
	}

	@Override
	protected List<S> reproduction(List<S> matingPool){
		return variation.variate(population, matingPool);
	}

	@Override
	protected List<S> replacement(List<S> population, List<S> offspringPopulation) {
		return replacement.replace(population, offspringPopulation);
	}

	protected List<S> removeNoWinnerMichiganSolution(List<S> population) {
		/* 未勝利個体削除*/
	    IntStream.range(0, population.size())
	        .forEach(i -> ((AbstractPittsburghFGBML)problem).removeNoWinnerMichiganSolution(population.get(i)));
		return population;
	}

	@Override
	public List<S> getResult(){
		return SolutionListUtils.getNonDominatedSolutions(getPopulation());
	}

	@Override
	public String getName() {
		return "Hybrid-style Multi-objective FGBML with NSGA-II";
	}

	@Override
	public String getDescription() {
		return "Hybrid-style Multi-objective Fuzzy Genetics-Based Machine Learning with NSGA-II";
	}

	public Map<String, Object> getAlgorithmStatusData() {
		return algorithmStatusData;
	}

	@Override
	public Observable<Map<String, Object>> getObservable() {
		return observable;
	}

	public long getTotalComputingTime() {
		return totalComputingTime;
	}

	public long getEvaluations() {
		return evaluations;
	}

}
