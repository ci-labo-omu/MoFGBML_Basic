package cilabo.fuzzy.classifier.classification.impl;

import java.util.List;

import cilabo.data.pattern.Pattern;
import cilabo.fuzzy.classifier.classification.Classification;
import cilabo.gbml.solution.michiganSolution.MichiganSolution;

/**
 * 未実装
 * (実装完了時は上のコメントを削除してください)
 * @author hiroki tagkigawa
 *
 */
@Deprecated
public final class CFmeanClassification<michiganSolution extends MichiganSolution<?>> implements Classification<michiganSolution>{

	@Override
	public michiganSolution classify(List<michiganSolution> michiganSolutionList, Pattern<?> pattern) {
		return null;
	}

	@Override
	public Classification<michiganSolution> copy() {
		return new CFmeanClassification<michiganSolution>();
	}
}
