package cilabo.data;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import cilabo.data.pattern.impl.PatternDensity;
import cilabo.data.pattern.impl.Pattern_Basic;
import cilabo.data.pattern.impl.Pattern_MultiClass;
import cilabo.fuzzy.rule.consequent.classLabel.impl.ClassLabel_Basic;
import cilabo.fuzzy.rule.consequent.classLabel.impl.ClassLabel_Multi;
import cilabo.main.Consts;
import cilabo.main.ExperienceParameter;

/**
 * データセット入力用メソッド群
 * @author Takigawa Hiroki
 */

public class InputDensity {

    /**
     * <h1>Input File for Single-Label Classification Dataset with Density Information</h1>
     * @param fileName : String
     * @return 入力済みDataSet (Pattern_WithDensity を含む)
     */
    public static DataSet<PatternDensity> inputDataSet_WithDensity(String fileName) {
        // Input クラスの共通ヘルパーメソッドを呼び出す
        List<double[]> lines = Input.inputDataAsList(fileName);

        // The first row is parameters of dataset
        DataSet<PatternDensity> data = new DataSet<PatternDensity>( // DataSet の型も Pattern_WithDensity に変更
                (int)lines.get(0)[0],
                (int)lines.get(0)[1],
                (int)lines.get(0)[2]);
        lines.remove(0);

        // Later second row are patterns
        for(int n = 0; n < data.getDataSize(); n++) {
            double[] line = lines.get(n);

            int id = n;
            double[] vector = Arrays.copyOfRange(line, 0, data.getNdim());
            Integer C = (int)line[data.getNdim()];
            double density = line[data.getNdim() + 1]; // density を double で読み込む

            AttributeVector inputVector = new AttributeVector(vector);
            ClassLabel_Basic classLabel = new ClassLabel_Basic(C);
            
            // Pattern_WithDensity のコンストラクタは density を受け取る
            PatternDensity pattern = new PatternDensity(
                    id,
                    inputVector,
                    classLabel,
                    density);
            data.addPattern(pattern);
        }
        return data;
    }
    public static void loadTrainTestFiles_Basic(String trainFile, String testFile) {

		/* Load Dataset ======================== */
		if(Objects.isNull(DataSetManager.getInstance().getTrains())) {
			throw new IllegalArgumentException("argument [trainFile] is null @" + "TrainTestDatasetManager.loadTrainTestFiles()");}
		if(Objects.isNull(DataSetManager.getInstance().getTrains())) {
			throw new IllegalArgumentException("argument [testFile] is null @" + "TrainTestDatasetManager.loadTrainTestFiles()");}

		DataSet<PatternDensity> train = InputDensity.inputDataSet_WithDensity(trainFile);
		DataSetManager.getInstance().addTrains(train);
		Consts.DATA_SIZE = train.getDataSize();
		Consts.ATTRIBUTE_NUMBER = train.getNdim();
		Consts.CLASS_LABEL_NUMBER = train.getCnum();

		DataSet<Pattern_Basic> test = Input.inputDataSet_Basic(testFile);
		DataSetManager.getInstance().addTests(test);

		if(Objects.isNull(DataSetManager.getInstance().getTrains())) {
			throw new IllegalArgumentException("failed to initialise trains@TrainTestDatasetManager.loadTrainTestFiles()");}
		else if(Objects.isNull(DataSetManager.getInstance().getTests())) {
			throw new IllegalArgumentException("failed to initialise tests@TrainTestDatasetManager.loadTrainTestFiles()");}
		return;
	}


    // loadTrainTestFiles_WithDensity のようなヘルパーメソッドも必要であればここに追加
    // public static void loadTrainTestFiles_WithDensity(String trainFile, String testFile) { ... }
}