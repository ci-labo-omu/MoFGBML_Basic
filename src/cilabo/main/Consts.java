package cilabo.main;

import java.io.File;
import java.lang.reflect.Field;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Locale;
import java.util.ResourceBundle;

import org.w3c.dom.Element;

import xml.XML_TagName;
import xml.XML_manager;

/**
 * 各種定数 定義クラス
 * Consts.[変数名]でアクセス可能
 *
 * 本ソース(Consts.java)の以下で指定している値はデフォルト値
 * もし、jarエクスポート後に変更したい値が出てきた場合は、
 * consts.propertiesに変更したい変数を書けば良い
 *     例: 「WINDOWS = 1」(in consts.properties)
 * .propertiesファイルは、Consts.setConsts(String source)メソッドによって読み込まれる
 *     例: Consts.setConsts("consts");  // load consts.properties
 *
 */
public class Consts {

	//Experimental Settings *********************************
	public static int POPULATION_SIZE = 60;
	public static int OFFSPRING_POPULATION_SIZE = 60;
	public static int TERMINATE_GENERATION = 5000;
	public static int TERMINATE_EVALUATION = 180000;
	public static int OUTPUT_FREQUENCY = 6000;

	//Random Number ***************************************
	public static int RAND_SEED = 2020;

	//OS ************************************
	public static int WINDOWS = 1;	//Windows
	public static int UNIX = 0;		//Unix or Mac

	//Fuzzy Clasifier ************************************
	/** don't care適応確率を定数にするかどうか */
	public static boolean IS_PROBABILITY_DONT_CARE = false;
	/** don't careにしない条件部の数 */
	public static int ANTECEDENT_NUMBER_DO_NOT_DONT_CARE = 5;
	/** don't care適応確率 */
	public static double DONT_CARE_RT = 0.8;
	/** 初期ル―ル数 */
	public static int INITIATION_RULE_NUM = 30;
	/** 1識別器あたりの最大ルール数 */
	public static int MAX_RULE_NUM = 60;
	/** 1識別器あたりの最小ルール数 */
	public static int MIN_RULE_NUM = 1;

	//FGBML ************************************
	/** Michigan適用確率 */
	public static double MICHIGAN_OPE_RT = 0.5;	//元RULE_OPE_RT
	/** ルール入れ替え割合 */
	public static double RULE_CHANGE_RT = 0.2;
	/** Michigan交叉確率 */
	public static double MICHIGAN_CROSS_RT = 0.9;	//元RULE_CROSS_RT
	/** Pittsburgh交叉確率 */
	public static double PITTSBURGH_CROSS_RT = 0.9;	//元RULESET_CROSS_RT
	/** ファジィ度*/
	public static double FUZZY_GRADE = 1f;

	//Folders' Name ************************************
	public static String ROOTFOLDER = "results";
	public static String ALGORITHM_ID_DIR = "ALGORITHM_ID";
	public static String EXPERIMENT_ID_DIR = "EXPERIMENT_ID";

	//Index ************************************
	/** 学習用データ */
	public static int TRAIN = 0;
	/** 評価用データ */
	public static int TEST = 1;

	public static final String XML_FILE_NAME = "results_XML";
	/******************************************/

	//dataset info *****************************
	public static int DATA_SIZE = 0;

	public static int ATTRIBUTE_NUMBER = 0;

	public static int CLASS_LABEL_NUMBER = 0;

	public static void set(String source) {
		String dir = "./";
		URLClassLoader urlLoader = null;
		ResourceBundle bundle = null;
		try {
			urlLoader = new URLClassLoader(new URL[] {new File(dir).toURI().toURL()});
			bundle = ResourceBundle.getBundle(source, Locale.getDefault(), urlLoader);
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}

		if(bundle.containsKey("POPULATION_SIZE")) { POPULATION_SIZE = Integer.parseInt(bundle.getString("POPULATION_SIZE")); }
		if(bundle.containsKey("OFFSPRING_POPULATION_SIZE")) { OFFSPRING_POPULATION_SIZE = Integer.parseInt(bundle.getString("OFFSPRING_POPULATION_SIZE")); }
		if(bundle.containsKey("TERMINATE_GENERATION")) { TERMINATE_GENERATION = Integer.parseInt(bundle.getString("TERMINATE_GENERATION")); }
		if(bundle.containsKey("TERMINATE_EVALUATION")) { TERMINATE_EVALUATION = Integer.parseInt(bundle.getString("TERMINATE_EVALUATION")); }
		if(bundle.containsKey("OUTPUT_FREQUENCY")) { OUTPUT_FREQUENCY = Integer.parseInt(bundle.getString("OUTPUT_FREQUENCY")); }

		if(bundle.containsKey("RAND_SEED")) { RAND_SEED = Integer.parseInt(bundle.getString("RAND_SEED")); }
		if(bundle.containsKey("WINDOWS")) { WINDOWS = Integer.parseInt(bundle.getString("WINDOWS")); }
		if(bundle.containsKey("UNIX")) { UNIX = Integer.parseInt(bundle.getString("UNIX")); }
		if(bundle.containsKey("IS_PROBABILITY_DONT_CARE")) { IS_PROBABILITY_DONT_CARE = Boolean.parseBoolean(bundle.getString("IS_PROBABILITY_DONT_CARE")); }
		if(bundle.containsKey("ANTECEDENT_LEN")) { ANTECEDENT_NUMBER_DO_NOT_DONT_CARE = Integer.parseInt(bundle.getString("ANTECEDENT_LEN")); }
		if(bundle.containsKey("DONT_CARE_RT")) { DONT_CARE_RT = Double.parseDouble(bundle.getString("DONT_CARE_RT")); }
		if(bundle.containsKey("INITIATION_RULE_NUM")) { INITIATION_RULE_NUM = Integer.parseInt(bundle.getString("INITIATION_RULE_NUM")); }
		if(bundle.containsKey("MAX_RULE_NUM")) { MAX_RULE_NUM = Integer.parseInt(bundle.getString("MAX_RULE_NUM")); }
		if(bundle.containsKey("MIN_RULE_NUM")) { MIN_RULE_NUM = Integer.parseInt(bundle.getString("MIN_RULE_NUM")); }
		if(bundle.containsKey("MICHIGAN_OPE_RT")) { MICHIGAN_OPE_RT = Double.parseDouble(bundle.getString("MICHIGAN_OPE_RT")); }
		if(bundle.containsKey("RULE_CHANGE_RT")) { RULE_CHANGE_RT = Double.parseDouble(bundle.getString("RULE_CHANGE_RT")); }
		if(bundle.containsKey("MICHIGAN_CROSS_RT")) { MICHIGAN_CROSS_RT = Double.parseDouble(bundle.getString("MICHIGAN_CROSS_RT")); }
		if(bundle.containsKey("PITTSBURGH_CROSS_RT")) { PITTSBURGH_CROSS_RT = Double.parseDouble(bundle.getString("PITTSBURGH_CROSS_RT")); }
		if(bundle.containsKey("FUZZY_GRADE")) { FUZZY_GRADE = Double.parseDouble(bundle.getString("FUZZY_GRADE")); }
		if(bundle.containsKey("TRAIN")) { TRAIN = Integer.parseInt(bundle.getString("TRAIN")); }
		if(bundle.containsKey("TEST")) { TEST = Integer.parseInt(bundle.getString("TEST")); }
		if(bundle.containsKey("ROOTFOLDER")) { ROOTFOLDER = bundle.getString("ROOTFOLDER"); }
		if(bundle.containsKey("ALGORITHM_ID_DIR")) { ALGORITHM_ID_DIR = bundle.getString("ALGORITHM_ID_DIR"); }
		if(bundle.containsKey("DATA_SIZE")) { DATA_SIZE = Integer.parseInt(bundle.getString("DATA_SIZE")); }
		if(bundle.containsKey("ATTRIBUTE_NUMBER")) { ATTRIBUTE_NUMBER = Integer.parseInt(bundle.getString("ATTRIBUTE_NUMBER")); }
		if(bundle.containsKey("CLASS_LABEL_NUMBER")) { CLASS_LABEL_NUMBER = Integer.parseInt(bundle.getString("CLASS_LABEL_NUMBER")); }

		bundle = null;
	}

	public static String getString() {
		Consts consts = new Consts();
		StringBuilder sb = new StringBuilder();
		String ln = System.lineSeparator();
		sb.append("Class: " + consts.getClass().getCanonicalName() + ln);
		sb.append("Consts: " + ln);
		for(Field field : consts.getClass().getDeclaredFields()) {
			try {
				field.setAccessible(true);
				sb.append( field.getName() + " = " + field.get(consts) + ln );
			}
			catch(IllegalAccessException e) {
				sb.append(field.getName() + " = " + "access denied" + ln);
			}
		}
		return sb.toString();
	}

	/**
	 * Constsを構成する Elementを返す
	 *
	 * @param xml_manager
	 * @return
	 */
	public static Element toElement() {
		Consts consts = new Consts();
		Element constsElement = XML_manager.getInstance().createElement(XML_TagName.consts);
		for(Field field : consts.getClass().getDeclaredFields()) {
			if(!field.getType().isArray()) {
				Element fieldElement;
				try {
					fieldElement = XML_manager.getInstance().createElement(field.getName(), field.get(consts).toString());
					XML_manager.getInstance().addElement(constsElement, fieldElement);
				} catch (IllegalArgumentException | IllegalAccessException e) {
					e.printStackTrace();
				}
			}
		}
		return constsElement;
	}

}
