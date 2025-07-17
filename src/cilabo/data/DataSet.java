package cilabo.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import cilabo.data.pattern.Pattern;

/**データセット用のデータコンテナクラス．<br>
 * Patternクラスを配列として持つ．
 * @param <pattern> データセットが扱うパターンクラスの型
 * @author Takigawa Hiroki
 */
public class DataSet<T extends Pattern<?>> {
	/** Number of Patterns*/
	private int DataSize;
	/** Number of Features*/
	private int Ndim;
	/** Number of Classes*/
	private int Cnum;
	/** Density Count of Patterns*/

	/**	データセットのPattern実装クラスの可変長配列 */
	private T[] patterns;

	/** コンストラクタ
	 * @param dataSize データセットのパターン数
	 * @param ndim 属性数．次元数
	 * @param cnum 結論部クラスのラベル種類数
	 */
	public DataSet(int dataSize, int ndim, int cnum) {
		if(dataSize <= 0 || ndim <= 0 || cnum <= 0) {
			System.err.println("incorect input data set information @" + this.getClass().getSimpleName());
		}
		DataSize = dataSize;
		Ndim = ndim;
		Cnum = cnum;
		final T[] tempPatterns = (T[]) new Pattern[dataSize];
		this.patterns = tempPatterns;
	}
	private int currentPatternIndex = 0;

	/** このインスタンスが持つリストの最後に，指定されたパターン実装クラスを追加します。<br>
	 * Appends pattern class to the end of the list that this instance has
	 * @param pattern リストに追加されるパターン実装クラス．pattern class to be appended to the list */
    public void addPattern(T pattern) {
        // 内部実装が変更された部分
        this.patterns[currentPatternIndex] = pattern; // 配列に直接代入
        currentPatternIndex++;
    }

	/**
	 * このインスタンスが持つリスト内の指定された位置にあるパターン実装クラスを返します。<br>
	 * Returns pattern class at the specified position in the list that this instance has.
	 * @param index 返されるパターン実装クラスのインデックス．index of pattern class to return
	 * @return リスト内の指定された位置にあるパターン実装クラス．pattern class at the specified position in the list
	 */
	public T getPattern(int index) {
		return this.patterns[index];
	}

	/**
	 * このインスタンスが持つリスト内の指定された位置にあるパターン実装クラスを返します。(並列処理用)<br>
	 * Returns pattern class at the specified position in the list that this instance has.(method for parallel processing)
	 * @param index 返されるパターン実装クラスのインデックス．index of pattern class to return
	 * @return リスト内の指定された位置にあるパターン実装クラス．pattern class at the specified position in the list
	 */
	// Option A: IDとインデックスが一致するなら、既存のgetPatternを呼び出す
	public T getPatternWithID(int index) { // indexをIDと解釈
	    // getPattern が既に範囲チェックをしているはず
	    return this.getPattern(index);
	}

	/** このインスタンスが持つパターン実装クラスのリストを返します。<br>
	 * Returns pattern class list that this instance has.
	 * @return 返されるパターン実装クラスのリスト．list of pattern class to return
	 */
	public List<T> getPatterns(){
	    // 配列をリストに変換して返す
	    return Arrays.asList(this.patterns); // ★ 配列を List に変換
	    // もし変更可能なリストが必要なら：
	    // return new ArrayList<>(Arrays.asList(this.patterns));
	}

	@Override
	public String toString() {
		// this.patterns は T[] なので、.size() や .get() は使えない
	    // 代わりに .length や [] アクセス、または Arrays.toString() を使う
	    if(this.patterns.length == 0) { // ★ .size() を .length に変更
	        return null; // または "Empty DataSet" などの文字列
	    }
	    String ln = System.lineSeparator();
	    // Header
	    // DataSize はフィールドから取得
	    String str = String.format("%d,%d,%d" + ln, this.DataSize, this.Ndim, this.Cnum); // ★ DataSize は this.dataSize に変更
	    // Patterns
	    for(int n = 0; n < this.patterns.length; n++) { // ★ .size() を .length に変更
	        Pattern<?> pattern = this.patterns[n]; // ★ .get(n) を [] アクセスに変更
	        str += pattern.toString() + ln;
	    }
	    return str;
	}

	/*Rowanがくれたデータセット分割メソッド*/
	/*public DataSet<pattern> split(double split_rate){
        Random randgen = new Random();
        DataSet<pattern> subDataSet = new DataSet<>((int) Math.ceil(this.DataSize*split_rate),this.Ndim, this.Cnum);
        while(this.patterns.size()>(1-split_rate)*this.DataSize){
            int idx = randgen.nextInt(this.patterns.size());
            subDataSet.addPattern(this.patterns.get(idx));
            this.patterns.remove(idx);
        }
        this.DataSize = this.patterns.size();
        return subDataSet;
    }*/

	/**
	 * このインスタンスが持つ属性数．次元を返します。
	 * @return 返される属性数．次元
	 */
	public int getNdim() {
		return this.Ndim;
	}

	/**
	 * このインスタンスが持つクラスラベル数を返します。
	 * @return 返されるクラスラベル数
	 */
	public int getCnum() {
		return this.Cnum;
	}

	/**
	 * このインスタンスが持つデータセットのパターン数を返します。
	 * @return 返されるデータセットのパターン数
	 */
	public int getDataSize() {
		return this.DataSize;
	}



	//並列分散実装用 (ver. 21以下)
	// ************************************************************
	// Fields

//		int setting = 0;
//		InetSocketAddress[] serverList = null;

	// ************************************************************
	// Constructor

	//並列分散実装用 (ver. 21以下)
//		public DataSetInfo(int Datasize, int Ndim, int Cnum, int setting, InetSocketAddress[] serverList){
//			this.DataSize = Datasize;
//			this.Ndim = Ndim;
//
//			this.setting = setting;
//			this.serverList = serverList;
//		}
//
//		public DataSetInfo(int Ndim, int Cnum, int DataSize, ArrayList<Pattern> patterns) {
//			this.Ndim = Ndim;
//			this.DataSize = DataSize;
//			this.patterns = patterns;
//		}

	// ************************************************************
	// Methods

//		public int getSetting() {
//			return this.setting;
//		}
//
//		public void setSetting(int setting) {
//			this.setting = setting;
//		}
//
//		public InetSocketAddress[] getServerList() {
//			return this.serverList;
//		}
//
//		public void setServerList(InetSocketAddress[] serverList) {
//			this.serverList = serverList;
//		}

}
