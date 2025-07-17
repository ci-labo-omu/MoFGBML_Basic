package cilabo.data.pattern;

import java.util.Objects;

import cilabo.data.AttributeVector;
import cilabo.fuzzy.rule.consequent.classLabel.ClassLabel;

/** パターン用データコンテナクラス．abstract pattern class<br>
 * @param <ClassLabelObject> 実装したPatternクラスが持つラベルクラス
 * @author Takigawa Hiroki
 */
public abstract class Pattern <ClassLabelObject extends ClassLabel<?>>{

	/** このインスタンスに与えられたID．ID that is ginve to this instance */
	protected final int id;
	/** 属性値クラス．attribute class */
	protected final AttributeVector attributeVector;
	/** クラスラベル．class label*/
	protected final ClassLabelObject targetClass;
	/** このインスタンスが持つ密度情報 */
	//protected final int density;

	/** 入力されたデータを基にインスタンスを生成
	 * @param id このインスタンスに与えられたID．ID which is ginve to this instance
	 * @param attributeVector 属性値クラス．attribute class
	 * @param targetClass クラスラベル．class label */
	public Pattern(int id, AttributeVector attributeVector, ClassLabelObject targetClass) {
		if(id < 0) {
			throw new IllegalArgumentException("argument [id] must be positive integer @" + this.getClass().getSimpleName());}
		else if(Objects.isNull(attributeVector)) {
			throw new IllegalArgumentException("argument [inputVector] is null @" + this.getClass().getSimpleName());}
		else if(Objects.isNull(targetClass)) {
			throw new IllegalArgumentException("argument [trueClass] is null @" + this.getClass().getSimpleName());}
		
		this.id = id;
		this.attributeVector = attributeVector;
		this.targetClass = targetClass;
		//this.density = density;
	}

	/**
	 * このインスタンスが持つIDを返します。<br>
	 * Returns ID that this instance has.
	 * @return 返されるこのインスタンスに与えられたID．ID which is given to this instance to return
	 */
	public int getID() {
		return this.id;
	}

	/**
	 * このインスタンスが持つ属性値クラスを返します。<br>
	 * Returns AttributeVector that this instance has.
	 * @return 返される属性値クラス．AttributeVector to return
	 */
	public AttributeVector getAttributeVector() {
		if(Objects.isNull(attributeVector)) {System.err.print("attributeVector hasn't been initialised");}
		return this.attributeVector;
	}

	/**
	 * このインスタンスが持つ属性値配列を返します。<br>
	 * Returns Attribute array that this instance has.
	 * @return 返される属性値配列．Array of AttributeVector to return
	 */
	public double[] getAttributeArray() {
		return this.attributeVector.getAttributeArray();
	}

	/** 指定された位置にある属性値を返します。<br>
	 * Returns attribute value at the specified position.
	 * @param index 返される属性値のインデックス．index of attribute value to return
	 * @return 返される属性値．attribute value to return
	 */
	public double getAttributeValue(int index) {
		return this.attributeVector.getAttributeValue(index);
	}
	

	/**このインスタンスが持つ結論部ラベルクラスを返します。<br>
	 * Returns conclusion label class that this instance has.
	 * @return 返される結論部ラベルクラス．conclusion label class to return
	 */
	public ClassLabelObject getTargetClass() {
		return this.targetClass;
	}
}
