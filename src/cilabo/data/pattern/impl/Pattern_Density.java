package cilabo.data.pattern.impl;

import cilabo.data.AttributeVector;
import cilabo.data.pattern.Pattern;
import cilabo.fuzzy.rule.consequent.classLabel.impl.ClassLabel_Basic;

/** 基本的なパターンクラス．basically pattern class<br>
 * 単一のクラスラベルと各属性値を持つ．this class has only one class label and own pattern's attributes
 * @author Takigawa Hiroki */
public final class Pattern_Density extends Pattern <ClassLabel_Basic> {

	/** Pattern_Basic コンストラクタ
	 * @param id このインスタンスに与えられたID．ID which is given to this instance
	 * @param attributeVector 属性値クラス．attribute class
	 * @param targetClass ラベルクラス．label class */
	
	public Pattern_Density(int id, AttributeVector attributeVector, ClassLabel_Basic targetClass) {
		super(id, attributeVector, targetClass);
	}

	@Override
	public String toString() {
		if(this.attributeVector == null || this.targetClass == null) { return "null"; }

		String str = String.format("[id:%d, input:{%s}, Class:%s]", this.id, this.attributeVector.toString(), this.targetClass.toString());
		return str;
	}
}
