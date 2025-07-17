// cilabo.data.pattern.impl.Pattern_WithDensity.java (新規作成)
package cilabo.data.pattern.impl;

import cilabo.data.AttributeVector;
import cilabo.data.pattern.Pattern; // 抽象クラス Pattern をインポート
import cilabo.fuzzy.rule.consequent.classLabel.impl.ClassLabel_Basic;
import java.util.Objects; // Objects.isNull のために必要

public final class PatternDensity extends Pattern<ClassLabel_Basic> {

    private final double density; // density フィールドをここで定義

    /** Pattern_WithDensity コンストラクタ
     * @param id ...
     * @param attributeVector ...
     * @param targetClass ...
     * @param density 密度情報 */
    public PatternDensity(int id, AttributeVector attributeVector, ClassLabel_Basic targetClass, double density) {
        // 親クラスの density なしコンストラクタを呼び出す
        super(id, attributeVector, targetClass);

        // density のバリデーションとフィールドへの代入はここで個別に行う
        if(density <= 0) {
            throw new IllegalArgumentException("argument [density] must be positive value @" + this.getClass().getSimpleName());
        }
        this.density = density;
    }

    /** このインスタンスが持つ密度情報を返します。
     * @return 返される密度情報．density information to return
     */
    public double getDensity() {
        return this.density;
    }

    @Override
    public String toString() {
        if(this.attributeVector == null || this.targetClass == null) { return "null"; }
        String str = String.format("[id:%d, input:{%s}, Class:%s, Density:%.2f]",
                                    this.id, this.attributeVector.toString(),
                                    this.targetClass.toString(), this.density);
        return str;
    }
}