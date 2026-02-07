# MoFGBML_Basic
**本リポジトリについて**

AIが金融や医療診断といった高リスクの意思決定に利用される際，AIの信頼性を保証するために透明性が求められます．特に，本質的に解釈可能なモデルは，内部機構が人間に理解可能なため，説明責任が求められる分野において有用です．ファジィシステムは，現実世界の不確実性を考慮した柔軟な意思決定を可能にする代表的な本質的に解釈可能なモデルです．多目的ファジィ遺伝的機械学習（Multi-objective Fuzzy Genetics-Based Machine Learning: MoFGBML）は，進化型多目的最適化アルゴリズムを用いることで，複数の目的間のトレードオフを考慮した多数のファジィ識別器を生成します．本リポジトリでは，精度と解釈可能性を同時に最適化するMoFGBMLを実行することができます．精度と公平性を同時に最適化するMoFGBMLについては，以下のリポジトリを利用してください．

# https://github.com/TakeruKonishi/MoFGBML_Fairness

**依存関係について**  

jMetal等のライブラリを使用しているため，ソースコードだけでは，依存関係でエラーが起こるかもしれません．Mavenというビルドツールを使っているので，依存関係のエラーを解決する手間が省けていると思います．


**簡単な使い方** 

本リポジトリはMavenプロジェクトになっているので，EclipseなどのIDEを使って開いてください．pom.xmlに依存関係を定義しているので，Eclipseであれば，`pom.xmlを右クリック → 実行 → 3 Maven install`を実行することで，依存関係のエラーが解決できます．


**実行可能JARファイルの生成**

まず，pom.xmlのJARファイル名，main関数を指定してください．次に，`pom.xmlを右クリック → 実行 → 6 Maven build`を実行し，ゴールにpackageを指定して実行してください．その後，targetディレクトリ内に実行可能JARファイルとその他必要な依存関係ライブラリが生成されるので，適宜実験を行ってください．


**その他**  

適宜データセットを追加，constsを変更して使用してください．

# MoFGBML_Basic
**About this repository**

When AI is used for high-risk decisions such as finance and medical diagnosis, transparency is required to ensure its reliability. In particular, inherently interpretable models are useful where accountability is demanded, as their internal mechanisms are understandable to humans. Fuzzy systems are representative inherently interpretable models that can make flexible decisions considering real-world uncertainties. Multi-objective Fuzzy Genetics-Based Machine Learning (MoFGBML) generates a number of fuzzy classifiers considering trade-offs among multiple objectives by using an evolutionary multi-objective optimization algorithm. Using this repository, you can perform MoFGBML that simultaneously optimizes accuracy and interpretability. For MoFGBML that simultaneously optimizes accuracy and fairness, please use the following repository.

# https://github.com/TakeruKonishi/MoFGBML_Fairness

**About dependencies**

Since we are using libraries such as jMetal, dependency errors may occur in the source code alone. We use a build tool called Maven, which saves us the trouble of resolving dependency errors.


**Simple usage**  

Open this repository as a Maven project using an IDE such as Eclipse. Since the dependencies are defined in pom.xml, you can resolve the dependency errors by `right-clicking pom.xml → Run → 3 Maven install` if you are using Eclipse.


**Generation of executable JAR file**  

First, specify the JAR file name and main function in pom.xml'. Next, `right-click pom.xml → Run → 6 Maven build` and specify package as the goal. This will generate an executable JAR file and other necessary dependency libraries in the target directory. Please conduct your experiments.


**Other**  

Please add datasets and change consts as necessary.
