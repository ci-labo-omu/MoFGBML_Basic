# MoFGBML_Basic
**依存関係について**  
jMetalなどのライブラリを使用しているため，ソースコードだけでは，依存関係でエラーが起こるかもしれません．  
今回は，Mavenというビルドツールを使っているので，依存関係のエラーを解決する手間が省けていると思います．

**簡単な使い方**  
本リポジトリはMavenプロジェクトになっていると思うので，EclipseなどのIDEを使って開いてください．  
その後，pom.xmlに依存関係を定義しているので，Eclipseであれば，`pom.xmlを右クリック → 実行 → 3 Maven install`を実行することで，依存関係のエラーが解決できると思います．

**実行可能JARファイルの生成**  
まず，pom.xmlのJARファイル名，main関数を指定してください．次に，`pom.xmlを右クリック → 実行 → 6 Maven build`を実行し，ゴールにpackageを指定して実行してください．  
そうすると，targetディレクトリ内に実行可能JARファイルとその他必要な依存関係ライブラリが生成されるので，適宜実験を行ってください．

**その他**  
適宜データセットを追加，constsを変更して使用してください．

# MoFGBML_Basic
**About dependencies**
Since we are using libraries such as jMetal, dependency errors may occur in the source code alone.  
In this case, we use a build tool called Maven, which saves us the trouble of resolving dependency errors.

**Simple usage**  
Open this repository as a Maven project using an IDE such as Eclipse.  
Then, since the dependencies are defined in pom.xml, you can resolve the dependency errors by `right-clicking pom.xml → Run → 3 Maven install` if you are using Eclipse.

**Generation of executable JAR file**  
First, specify the JAR file name and main function in pom.xml'. Next, `right-click pom.xml → Run → 6 Maven build` and specify package as the goal.  
This will generate an executable JAR file and other necessary dependency libraries in the target directory. Please conduct your experiments.

**Other**  
Please add datasets and change consts as necessary.
