package cilabo.hca;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import cilabo.art.HCAplusNet;

public class HCAplusManager {

    /**
     * HCA+ツリーから葉ノードを再帰的に抽出し、結果をHCAplusNet形式で返します (MATLAB: HCAplus_GetLEAVESnet_Classification.m)。
     * * @param net 現在のノード (HCAplusNet)
     * @param maxLevelRef 最大レベルを保持・更新する配列 (参照渡しとして機能)
     * @return 葉ノード情報を持つHCAplusNetオブジェクト
     */
	public HCAplusNet getLeavesNet(HCAplusNet net, int[] maxLevelRef) {
	    
	    // 葉ノードの情報を収集するためのリスト
	    List<double[]> newMeans = new ArrayList<>();
	    List<int[]> newCL = new ArrayList<>();
	    List<Double> newAdaptiveSig = new ArrayList<>();

	    // 1. ノードがない場合は、空のHCAplusNetを返す（末端のチェック）
	    if (net == null || net.numNodes == 0) {
	        return new HCAplusNet(0, 0, 0); // 初期値は適当なもので構いません
	    }
	    
	    // 最大レベルの更新
	    if (net.level > maxLevelRef[0]) {
	        maxLevelRef[0] = net.level;
	    }

	    // 2. ツリーを走査しながら葉ノードを収集
	    
	    // net.children のサイズは net.numNodes と一致しているとは限らない
	    // MATLABのロジック: netの各ノードについて、子が存在するかをチェックする
	    for (int nodeIndex = 0; nodeIndex < net.numNodes; nodeIndex++) {
	        
	        HCAplusNet childModel = null;
	        if (nodeIndex < net.children.size()) {
	            childModel = net.children.get(nodeIndex);
	        }

	        if (childModel != null) {
	            // 2a. 子ノードが存在する場合 (非葉ノード): 再帰
	            
	            HCAplusNet leavesnetChild = getLeavesNet(childModel, maxLevelRef);
	            
	            // 結果の結合
	            if (leavesnetChild != null && leavesnetChild.numNodes > 0) {
	                newMeans.addAll(leavesnetChild.weights);
	                newCL.addAll(leavesnetChild.countLabelMatrix);
	                newAdaptiveSig.addAll(leavesnetChild.adaptiveSigs);
	            }

	        } else {
	            // 2b. 子ノードが存在しない場合 (葉ノード): 自身の情報を収集
	            
	            // 💡 修正点: データ収集前にリストの有効性を再確認する
	            if (nodeIndex < net.weights.size() && nodeIndex < net.countLabelMatrix.size()) {
	                // nodeIndex は有効なインデックス。親の情報を収集する。
	                newMeans.add(net.weights.get(nodeIndex)); // <-- この行が64行目付近と推定される
	                newCL.add(net.countLabelMatrix.get(nodeIndex));
	                newAdaptiveSig.add(net.adaptiveSigs.get(nodeIndex));
	            }

	            if (net.level > maxLevelRef[0]) {
	                maxLevelRef[0] = net.level;
	            }
	        }
	    }

	    // 3. 収集結果のパッケージング
	    if (newMeans.isEmpty()) {
	        // 葉ノードが存在しない場合、空のHCAplusNetを返す
	        return new HCAplusNet(net.lambda, net.minCIM, net.maxLevel);
	    }

	    HCAplusNet leavesnet = new HCAplusNet(net.lambda, net.minCIM, net.maxLevel);
	    leavesnet.weights = newMeans;
	    leavesnet.countLabelMatrix = newCL;
	    leavesnet.adaptiveSigs = newAdaptiveSig;
	    leavesnet.numNodes = newMeans.size();
	    
	    return leavesnet;
	}
    
    //----------------------------------------------------------------------
    
    /**
     * HCA+ツリー内の全ノード数をカウントします (MATLAB: CountNumNodes.m)。
     * * @param net ルートネットワークモデル
     * @return ツリー内のノード総数
     */
    public int countAllNodes(HCAplusNet trainedNet) {
        if (trainedNet == null || trainedNet.numNodes == 0) return 0;

        int totalNodes = 0;
        List<HCAplusNet> queue = new ArrayList<>();
        queue.add(trainedNet);
        
        while (!queue.isEmpty()) {
            HCAplusNet currentModel = queue.remove(0);
            
            // currentModel内のノード数 (MATLABではisfiniteチェックがあるが、Javaでは全て有効と仮定)
            totalNodes += currentModel.numNodes;
            
            // 子ノードをキューに追加
            for (HCAplusNet child : currentModel.children) {
                if (child != null) {
                    queue.add(child);
                }
            }
            // MATLABのCountNumNodes.mは、各ノードの`Child`の数だけエッジをカウントするロジックであり、
            // ノード数をカウントするのに適していません。
            // MATLABの出力 `[~, num_node] = size(t)` は、**エッジの数**を返します。
            // ここでは、MATLABのロジックではなく、ツリー構造を持つHCA+の「ノード総数」を正しく返すロジックを実装します。
            // **NOTE:** MATLABの`CountNumNodes.m`は、グラフのエッジ数を数えているため、
            // ノード総数 = エッジ数 + 1 (ルートノード) の近似となります。
            // ロジックを簡略化し、HCA+のノード総数をBFSで正しくカウントします。
            
            // MATLABの忠実な再現（複雑で非効率なグラフ構築を伴う）は避けるため、
            // ここでは「ツリー内のHCAplusNetオブジェクトの総数」をカウントします。
        }
        
        // ツリー内のHCAplusNetオブジェクトの総数をカウントするロジックに変更
        int netCount = 0;
        queue.add(trainedNet);
        List<HCAplusNet> allNets = new ArrayList<>();
        allNets.add(trainedNet);
        
        while (!queue.isEmpty()) {
            HCAplusNet currentModel = queue.remove(0);
            netCount++;
            for (HCAplusNet child : currentModel.children) {
                 if (child != null) {
                    queue.add(child);
                    allNets.add(child);
                 }
            }
        }
        
        // 各HCAplusNetオブジェクト内のノードの総数を加算
        totalNodes = allNets.stream().mapToInt(n -> n.numNodes).sum();
        return totalNodes;
    }
}