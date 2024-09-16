package com.wr.Day5_30;

import java.util.ArrayList;
import java.util.List;

public class Solution {
    public List<TreeNode> delNodes(TreeNode root, int[] to_delete) {
        List<TreeNode> treeNodes = new ArrayList<>();
        if(root == null){
            return null;
        }
        for (int j = 0; j < to_delete.length; j++) {
            if(root.left.val == to_delete.length){
                break;
            }
            treeNodes.add(root.left);
            if(root.right.val == to_delete.length){
                break;
            }
            treeNodes.add(root.right);
        }
        delNodes(root.left,to_delete);
        delNodes(root.right,to_delete);
        return treeNodes;
    }

    void preOrder(TreeNode  root) {
        if(root == null){
            return;
        }
        preOrder(root.left);
        preOrder(root.right);
    }

}