package com.wr.Day3;

public class Day1 {
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    class Solution {
        public TreeNode insertIntoMaxTree(TreeNode root, int val) {
            if(root==null){
                return new TreeNode(val,null,null);
            }
            if(root.val>val){
                return new TreeNode(val,null,root);
            }
            else{
                root.left = insertIntoMaxTree(root.left,val);
            }
            return root;
        }
    }
}
