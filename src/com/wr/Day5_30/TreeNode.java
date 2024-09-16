package com.wr.Day5_30;

import java.util.ArrayDeque;
import java.util.Queue;

public class TreeNode {

    public Integer val;
    public TreeNode left;
    public TreeNode right;

    public TreeNode() {
    }

    TreeNode(Integer val) {
        this.val = val;
    }

    TreeNode(Integer val, TreeNode left, TreeNode right) {
        this.val = val;
        this.left = left;
        this.right = right;
    }

    @Override
    public String toString() {
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.offer(this);
        StringBuilder res = new StringBuilder("[");
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.val == null) {
                res.append("null").append(", ");
                continue;
            }
            res.append(node.val).append(", ");
            if (node.left == null) {
                queue.offer(new TreeNode(null));
            } else {
                queue.offer(node.left);
            }
            if (node.right == null) {
                queue.offer(new TreeNode(null));
            } else {
                queue.offer(node.right);
            }
        }
        res.append("]");
        return res.toString();
    }

}