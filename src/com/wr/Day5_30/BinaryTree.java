package com.wr.Day5_30;

import java.util.ArrayDeque;
import java.util.Queue;

public class BinaryTree {

    public TreeNode root;

    public BinaryTree() {
    }

    private int start = 0;

    /**
     * 通过先根遍历序列构造二叉树
     * @param val 先根遍历序列
     * @return
     * @author xujin
     * @since 2023/2/6 16:47
     */
    public BinaryTree(Integer[] val) {
        this.start = 0;
        this.root = create(val);
    }

    /**
     * 层次构造
     * @param val 数组
     * @return com.wr.Day5_30.BinaryTree
     * @author xujin
     * @since 2023/5/30 10:10
     */
    public static BinaryTree createTreeLevel(Integer[] val) {
        BinaryTree tree = new BinaryTree();
        Queue<TreeNode> queue = new ArrayDeque<>();
        int idx = 0;
        tree.root = new TreeNode(val[idx++]);
        queue.offer(tree.root);
        while (idx < val.length || queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node == null) {
                continue;
            }
            if (val[idx] == null) {
                node.left = null;
            } else {
                node.left = new TreeNode(val[idx]);
                queue.offer(node.left);
            }
            idx++;
            if (idx >= val.length) {
                break;
            }
            if (val[idx] == null) {
                node.right = null;
            } else {
                node.right = new TreeNode(val[idx]);
                queue.offer(node.right);
            }
            idx++;
        }
        return tree;
    }

    public BinaryTree(TreeNode root) {
        this.root = root;
    }

    private TreeNode create(Integer[] val) {
        if (start >= val.length) {
            return null;
        }
        if (val[start] == null) {
            start++;
            return null;
        }
        TreeNode root = new TreeNode(val[start]);
        start++;
        root.left = create(val);
        root.right = create(val);
        return root;
    }

    public void preOrder(TreeNode root) {
        if (root == null) {
            System.out.println("null");
            return;
        }
        System.out.println(root.val);
        preOrder(root.left);
        preOrder(root.right);
    }

}
