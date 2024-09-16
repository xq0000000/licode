package com.wr.Mouth8;

import com.wr.base.TreeNode;

import java.util.LinkedList;
import java.util.Queue;

/**
 * ClassName: BinaryTreeBuilder
 * Description:
 * date: 2024/8/12 10:10
 *
 * @author Wang
 * @since JDK 1.8
 */
public class BinaryTreeBuilder {
    public static TreeNode arrayToBinaryTree(Integer[] arr) {
        if (arr == null || arr.length == 0) {
            return null;
        }

        TreeNode root = new TreeNode(arr[0]);
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int index = 1;

        while (index < arr.length) {
            TreeNode current = queue.poll();

            if (index < arr.length && arr[index] != null) {
                current.left = new TreeNode(arr[index]);
                queue.add(current.left);
            }
            index ++;
            if (index < arr.length && arr[index] != null) {
                current.right = new TreeNode(arr[index]);
                queue.add(current.right);
            }
            index ++;
        }
        return root;
    }

    public static void main(String[] args) {
        Integer[] arr = {3, 9, 20, null, null, 15, 7};
        TreeNode root = arrayToBinaryTree(arr);

        // Optional: Print the tree to verify
        printTree(root, "", true);
    }

    // Utility function to print the tree
    private static void printTree(TreeNode node, String indent, boolean last) {
        if (node != null) {
            System.out.print(indent);
            if (last) {
                System.out.print("R----");
                indent += "   ";
            } else {
                System.out.print("L----");
                indent += "|  ";
            }
            System.out.println(node.val);

            printTree(node.left, indent, false);
            printTree(node.right, indent, true);
        }
    }
}
