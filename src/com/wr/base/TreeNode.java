package com.wr.base;

/**
 * ClassName: TreeNode
 * Description:
 * date: 2024/8/12 9:31
 *
 * @author Wang
 * @since JDK 1.8
 */
public class TreeNode {
      public int val;
      public TreeNode left;
      public TreeNode right;
      public TreeNode() {}
      public TreeNode(int val) { this.val = val; }
      public TreeNode(int val, TreeNode left, TreeNode right) {
          this.val = val;
          this.left = left;
          this.right = right;
      }
}
